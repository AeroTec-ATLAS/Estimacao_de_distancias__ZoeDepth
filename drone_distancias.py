import sys
sys.path.append("/home/mariquitos/Downloads/DISTANCE_ESTIMATION/zoedepth")
import os
import csv
import cv2
import torch
import numpy as np
import math
import time
from PIL import Image
import torchvision.transforms as transforms
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from ultralytics import YOLO
import argparse

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# CONFIGURAÇÕES
DEFAULT_DATASET = 'nyu'
DEFAULT_PRETRAINED = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'

# GStreamer pipeline with appsink
UDP_PIPELINE = (
    'udpsrc port=5600 caps="application/x-rtp,media=video,payload=96,encoding-name=H264" ! '
    'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
    'video/x-raw,format=BGR ! appsink name=sink sync=false max-buffers=1 drop=true'
)

csv_filename = "resultados_detecao.csv"


def initialize_models(yolo_weights, zoedepth_dataset, zoedepth_pretrained):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(yolo_weights).to(device)
    config = get_config('zoedepth', 'eval', zoedepth_dataset)
    config.pretrained_resource = zoedepth_pretrained
    zoe_model = build_model(config).to(device)
    zoe_model.eval()
    return yolo_model, zoe_model


def process_frame(img, yolo_model, zoe_model, dataset, K):
    orig_h, orig_w = img.shape[:2]
    device = next(zoe_model.parameters()).device

    results = yolo_model(img[:, :, ::-1], classes=[0])
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    person_idxs = np.where(classes == 0)[0]
    if len(person_idxs) == 0:
        return img, []

    color = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transforms.ToTensor()(color).unsqueeze(0).to(device)
    out = zoe_model(tensor, dataset=dataset)
    if isinstance(out, dict):
        depth_map = out.get('metric_depth', out.get('out'))
    elif isinstance(out, (list, tuple)):
        depth_map = out[-1]
    else:
        depth_map = out
    depth_map = depth_map.squeeze().detach().cpu().numpy()

    depth_map_orig = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    norm = (depth_map_orig - depth_map_orig.min()) / (depth_map_orig.max() - depth_map_orig.min())
    depth_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    FX, FY = K[0,0], K[1,1]
    CX = orig_w / 2
    CY = orig_h / 2

    distances = []
    for i in person_idxs:
        x1, y1, x2, y2 = map(int, boxes[i])
        x1c, x2c = max(0, x1), min(orig_w, x2)
        y1c, y2c = max(0, y1), min(orig_h, y2)
        bb = depth_map_orig[y1c:y2c, x1c:x2c]
        if bb.size == 0:
            continue
        
        # Coordenadas do fundo da bbx (pés da pessoa)
        uc = (x1c + x2c) // 2
        vc = y2c

        # Coordenadas cartesianas
        x_img = uc - CX
        y_img = vc - CY

        # Profundidade
        z = float(np.percentile(bb.flatten(), 5))

        # Projeção para coordenadas reais
        X = x_img * z / FX
        Y = y_img * z / FY

        # Distância da câmara ao ponto 3D (X, Y, Z)
        dist = math.sqrt(X**2 + Y**2 + z**2)

        # Guarda tudo
        distances.append((i, x_img, y_img, dist))

        # Visualização
        cv2.rectangle(depth_color, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        cv2.circle(depth_color, (uc, vc), 5, (255, 0, 0), -1)
        cv2.putText(depth_color, f"X={x_img:.0f}px", (x1c, y2c+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(depth_color, f"Y={y_img:.0f}px", (x1c, y2c+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(depth_color, f"D={dist:.2f}m", (x1c, y2c+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return depth_color, distances


def main():
    parser = argparse.ArgumentParser(description="YOLO + ZoeDepth (stream UDP)")
    parser.add_argument('--dataset', type=str, choices=['nyu', 'kitti'], default=DEFAULT_DATASET)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt')
    args = parser.parse_args()

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pessoa", "X_img (px)", "Y_img (px)", "Distância (m)"])


    yolo_model, zoe_model = initialize_models(args.yolo_weights, args.dataset, args.pretrained)
    calib = np.load("CalibrationMatrix_college_cpt.npz")
    K = calib['Camera_matrix']

    Gst.init(None)
    pipeline = Gst.parse_launch(UDP_PIPELINE)
    sink = pipeline.get_by_name('sink')
    pipeline.set_state(Gst.State.PLAYING)

    frame_count = 0
    SKIP_FRAMES = 3
    last_processed = None
    last_distances = []

    try:
        while True:
            sample = sink.emit("pull-sample")
            if not sample:
                print("Stream ended")
                break

            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            width = caps.get_value('width')
            height = caps.get_value('height')

            ok, map_info = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 3))
            buf.unmap(map_info)

            frame_count += 1
            if frame_count % SKIP_FRAMES == 0:
                processed_frame, last_distances = process_frame(frame, yolo_model, zoe_model, args.dataset, K)
                last_processed = processed_frame

                for idx, x_img, y_img, dist in last_distances:
                    print(f"Frame {frame_count}, Pessoa {idx}: X={x_img:.1f}px, Y={y_img:.1f}px, Dist={dist:.2f}m")
                    
                with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for idx, x_img, y_img, dist in last_distances:
                        writer.writerow([f"Pessoa {idx}", f"{x_img:.1f}", f"{y_img:.1f}", f"{dist:.2f}"])

            display_frame = last_processed if last_processed is not None else frame
            cv2.imshow("Live Depth Estimation", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
