import sys
sys.path.append("/home/raquel2/ZoeDepth")  ##mudar caminho
import os
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

# CONFIGURAÇÕES
DEFAULT_DATASET = 'nyu'
DEFAULT_PRETRAINED = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt' #mudar se for preciso

# Pipeline GStreamer para receber H.264 via UDP
UDP_PIPELINE = (
    "udpsrc port=5600 ! application/x-rtp,encoding-name=H264,payload=96 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
)

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def initialize_models(yolo_weights, zoedepth_dataset, zoedepth_pretrained):
    # YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(yolo_weights).to(device)
    # ZoeDepth
    config = get_config('zoedepth', 'eval', zoedepth_dataset)
    config.pretrained_resource = zoedepth_pretrained
    zoe_model = build_model(config).to(device)
    zoe_model.eval()
    return yolo_model, zoe_model

def process_frame(img, yolo_model, zoe_model, dataset, K):
    orig_h, orig_w = img.shape[:2]
    device = next(zoe_model.parameters()).device
    # Deteção de pessoas
    results = yolo_model(img[:, :, ::-1], classes=[0])  # só deteta pessoas # BGR->RGB
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    person_idxs = np.where(classes == 0)[0]
    if len(person_idxs) == 0:
        return img, []

    # Estimação de profundidade
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

    # Matriz da câmara (vem do código de calibração)
    FX, FY = K[0,0], K[1,1]
    CX, CY = K[0,2], K[1,2]

    distances = []
    for i in person_idxs:
        x1, y1, x2, y2 = map(int, boxes[i])
        x1c, x2c = max(0, x1), min(orig_w, x2)
        y1c, y2c = max(0, y1), min(orig_h, y2)
        bb = depth_map_orig[y1c:y2c, x1c:x2c]
        if bb.size == 0:
            continue
        
        z = float(np.percentile(bb.flatten(), 5))
        uc = (x1c + x2c) // 2
        vc = (y1c + y2c) // 2
        X = (uc - CX) * z / FX
        Y = (vc - CY) * z / FY
        dist = math.sqrt(X**2 + Y**2 + z**2)
        distances.append((i, dist, X, Y, z))
 
        cv2.rectangle(depth_color, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
        cv2.circle(depth_color, (uc, vc), 5, (255,0,0), -1)
        cv2.putText(depth_color, f"{dist:.2f}m", (x1c, y1c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(depth_color, f"X={-X:.2f}m", (x1c, y2c+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(depth_color, f"Y={-Y:.2f}m", (x1c, y2c+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    return depth_color, distances

def main():
    parser = argparse.ArgumentParser(description="YOLO + ZoeDepth (webcam local)")
    parser.add_argument('--dataset', type=str, choices=['nyu','kitti'], default=DEFAULT_DATASET)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt')
    args = parser.parse_args()

    # Modelos
    yolo_model, zoe_model = initialize_models(args.yolo_weights, args.dataset, args.pretrained)

    # Load da matriz 
    calib = np.load("CalibrationMatrix_college_cpt.npz")
    K = calib['Camera_matrix']

    # Abre o stream UDP via GStreamer
    cap = cv2.VideoCapture(UDP_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Não foi possível abrir o stream UDP.")
        return

    frame_count = 0
    SKIP_FRAMES = 3
    last_processed = None
    last_distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o frame.")
            break

        frame_count += 1


        if frame_count % SKIP_FRAMES == 0:
            processed_frame, last_distances = process_frame(frame, yolo_model, zoe_model, args.dataset, K)
            last_processed = processed_frame
            for idx, d, x, y, z in last_distances:
                print(f"Frame {frame_count}, Person {idx}: {d:.2f}m (X={-x:.2f}m, Y={-y:.2f}m, Z={z:.2f}m)")


        display_frame = last_processed if last_processed is not None else frame
        cv2.imshow("Live Depth Estimation", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
