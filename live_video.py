import sys
sys.path.append("/home/raquel2/ZoeDepth")
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

# SCRIPT CONFIGURATION
DEFAULT_DATASET = 'nyu'
DEFAULT_PRETRAINED = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'


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


def initialize_models(yolo_weights, zoedepth_dataset, zoedepth_pretrained, device):
    # YOLO model
    yolo_model = YOLO(yolo_weights).to(device)
    # ZoeDepth model
    config = get_config('zoedepth', 'eval', zoedepth_dataset)
    config.pretrained_resource = zoedepth_pretrained
    zoe_model = build_model(config).to(device)
    zoe_model.eval()
    return yolo_model, zoe_model


def process_frame(img, yolo_model, zoe_model, dataset, K):
    orig_h, orig_w = img.shape[:2]

    #  Detect people
    results = yolo_model(img[:, :, ::-1])  # BGR->RGB
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    person_idxs = np.where(classes == 0)[0]
    if len(person_idxs) == 0:
        return img, []

    # Depth estimation
    color = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transforms.ToTensor()(color).unsqueeze(0).to(zoe_model.device)
    out = zoe_model(tensor, dataset=dataset)
    if isinstance(out, dict):
        depth_map = out.get('metric_depth', out.get('out'))
    elif isinstance(out, (list, tuple)):
        depth_map = out[-1]
    else:
        depth_map = out
    depth_map = depth_map.squeeze().detach().cpu().numpy()

    # Resize depth to original
    depth_map_orig = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Normalize for visualization
    norm = (depth_map_orig - depth_map_orig.min()) / (depth_map_orig.max() - depth_map_orig.min())
    depth_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Camera intrinsics
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
        # use 5th percentile to avoid outliers
        z = float(np.percentile(bb.flatten(), 5))
        # Calcula centro da bounding box
        uc = (x1c + x2c) // 2
        vc = (y1c + y2c) // 2
        # Converte coordenadas para espaço 3D usando intrínsecos
        X = (uc - CX) * z / FX
        Y = (vc - CY) * z / FY
        dist = math.sqrt(X**2 + Y**2 + z**2)
        distances.append((i, dist))
        # annotate
        cv2.rectangle(depth_color, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
        cv2.circle(depth_color, (uc, vc), 5, (255,0,0), -1)
        cv2.putText(depth_color, f"{dist:.2f}m", (x1c, y1c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return depth_color, distances


def main():
    parser = argparse.ArgumentParser(description="Live YOLO + ZoeDepth distance estimation")
    parser.add_argument('--gst_src', type=str, required=True, help='GStreamer pipeline for input')
    parser.add_argument('--receiver_ip', type=str, required=True, help='Receiver IP for output UDP stream')
    parser.add_argument('--dataset', type=str, choices=['nyu','kitti'], default=DEFAULT_DATASET)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument('--yolo_weights', type=str, default='best.pt')
    args = parser.parse_args()

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # models
    yolo_model, zoe_model = initialize_models(args.yolo_weights, args.dataset, args.pretrained, device)

    # load camera intrinsics
    calib = np.load("CalibrationMatrix_college_cpt.npz")
    K = calib['Camera_matrix']

    # capture input
    cap = cv2.VideoCapture(args.gst_src, cv2.CAP_GSTREAMER)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else 30

    # output streaming pipeline
    gst_out = (
        f'appsrc ! videoconvert ! nvvidconv ! nvv4l2h264enc bitrate=4000000 ! '
        f'rtph264pay config-interval=1 pt=96 ! udpsink host={args.receiver_ip} port=5000'
    )
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (w,h), True)
    if not out.isOpened():
        print("Failed to open GStreamer pipeline.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        processed_frame, distances = process_frame(frame, yolo_model, zoe_model, args.dataset, K)
        out.write(processed_frame)

        # debug print distances
        for idx, d in distances:
            print(f"Frame {frame_count}, Person {idx}: {d:.2f}m")

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
