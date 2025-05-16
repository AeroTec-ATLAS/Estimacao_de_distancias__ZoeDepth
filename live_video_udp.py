import sys
sys.path.append("/home/raquel2/ZoeDepth")  # ajustar conforme necessário
from zoedepth.models.builder import build_model
import os
import math
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from zoedepth.utils.config import get_config
from ultralytics import YOLO
import argparse
import csv

# SCRIPT CONFIGURATION
DEFAULT_DATASET = 'nyu'
DEFAULT_PRETRAINED = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
CSV_LOG = 'distances_log.csv'

# Pipeline GStreamer para receber H.264 via UDP
UDP_PIPELINE = (
    "udpsrc port=5000 ! application/x-rtp,encoding-name=H264,payload=96 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
    "appsink drop=true sync=false"
)

def process_frame(img, yolo_model, zoe_model, dataset, K, ic_x, ic_y, frame_id, csv_writer):
    orig_h, orig_w = img.shape[:2]

    #  Detetar pessoas
    results = yolo_model(img[:, :, ::-1])  # BGR->RGB
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    person_idxs = np.where(classes == 0)[0]
    if len(person_idxs) == 0:
        return img

    # Depth estimation
    color = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transforms.ToTensor()(color).unsqueeze(0).to(zoe_model.device)
    out = zoe_model(tensor, dataset=dataset)
    depth_map = out.get('metric_depth') if isinstance(out, dict) else out[-1] if isinstance(out, (list, tuple)) else out
    depth_map = depth_map.squeeze().detach().cpu().numpy()

    # Resize depth to original
    depth_map_orig = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    norm = (depth_map_orig - depth_map_orig.min()) / (depth_map_orig.max() - depth_map_orig.min())
    depth_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    FX, FY = K[0,0], K[1,1]
    CX, CY = K[0,2], K[1,2]

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

        # Mundo real
        X = (uc - CX) * z / FX
        Y = (vc - CY) * z / FY
        dist = math.sqrt(X**2 + Y**2 + z**2)
        dist_plane = math.sqrt(X**2 + Y**2)

        # Pixels
        dx_pix = uc - ic_x
        dy_pix = vc - ic_y
        dist_pix = math.sqrt(dx_pix**2 + dy_pix**2)

        # Anotar imagem
        cv2.rectangle(depth_color, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
        cv2.circle(depth_color, (uc, vc), 5, (255,0,0), -1)
        cv2.circle(depth_color, (int(ic_x), int(ic_y)), 5, (0,255,255), -1)
        cv2.line(depth_color, (int(ic_x), int(ic_y)), (uc, vc), (255,255,0), 2)
        cv2.putText(depth_color, f"{dist:.2f}m", (x1c, y1c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Guardar em CSV
        csv_writer.writerow([frame_id, i, f"{dist:.2f}", int(dx_pix), int(dy_pix), f"{dist_pix:.1f}", f"{X:.2f}", f"{Y:.2f}", f"{dist_plane:.2f}"])

    return depth_color

def main():
    parser = argparse.ArgumentParser(description="Live YOLO + ZoeDepth distance estimation")
    parser.add_argument('--gst_src', type=str, required=True, help='GStreamer pipeline para input')
    parser.add_argument('--dataset', type=str, choices=['nyu','kitti'], default=DEFAULT_DATASET)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    args = parser.parse_args()

    yolo_model = YOLO("yolov8n.pt")
    config = get_config('zoedepth', 'eval', args.dataset)
    config.pretrained_resource = args.pretrained
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zoe_model = build_model(config).to(device)
    zoe_model.eval()

    calib = np.load("CalibrationMatrix_college_cpt.npz")
    K = calib['Camera_matrix']

    cap = cv2.VideoCapture(args.gst_src, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o stream UDP.")

    orig_w, orig_h = 640, 480  # fallback caso o 1º frame falhe
    ic_x, ic_y = orig_w // 2, orig_h // 2

    with open(CSV_LOG, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['frame_id', 'person_id', 'dist_3d_m', 'dx_pix', 'dy_pix', 'dist_pix', 'X_m', 'Y_m', 'dist_plane_m'])

        frame_id = 0
        while True:
            ret, img_bgr = cap.read()
            if not ret or img_bgr is None:
                print("Frame inválido - tentando novamente...")
                continue

            orig_h, orig_w = img_bgr.shape[:2]
            ic_x, ic_y = orig_w // 2, orig_h // 2
            frame_id += 1
            
            vis_img = process_frame(img_bgr, yolo_model, zoe_model, args.dataset, K, ic_x, ic_y, frame_id, csv_writer)
            cv2.imshow("Depth Visualization", vis_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
