import sys
sys.path.append("/home/raquel2/ZoeDepth")
import cv2
import torch
import numpy as np
import threading
import queue
import argparse
from PIL import Image
import torchvision.transforms as transforms
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from ultralytics import YOLO

DEFAULT_DATASET = 'nyu'
DEFAULT_PRETRAINED = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'

frame_queue = queue.Queue(maxsize=2)
processed_queue = queue.Queue(maxsize=2)

def initialize_models(yolo_weights, zoedepth_dataset, zoedepth_pretrained, device):
    # Carrega YOLO
    yolo_model = YOLO(yolo_weights).to(device)


    config = get_config('zoedepth', 'eval', zoedepth_dataset)
    config.pretrained_resource = zoedepth_pretrained

    config.use_pretrained_midas = False

    zoe_model = build_model(config).to(device)
    zoe_model.eval()
    return yolo_model, zoe_model

def process_frame(img, yolo_model, zoe_model, dataset, K):
    orig_h, orig_w = img.shape[:2]
    results = yolo_model(img[:, :, ::-1])
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    person_idxs = np.where(classes == 0)[0]
    if len(person_idxs) == 0:
        return img

    color = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transforms.ToTensor()(color).unsqueeze(0).to(zoe_model.device)
    out = zoe_model(tensor, dataset=dataset)
    depth_map = out.get('metric_depth') if isinstance(out, dict) else out[-1] if isinstance(out, (list, tuple)) else out
    depth_map = depth_map.squeeze().detach().cpu().numpy()
    depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    FX, FY = K[0, 0], K[1, 1]
    CX, CY = K[0, 2], K[1, 2]

    norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    for i in person_idxs:
        x1, y1, x2, y2 = map(int, boxes[i])
        x1c, x2c = max(0, x1), min(orig_w, x2)
        y1c, y2c = max(0, y1), min(orig_h, y2)
        bb = depth_map[y1c:y2c, x1c:x2c]
        if bb.size == 0:
            continue
        z = float(np.percentile(bb.flatten(), 5))
        uc = (x1c + x2c) // 2
        vc = (y1c + y2c) // 2
        X = (uc - CX) * z / FX
        Y = (vc - CY) * z / FY
        dist = (X**2 + Y**2 + z**2)**0.5
        cv2.rectangle(depth_color, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        cv2.putText(depth_color, f"{dist:.2f}m", (x1c, y1c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return depth_color

def capture_thread(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (160, 120))
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

def processing_thread(yolo_model, zoe_model, dataset, K):
    while True:
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        processed = process_frame(frame, yolo_model, zoe_model, dataset, K)
        try:
            processed_queue.put_nowait(processed)
        except queue.Full:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--pretrained', default=DEFAULT_PRETRAINED)
    parser.add_argument('--yolo_weights', default="yolov8n.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo em uso: {device}")

    yolo_model, zoe_model = initialize_models(args.yolo_weights, args.dataset, args.pretrained, device)
    calib = np.load("CalibrationMatrix_college_cpt.npz")
    K = calib['Camera_matrix']

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return


    threading.Thread(
        target=capture_thread,
        args=(cap,),
        daemon=True,
        name="capture_thread"
    ).start()


    num_processing_threads =1 
    for i in range(num_processing_threads):
        threading.Thread(
            target=processing_thread,
            args=(yolo_model, zoe_model, args.dataset, K),
            daemon=True,
            name=f"proc_thread_{i}"
        ).start()

    print("Pressiona 'q' para sair")
    while True:
        if not processed_queue.empty():
            img = processed_queue.get()
            cv2.imshow("Live Depth Estimation", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
