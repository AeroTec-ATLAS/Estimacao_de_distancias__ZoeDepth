import sys
sys.path.append("/home/raquel2/ZoeDepth") ##mudar caminho

from zoedepth.models.builder import build_model
import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from zoedepth.utils.config import get_config
from ultralytics import YOLO
import argparse

#CONFIGURAÇÕES
DEFAULT_DATASET = 'nyu'
DEFAULT_PRETRAINED = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt' #mudar se for preciso


def process_single_image(zoe_model, yolo_model, image_path, output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing {base}...")

    # Dar Load da imagem e detetar pessoas
    img_bgr = cv2.imread(image_path)
    orig_h, orig_w = img_bgr.shape[:2]
    # centro da imagem em pixels
    ic_x, ic_y = orig_w / 2, orig_h / 2

    results = yolo_model(img_bgr[:, :, ::-1])  # BGR → RGB
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    person_idxs = np.where(classes == 0)[0]
    if len(person_idxs) == 0:
        print("No person detected.")
        return

    # Predict depth map (ZoeDepth já faz o resize interno)
    color = Image.open(image_path).convert('RGB')
    tensor = transforms.ToTensor()(color).unsqueeze(0).to(zoe_model.device)
    out = zoe_model(tensor, dataset=dataset)
    if isinstance(out, dict):
        depth_map = out.get('metric_depth', out.get('out'))
    elif isinstance(out, (list, tuple)):
        depth_map = out[-1]
    else:
        depth_map = out
    depth_map = depth_map.squeeze().detach().cpu().numpy()  # shape: (H_d, W_d)

    # Upscale depth_map para a resolução original
    depth_map_orig = cv2.resize(
        depth_map,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    # estatísticas de profundidade dentro de cada bbox
    print("Debug depth stats por pessoa:")
    for idx in person_idxs:
        x1, y1, x2, y2 = map(int, boxes[idx])
        # garante os limites
        x1c, x2c = max(0, x1), min(orig_w, x2)
        y1c, y2c = max(0, y1), min(orig_h, y2)
        bb = depth_map_orig[y1c:y2c, x1c:x2c].flatten()
        if bb.size == 0:
            print(f" Person {idx}: bbox degenerada → "
                  f"x1={x1},x2={x2},y1={y1},y2={y2}")
        else:
            print(f" Person {idx}: bbox_pixels={bb.size}, "
                  f"min={bb.min():.2f}, 5%={np.percentile(bb,5):.2f}, "
                  f"med={np.median(bb):.2f}, 95%={np.percentile(bb,95):.2f}, "
                  f"max={bb.max():.2f}")
    print("Fim debug\n")

    # Matriz da câmara
    calib = np.load("CalibrationMatrix_college_cpt.npz")
    K = calib['Camera_matrix']
    FX, FY = K[0,0], K[1,1]
    CX, CY = K[0,2], K[1,2]

    # mapa de profundidade
    norm = (depth_map_orig - depth_map_orig.min()) / (depth_map_orig.max() - depth_map_orig.min())
    depth_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    distances = []
    dists_pixel = []
    dists_plane = []
    for i in person_idxs:
        x1, y1, x2, y2 = map(int, boxes[i])
        x1c, x2c = max(0, x1), min(orig_w, x2)
        y1c, y2c = max(0, y1), min(orig_h, y2)
        bb = depth_map_orig[y1c:y2c, x1c:x2c]
        if bb.size == 0:
            print(f" Skipping Person {i}: bbox inválida.")
            continue


        z = float(np.percentile(bb.flatten(), 5))


        uc = int((x1c + x2c) // 2)
        vc = int((y1c + y2c) // 2)
        X = (uc - CX) * z / FX
        Y = (vc - CY) * z / FY
        dist = np.sqrt(X**2 + Y**2 + z**2)
        distances.append(dist)

        # Mostra o quanto tem de se mover em cada direção (mundo real)
        print(f"Person {i}: precisa mover-se X={-X:.2f}m, Y={-Y:.2f}m para o centro da imagem")

        dist = np.sqrt(X**2 + Y**2 + z**2)
        distances.append(dist)

        # cálculo da distância em pixels até o centro da imagem
        dx_pix = uc - ic_x
        dy_pix = vc - ic_y
        dist_pix = np.sqrt(dx_pix**2 + dy_pix**2)
        dists_pixel.append(dist_pix)

        # distância real no plano (ignora Z)
        dist_plane = np.sqrt(X**2 + Y**2)
        dists_plane.append(dist_plane)

        # Desenha na imagem depth
        cv2.rectangle(depth_color, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        cv2.circle(depth_color, (uc, vc), 5, (255, 0, 0), -1)
        # marcar centro da imagem e ligar
        cv2.circle(depth_color, (int(ic_x), int(ic_y)), 5, (0, 255, 255), -1)
        cv2.line(depth_color,
                 (int(ic_x), int(ic_y)),
                 (uc, vc),
                 (255, 255, 0), 2)
        # distâncias
        cv2.putText(depth_color, f"{dist:.2f}m", (x1c, y1c - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(depth_color,
                    f"{int(dist_pix)}px",
                    (uc, vc + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,0), 1)
        cv2.putText(depth_color,
                    f"{dist_plane:.2f}m plano",
                    (x1c, y2c + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,255), 1)

    #Guarda resultados
    os.makedirs(output_dir, exist_ok=True)
    out_img = os.path.join(output_dir, base + '_depth.png')
    cv2.imwrite(out_img, depth_color)
    txt_file = os.path.join(output_dir, base + '_distances.txt')
    with open(txt_file, 'w') as f:
        for idx, (d3, dpix, dpl) in enumerate(zip(distances, dists_pixel, dists_plane), 1):
            f.write(f"Person {idx}: {d3:.2f} m | {dpix:.1f} px | {dpl:.2f} m plano\n")

    print(f"Done. Distances: {[f'{d:.2f}m' for d in distances]}")
    print(f" Pixels: {[f'{dpix:.1f}px' for dpix in dists_pixel]}")
    print(f" Planar: {[f'{dpl:.2f}m' for dpl in dists_plane]}")

def main():
    parser = argparse.ArgumentParser(
        description="ZoeDepth + YOLOv8 distance estimation for a single image"
    )
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Where to save outputs')
    parser.add_argument('--dataset', type=str, choices=['nyu','kitti'],
                        default=DEFAULT_DATASET, help='ZoeDepth dataset')
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED,
                        help='ZoeDepth pretrained weights')
    args = parser.parse_args()

    #modelos
    yolo_model = YOLO("yolov8n.pt")
    config = get_config('zoedepth', 'eval', args.dataset)
    config.pretrained_resource = args.pretrained
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zoe_model = build_model(config).to(device)
    zoe_model.eval()

    process_single_image(
        zoe_model, yolo_model,
        args.image_path, args.output_dir, args.dataset
    )

if __name__ == '__main__':
    main()
