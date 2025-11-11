import numpy as np
import torch
import cv2
import argparse
from skimage.measure import regionprops, label
from skimage.filters import sobel, laplace
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import UNetHeatmapModel

def estimate_normal_from_spot(img_spot, plot_ellipse=False):

    img_to_thresh = img_spot
    if img_to_thresh.max() <= 1.0:
        img_to_thresh = img_to_thresh * 255.0
    img_to_thresh = np.clip(img_to_thresh, 0, 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(img_to_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Контуров пятна не найдено")
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        print(f"Контур слишком короткий: ({len(largest_contour)})")
        return None, None, None

    try:
        ellipse = cv2.fitEllipse(largest_contour)
        (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse
    except cv2.error as e:
        print(f"Ошибка при построении эллипса: {e}")
        return None, None, None

    major_axis = max(axis_a, axis_b)
    minor_axis = min(axis_a, axis_b)

    aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0.0

    angle_rad = np.deg2rad(angle_deg)

    cos_theta = np.clip(aspect_ratio, 0.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = angle_rad + np.pi / 2

    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    norm_length = np.sqrt(nx**2 + ny**2 + nz**2)
    if norm_length > 0:
        nx /= norm_length
        ny /= norm_length
        nz /= norm_length
    else:
        print("Краевой случай, длина 0. Вернем (0, 0, 1)")
        return 0.0, 0.0, 1.0

    if plot_ellipse:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_spot, cmap='gray')
        plt.title("Ориганал")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(binary_mask, cmap='gray')
        cv2.ellipse(binary_mask, ellipse, color=255, thickness=1)
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f"Найденный эллипс\nОтношение осей: {aspect_ratio:.3f}, Угол: {angle_deg:.1f} градусов")
        plt.axis('off')
        plt.show()

    return nx, ny, nz

def parse_args():
    parser = argparse.ArgumentParser(description="Инференс одного изображения")
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Путь к изображению')
    parser.add_argument('--margin', type=int, default=10,
                        help='Отступ')
    parser.add_argument('--target_crop_size', type=int, default=64,
                        help='Размер кропа')
    parser.add_argument('--segmentation_threshold', type=int, default=60,
                        help='Трешхолд сегментации пятен')
    
    parser.add_argument('--device', type=str, default=None,
                        help="Девайс ('cpu', 'cuda', 'cuda:0', и т.д.)")

    return parser.parse_args()

def load_and_preprocess_image(image_path):
    img = np.array(Image.open(image_path))
    image = img.astype(np.float32)
    image = cv2.GaussianBlur(image, (5, 5), sigmaX=0, sigmaY=0)
    return image

def segment_and_find_regions(image, threshold=60):
    binary_mask = (image > threshold).astype(np.uint8)
    labeled_img = label(binary_mask)
    regions = regionprops(labeled_img)
    return regions

def extract_and_process_crop(image, region_bbox, margin=10, target_crop_size=64):
    minr, minc, maxr, maxc = region_bbox

    minr_margin = max(minr - margin, 0)
    minc_margin = max(minc - margin, 0)
    maxr_margin = min(maxr + margin, image.shape[0])
    maxc_margin = min(maxc + margin, image.shape[1])

    crop_full = image[minr_margin:maxr_margin, minc_margin:maxc_margin]

    if crop_full.shape[0] < 35 or crop_full.shape[1] < 35:
        return None

    crop_center_x = minc_margin + crop_full.shape[1] / 2.0
    crop_center_y = minr_margin + crop_full.shape[0] / 2.0

    half_crop = target_crop_size // 2
    minc_crop = int(round(crop_center_x - half_crop))
    minr_crop = int(round(crop_center_y - half_crop))
    maxc_crop = minc_crop + target_crop_size
    maxr_crop = minr_crop + target_crop_size

    crop_extract = image[minr_crop:maxr_crop, minc_crop:maxc_crop]
    crop_extract_norm = crop_extract.astype(np.float32) / 255.0

    pad_top = max(0, -minr_crop)
    pad_bottom = max(0, maxr_crop - image.shape[0])
    pad_left = max(0, -minc_crop)
    pad_right = max(0, maxc_crop - image.shape[1])

    crop_64 = np.pad(crop_extract_norm, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0.0)

    sobel_x = sobel(crop_64, axis=1)
    sobel_y = sobel(crop_64, axis=0)
    laplacian_img = laplace(crop_64)

    crop_4ch = np.stack([crop_64, sobel_x, sobel_y, laplacian_img], axis=0)

    return {
        'crop_4ch': crop_4ch,
        'crop_64_visual': crop_64,
        'minc_crop': minc_crop,
        'minr_crop': minr_crop,
        'crop_center_x': crop_center_x,
        'crop_center_y': crop_center_y
    }

def predict_centroid(model, crop_4ch_tensor, device, crop_size=64):
    model.eval()
    with torch.no_grad():
        heatmap_pred = model(crop_4ch_tensor)

    H, W = heatmap_pred.shape[-2], heatmap_pred.shape[-1]
    x_coords_norm = torch.arange(W, dtype=torch.float64, device=device) / (W - 1)
    y_coords_norm = torch.arange(H, dtype=torch.float64, device=device) / (H - 1)
    X_coords_norm, Y_coords_norm = torch.meshgrid(x_coords_norm, y_coords_norm, indexing='xy')

    heatmap_flat = heatmap_pred.squeeze(1).flatten(start_dim=1).double()
    X_flat_norm = X_coords_norm.flatten().double()
    Y_flat_norm = Y_coords_norm.flatten().double()

    x_pred_centroid_norm = torch.sum(heatmap_flat * X_flat_norm, dim=1).item() 
    y_pred_centroid_norm = torch.sum(heatmap_flat * Y_flat_norm, dim=1).item() 

    x_pred_centroid_px = x_pred_centroid_norm * (W - 1)
    y_pred_centroid_px = y_pred_centroid_norm * (H - 1)

    return x_pred_centroid_px, y_pred_centroid_px

def visualize_results(results, save_path_prefix="prediction_result"):
    num_crops = min(len(results), 16)
        
    cols = 4
    rows = (num_crops + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_crops > 1 else [axes]

    for i in range(num_crops):
        ax = axes[i]
        res = results[i]
        crop_vis = res['crop_64_visual']
        crop_min = crop_vis.min()
        crop_max = crop_vis.max()
        if crop_max - crop_min > 1e-8:
            crop_vis_display = (crop_vis - crop_min) / (crop_max - crop_min)
        else:
            crop_vis_display = crop_vis

        ax.imshow(crop_vis_display, cmap='gray')
        ax.set_title(f"Crop {i+1}")
        ax.axis('off')
        
        x_pred_px_in_crop = res['pred_x_centroid_px']
        y_pred_px_in_crop = res['pred_y_centroid_px']
        ax.plot(x_pred_px_in_crop, y_pred_px_in_crop, 'bo', markersize=3)

        scale_arrow = 10
        if res['normal_x'] is not None and res['normal_y'] is not None:
            arrow_dx = res['normal_x'] * scale_arrow
            arrow_dy = res['normal_y'] * scale_arrow
            ax.arrow(x_pred_px_in_crop, y_pred_px_in_crop, arrow_dx, arrow_dy, head_width=2, head_length=2, fc='red', ec='red', label='Normal')

    for j in range(num_crops, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    crop_save_path = f"{save_path_prefix}_crops.png"
    plt.savefig(crop_save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Визуализация сохранена в '{crop_save_path}'")

def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется: {device}")

    model_path = 'PATH_TO_MODEL_WEIGHTS'
    print(f"Загрузка весов из '{model_path}'...")
    model = UNetHeatmapModel(
        crop_size=args.target_crop_size, 
        heatmap_size=args.target_crop_size, 
        dropout_p=0.0
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("мОдель успешно загружена")

    print(f"Загрузка и предобработка '{args.image_path}'")
    try:
        image = load_and_preprocess_image(args.image_path)
    except Exception as e:
        print(f"Ошибка во время предобработки: {e}")
        return

    print("Поиск пятен")
    try:
        regions = segment_and_find_regions(image, threshold=args.segmentation_threshold)
    except Exception as e:
        print(f"Ошибка поиска пятен: {e}")
        return

    results = []
    print("Анализ пятен")
    for i, region in enumerate(tqdm(regions, desc="Пятна")):
        region_bbox = region.bbox
        try:
            crop_data = extract_and_process_crop(
                image, region_bbox, 
                margin=args.margin, 
                target_crop_size=args.target_crop_size
            )
        except Exception as e:
            print(f"Ошибка обработки пятна {i}: {e}")
            continue
        
        if crop_data is None:
            continue

        try:
            crop_tensor = torch.tensor(crop_data['crop_4ch'], dtype=torch.float32).unsqueeze(0).to(device) # (1, 4, 64, 64)
        except Exception as e:
            print(f"Ошибка torch на пятне {i}: {e}")
            continue

        try:
            x_pred_centroid_px, y_pred_centroid_px = predict_centroid(
                model, crop_tensor, device, crop_size=args.target_crop_size
            )
        except Exception as e:
            print(f"Ошибка прогона через модель пятна {i}: {e}")
            continue

        x_pred_img = x_pred_centroid_px + crop_data['minc_crop']
        y_pred_img = y_pred_centroid_px + crop_data['minr_crop']

        print(f"Пятно {i+1}: центр предсказан в ({x_pred_img:.2f}, {y_pred_img:.2f})")

        crop_for_normal = crop_data['crop_64_visual']
        nx, ny, nz = estimate_normal_from_spot(img_spot=crop_for_normal, plot_ellipse=False)

        results.append({
            'pred_x_img': x_pred_img,
            'pred_y_img': y_pred_img,
            'minc_crop': crop_data['minc_crop'],
            'minr_crop': crop_data['minr_crop'],
            'crop_64_visual': crop_data['crop_64_visual'],
            'crop_center_x': crop_data['crop_center_x'],
            'crop_center_y': crop_data['crop_center_y'],
            'pred_x_centroid_px': x_pred_centroid_px,
            'pred_y_centroid_px': y_pred_centroid_px,
            'normal_x': nx,
            'normal_y': ny,
            'normal_z': nz
        })

    print("\n Результаты")
    if results:
        for i, res in enumerate(results):
            if res['normal_x'] is not None:
                print(f"Пятно {i+1}: ({res['pred_x_img']:.6f}, {res['pred_y_img']:.6f}), Нормаль=({res['normal_x']:.6f}, {res['normal_y']:.6f}, {res['normal_z']:.6f})")
            else:
                print(f"Пятно {i+1}: ({res['pred_x_img']:.6f}, {res['pred_y_img']:.6f}), Нормаль=не удалось оценить")
    else:
        print("Нет результатов")

    if results:
        visualize_results(results)
    else:
        print("Нет результатов для визуализации")

if __name__ == "__main__":
    main()