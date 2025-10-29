import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import cv2
from skimage.measure import regionprops, label
from skimage.filters import sobel, laplace
from PIL import Image
from tqdm import tqdm

from src.model import UNetHeatmapModel

def parse_args():
    parser = argparse.ArgumentParser(description="Eval model")
    
    parser.add_argument('--data_csv', type=str, default='resultsCleared.csv',
                        help='Путь к csv-файлу с реальными координатами центров')
    parser.add_argument('--image_folder', type=str, default='imagesGood',
                        help='Путь к папке с bmp изображениями')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Путь к чекпойнту pth')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Путь к конфигурации')
    
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                        help='Путь к сохранению результатов')
      
    parser.add_argument('--device', type=str, default=None,
                        help="Девайс - ('cpu', 'cuda', 'cuda:0', и т.д.)")

    return parser.parse_args()

def load_model(model_path, config_path, device):
    model_params = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                train_config = json.load(f)
            model_params = {
                'crop_size': train_config.get('crop_size', 64),
                'heatmap_size': train_config.get('heatmap_size', 64),
                'dropout_p': train_config.get('dropout_p', 0.0),
            }
            print(f"Параметры загружены: {model_params}")
        except Exception as e:
            print(f"Ошибка загрузки параметров '{config_path}': {e}")
    else:
        print("Файл конфигурации отсутствует")
        
    model = UNetHeatmapModel(**model_params)
    
    print(f"загрузка чекпойнта '{model_path}'")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка при загрузке весов: {e}")
        raise

def preprocess_single_image(img_array):
    image = img_array.astype(np.float32)
    image = cv2.GaussianBlur(image, (5, 5), sigmaX=0, sigmaY=0)
    return image

def apply_model_to_images(model, df_true_coords, image_folder, margin, target_crop_size, device):
    all_results = []
    image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith('.bmp')]
    image_paths = [os.path.join(image_folder, f) for f in image_filenames]

    print(f"Найдено {len(image_paths)} изображений для обработки")

    for image_path in tqdm(image_paths, desc="Обработка:"):
        filename = os.path.basename(image_path)
        row = df_true_coords.loc[df_true_coords['filename'] == filename]
        if row.empty:
            continue

        img = np.array(Image.open(image_path))
        image = preprocess_single_image(img)

        binary_mask = (image > 60).astype(np.uint8)
        labeled_img = label(binary_mask)
        regions = regionprops(labeled_img)

        true_centers = [
            (row.iloc[0]['x1'], row.iloc[0]['y1']),
            (row.iloc[0]['x2'], row.iloc[0]['y2'])
        ]

        for region in regions:
            minr, minc, maxr, maxc = region.bbox

            minr_margin = max(minr - margin, 0)
            minc_margin = max(minc - margin, 0)
            maxr_margin = min(maxr + margin, image.shape[0])
            maxc_margin = min(maxc + margin, image.shape[1])

            crop_full = image[minr_margin:maxr_margin, minc_margin:maxc_margin]

            if crop_full.shape[0] < 35 or crop_full.shape[1] < 35:
                continue

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
            
            crop_tensor = torch.tensor(crop_4ch, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                heatmap_pred = model(crop_tensor)

            H, W = heatmap_pred.shape[-2], heatmap_pred.shape[-1]
            x_coords = torch.arange(W, dtype=torch.float64, device=device) / (W - 1)
            y_coords = torch.arange(H, dtype=torch.float64, device=device) / (H - 1)
            X_coords, Y_coords = torch.meshgrid(x_coords, y_coords, indexing='xy')

            heatmap_flat_f64 = heatmap_pred.squeeze(1).flatten(start_dim=1).double()
            X_flat_f64 = X_coords.flatten().double()
            Y_flat_f64 = Y_coords.flatten().double()

            x_pred_crop = torch.sum(heatmap_flat_f64 * X_flat_f64, dim=1).item()
            y_pred_crop = torch.sum(heatmap_flat_f64 * Y_flat_f64, dim=1).item()

            x_pred_img = x_pred_crop + minc_crop
            y_pred_img = y_pred_crop + minr_crop

            dists_to_true = [np.sqrt((x_pred_img - cx) ** 2 + (y_pred_img - cy) ** 2) for cx, cy in true_centers]
            closest_true_idx = np.argmin(dists_to_true)
            true_x, true_y = true_centers[closest_true_idx]

            error = np.sqrt((x_pred_img - true_x)**2 + (y_pred_img - true_y)**2)

            all_results.append({
                'pred_x': float(x_pred_img),
                'pred_y': float(y_pred_img),
                'true_x': float(true_x),
                'true_y': float(true_y),
                'error': float(error),
                'filename': filename
            })

    return all_results

def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        df = pd.read_csv(args.data_csv)
        df['x1'] = df['x1'].str.replace(',', '.').astype(float)
        df['x2'] = df['x2'].str.replace(',', '.').astype(float)
        df['y1'] = df['y1'].str.replace(',', '.').astype(float)
        df['y2'] = df['y2'].str.replace(',', '.').astype(float)
    except Exception as e:
        print(f"Проблема с csv: {e}")
        return

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется: {device}")

    try:
        if args.config_path is None:
            model_dir = os.path.dirname(args.model_path)
            potential_config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(potential_config_path):
                args.config_path = potential_config_path
                print(f"Используется файл конфигурации: {args.config_path}")
            else:
                print("Не обнаружено файла конфигурации")

        model = load_model(args.model_path, args.config_path, device)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    try:
        results = apply_model_to_images(
            model=model,
            df_true_coords=df,
            image_folder=args.image_folder,
            margin=args.margin,
            target_crop_size=args.crop_size,
            device=device
        )
        print(f"Получено {len(results)} результатов.")
    except Exception as e:
        print(f"Ошибка при получении результатов: {e}")
        return

    if results:
        errors = np.array([r['error'] for r in results])
        avg_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)

        print("\n" + "="*40)
        print(f"Средняя ошибка: {avg_error:.4f}")
        print(f"Медианная ошибка: {median_error:.4f}")
        print(f"Стандартное отклонение ошибок: {std_error:.4f}")
        print("="*40)

        print("\nТоп-10 самых больших ошибок")
        sorted_results = sorted(results, key=lambda x: x['error'], reverse=True)
        for i in range(min(10, len(sorted_results))):
            res = sorted_results[i]
            print(f"Rank {i+1}: Error = {res['error']:.4f}, File = {res['filename']}, Pred = ({res['pred_x']:.2f}, {res['pred_y']:.2f}), True = ({res['true_x']:.2f}, {res['true_y']:.2f})")

        results_path = os.path.join(args.save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    else:
        print("Результатов не получено")

if __name__ == "__main__":
    main()


