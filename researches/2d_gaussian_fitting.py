import numpy as np
from scipy.optimize import curve_fit
from PIL import Image
import cv2
from skimage.measure import regionprops, label
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Инференс одного изображения")
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Путь к изображению')
    parser.add_argument('--margin', type=int, default=10,
                        help='Отступ')
    parser.add_argument('--segmentation_threshold', type=int, default=60,
                        help='Трешхолд сегментации пятен')

    return parser.parse_args()

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def fit_2d_gaussian(crop):
    h, w = crop.shape
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    x, y = np.meshgrid(x, y)

    initial_amplitude = crop.max()
    initial_xo, initial_yo = np.unravel_index(np.argmax(crop), crop.shape)
    initial_sigma_x = initial_sigma_y = 1.5
    initial_theta = 0
    initial_offset = crop.min()

    initial_guess = (initial_amplitude, initial_xo, initial_yo, initial_sigma_x, initial_sigma_y, initial_theta, initial_offset)

    try:
        popt, _ = curve_fit(gaussian_2d, (x, y), crop.ravel(), p0=initial_guess, maxfev=10000)
        xo, yo = popt[1], popt[2]
        return xo, yo
    except RuntimeError:
        total_intensity = crop.sum()
        if total_intensity == 0:
            return w / 2.0, h / 2.0
        x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        x_center = (x_indices * crop).sum() / total_intensity
        y_center = (y_indices * crop).sum() / total_intensity
        return x_center, y_center
    
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

def extract_and_process_crop(image, region_bbox, margin=10):
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

    crop_full_norm = crop_full.astype(np.float32) / 255.0

    return {
        'crop_full_norm': crop_full_norm,
        'minc_crop': minc_margin, 
        'minr_crop': minr_margin, 
        'crop_center_x': crop_center_x,
        'crop_center_y': crop_center_y
    }

def predict_centroid_gaussian(crop):
    x_pred, y_pred = fit_2d_gaussian(crop)
    return x_pred, y_pred

def visualize_results(results, save_path_prefix="gaussian_prediction_result"):
    num_crops = min(len(results), 16)
        
    cols = 4
    rows = (num_crops + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_crops > 1 else [axes]

    for i in range(num_crops):
        ax = axes[i]
        res = results[i]
        crop_vis = res['crop_full_visual']
        crop_vis_display = crop_vis

        ax.imshow(crop_vis_display, cmap='gray')
        ax.set_title(f"Crop {i+1} (Shape: {crop_vis.shape})")
        ax.axis('off')
        
        x_pred_in_crop_norm = res['pred_x_centroid_crop_norm']
        y_pred_in_crop_norm = res['pred_y_centroid_crop_norm']
        ax.plot(x_pred_in_crop_norm, y_pred_in_crop_norm, 'bo', markersize=3)

    for j in range(num_crops, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    crop_save_path = f"{save_path_prefix}_crops.png"
    plt.savefig(crop_save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Визуализация сохранена в '{crop_save_path}'")

def main():
    args = parse_args()

    print(f"Использование метода Gaussian Fitting")

    print(f"Загрузка и предобработка '{args.image_path}'")
    try:
        img_gray = load_and_preprocess_image(args.image_path)
    except Exception as e:
        print(f"Ошибка во время предобработки: {e}")
        return

    print("Поиск пятен")
    try:
        regions = segment_and_find_regions(img_gray, threshold=args.segmentation_threshold)
    except Exception as e:
        print(f"Ошибка поиска пятен: {e}")
        return

    results = []
    print("Анализ пятен")
    for i, region in enumerate(tqdm(regions, desc="Regions")):
        region_bbox = region.bbox
        try:
            crop_data = extract_and_process_crop(img_gray, region_bbox, margin=args.margin)
        except Exception as e:
            print(f"Ошибка обработки пятна {i}: {e}")
            continue
        
        if crop_data is None:
            continue

        x_pred_crop_norm, y_pred_crop_norm = predict_centroid_gaussian(crop_data['crop_full_norm'])

        x_pred_img = x_pred_crop_norm + crop_data['minc_crop']
        y_pred_img = y_pred_crop_norm + crop_data['minr_crop']

        print(f"Пятно {i+1}: центр предсказан в ({x_pred_img:.2f}, {y_pred_img:.2f})")

        results.append({
            'pred_x_img': x_pred_img,
            'pred_y_img': y_pred_img,
            'minc_crop': crop_data['minc_crop'],
            'minr_crop': crop_data['minr_crop'],
            'crop_full_visual': crop_data['crop_full_norm'],
            'crop_center_x': crop_data['crop_center_x'],
            'crop_center_y': crop_data['crop_center_y'],
            'pred_x_centroid_crop_norm': x_pred_crop_norm,
            'pred_y_centroid_crop_norm': y_pred_crop_norm,
        })

    print("\n Результаты")
    if results:
        for i, res in enumerate(results):
            print(f"Пятно {i+1}: ({res['pred_x_img']:.6f}, {res['pred_y_img']:.6f})")
    else:
        print("Нет результатов")

    if results:
        visualize_results(results)
    else:
        print("Нет результатов для визуализации")
    
if __name__ == "__main__":
    main()

