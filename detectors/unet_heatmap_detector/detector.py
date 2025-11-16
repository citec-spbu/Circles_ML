import numpy as np
import os
import torch
import cv2
from skimage.measure import regionprops, label
from skimage.filters import sobel, laplace
from typing import List
from detectors import BaseDetector, DetectionResult

from .model import UNetHeatmapModel

def estimate_normal_from_spot(img_spot):
    """Оцениваем нормаль пятна с помощью анализа эллипса"""
    img_to_thresh = img_spot
    if img_to_thresh.max() <= 1.0:
        img_to_thresh = img_to_thresh * 255.0
    img_to_thresh = np.clip(img_to_thresh, 0, 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(img_to_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        return None, None, None

    try:
        ellipse = cv2.fitEllipse(largest_contour)
        (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse
    except cv2.error as e:
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
        return 0.0, 0.0, 1.0

    return nx, ny, nz

def segment_and_find_regions(image, threshold=60):
    """Сегментация пятен по порогу"""
    binary_mask = (image > threshold).astype(np.uint8)
    labeled_img = label(binary_mask)
    regions = regionprops(labeled_img)
    return regions

def extract_and_process_crop(image, region_bbox, margin=10, target_crop_size=64):
    """Обработка каждого кропа пятна, формирование доп. признаков"""
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

def predict_centroid(model, crop_4ch_tensor, device):
    """Предсказание модели и постобработка результатов"""
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

class UNetHeatmapDetector(BaseDetector):
    """Детектор центров пятен и нормалей с помощью построения скрытого heatmap"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.device = None
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Загружаем модель, если еще не загружена"""
        if not self._model_loaded:
            model_path_config = self.config.get('model_path', 'models/unet_heatmap_model.pth')
            detector_dir = os.path.dirname(__file__)
            self.model_path_absolute = os.path.abspath(os.path.join(detector_dir, model_path_config))
            
            device_str = self.config.get('device', None)
            if device_str:
                self.device = torch.device(device_str)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if not os.path.exists(self.model_path_absolute):
                 raise FileNotFoundError(f"Файл модели не найден по пути: {self.model_path_absolute}")
            
            crop_size = self.config.get('target_crop_size', 64)
            heatmap_size = self.config.get('target_heatmap_size', crop_size)
            dropout_p = self.config.get('model_dropout_p', 0.0)
            
            self.model = UNetHeatmapModel(
                crop_size=crop_size,
                heatmap_size=heatmap_size,
                dropout_p=dropout_p
            )
            state_dict = torch.load(self.model_path_absolute, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Основная функция детекции пятен и выделения нужных характеристик"""
        margin = self.config.get('margin', 10)
        target_crop_size = self.config.get('target_crop_size', 64)
        segmentation_threshold = self.config.get('segmentation_threshold', 60)

        self._ensure_model_loaded()

        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray_image = image.astype(np.float32)

        image = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=0, sigmaY=0)
        try:
            regions = segment_and_find_regions(image, threshold=segmentation_threshold)
        except Exception as e:
            return []

        results = []
        for i, region in enumerate(regions):
            region_bbox = region.bbox
            try:
                crop_data = extract_and_process_crop(
                    image, region_bbox,
                    margin=margin,
                    target_crop_size=target_crop_size
                )
            except Exception as e:
                continue

            if crop_data is None:
                continue

            try:
                crop_tensor = torch.tensor(crop_data['crop_4ch'], dtype=torch.float32).unsqueeze(0).to(self.device) # (1, 4, 64, 64)
            except Exception as e:
                continue

            try:
                x_pred_centroid_px, y_pred_centroid_px = predict_centroid(
                    self.model, crop_tensor, self.device
                )
            except Exception as e:
                continue

            x_pred_img = x_pred_centroid_px + crop_data['minc_crop']
            y_pred_img = y_pred_centroid_px + crop_data['minr_crop']

            crop_for_normal = crop_data['crop_64_visual']
            nx, ny, nz = estimate_normal_from_spot(img_spot=crop_for_normal)

            confidence = 1.0

            results.append(DetectionResult(
                center_x=float(x_pred_img),
                center_y=float(y_pred_img),
                normal_x=nx if nx is not None else 0.0,
                normal_y=ny if ny is not None else 0.0,
                normal_z=nz if nz is not None else 1.0,
                radius=None,
                confidence=confidence
            ))

        return results
    
    def _on_config_update(self):
        """Перезагрузка модели при изменении конфигурации"""
        self._model_loaded = False

    def cleanup(self):
        """Очистка памяти"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device is not None:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        self._model_loaded = False