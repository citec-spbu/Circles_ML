import numpy as np
from scipy.optimize import curve_fit
import cv2
from skimage.measure import regionprops, label
from typing import List
from detectors import BaseDetector, DetectionResult

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Задаем двумерную гауссиану"""
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def fit_2d_gaussian(crop):
    """ПОдгонка гауссианы к пятну (если не сошлось, то передаем центроид)"""
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
        sigma_x, sigma_y = popt[3], popt[4]
        theta = popt[5]
        return xo, yo, sigma_x, sigma_y, theta
    except RuntimeError:
        total_intensity = crop.sum()
        if total_intensity == 0:
            return w / 2.0, h / 2.0, 1.0, 1.0, 0.0
        x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        x_center = (x_indices * crop).sum() / total_intensity
        y_center = (y_indices * crop).sum() / total_intensity
        return x_center, y_center, 1.0, 1.0, 0.0
    
def segment_and_find_regions(image, threshold=60):
    """Сегментация пятен по порогу"""
    binary_mask = (image > threshold).astype(np.uint8)
    labeled_img = label(binary_mask)
    regions = regionprops(labeled_img)
    return regions

def extract_and_process_crop(image, region_bbox, margin=10):
    """Извлечение и обработка кропа"""
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

def predict_centroid_and_normal_gaussian(crop):
    """Предсказание центра и нормали с помощью подгонки двумерной гауссианы"""
    x_pred, y_pred, sigma_x, sigma_y, theta = fit_2d_gaussian(crop)

    major_axis = max(sigma_x, sigma_y)
    minor_axis = min(sigma_x, sigma_y)

    aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0.0
    angle_rad = theta
    cos_theta = np.clip(aspect_ratio, 0.0, 1.0)
    theta_normal = np.arccos(cos_theta)
    phi = angle_rad + np.pi / 2

    nx = np.sin(theta_normal) * np.cos(phi)
    ny = np.sin(theta_normal) * np.sin(phi)
    nz = np.cos(theta_normal)

    norm_length = np.sqrt(nx**2 + ny**2 + nz**2)
    if norm_length > 0:
        nx /= norm_length
        ny /= norm_length
        nz /= norm_length
    else:
        return x_pred, y_pred, 0.0, 0.0, 1.0

    return x_pred, y_pred, nx, ny, nz

class GaussianFitterDetector(BaseDetector):
    """Детектор центров пятен с помощью подгонки 2d гауссианы и оценка нормали по парметрам эллипса"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        margin = self.config.get('margin', 10)
        segmentation_threshold = self.config.get('segmentation_threshold', 60)

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
                    margin=margin
                )
            except Exception as e:
                continue

            if crop_data is None:
                continue 

            x_pred_crop_norm, y_pred_crop_norm, nx, ny, nz = predict_centroid_and_normal_gaussian(crop_data['crop_full_norm'])

            x_pred_img = x_pred_crop_norm + crop_data['minc_crop']
            y_pred_img = y_pred_crop_norm + crop_data['minr_crop']

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