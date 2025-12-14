import cv2
import numpy as np
import math
from typing import List
from .. import BaseDetector, DetectionResult

def estimate_normal_from_spot(img_spot):
    """Оцениваем нормаль пятна с помощью анализа эллипса"""
    img_to_thresh = img_spot
    if img_to_thresh.max() <= 1.0:
        img_to_thresh = img_to_thresh * 255.0
    img_to_thresh = np.clip(img_to_thresh, 0, 255).astype(np.uint8)
    
    img_smooth = cv2.GaussianBlur(img_to_thresh, (3, 3), 0.8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_smooth)

    _, binary_mask = cv2.threshold(
        img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 50:
        return None

    try:
        ellipse = cv2.fitEllipse(largest_contour)
        (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse
    except cv2.error:
        return None

    major_axis = max(axis_a, axis_b)
    minor_axis = min(axis_a, axis_b)
    if major_axis <= 0:
        return None

    aspect_ratio = minor_axis / major_axis
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
        return nx, ny, nz
    return None


def estimate_normal_from_spot_alt(img_spot, num_bins=36):
    """Оцениваем нормаль пятна с помощью анализа радиального профиля"""
    img_spot = img_spot.astype(np.float32)
    if img_spot.max() > 1.0:
        img_spot /= 255.0
    
    h, w = img_spot.shape
    cy, cx = h // 2, w // 2
    
    angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    max_radius = min(cx, h - cy, w - cx)
    if max_radius <= 1:
        return None
    
    profile = np.zeros(num_bins, dtype=np.float32)
    
    for i, angle in enumerate(angles):
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        r = np.arange(0, max_radius, 0.5, dtype=np.float32)
        x_line = r * dx + cx
        y_line = r * dy + cy
        
        valid = (
            (x_line >= 0) & (x_line < w) &
            (y_line >= 0) & (y_line < h)
        )
        if np.sum(valid) > 10:
            profile[i] = np.mean(
                img_spot[y_line[valid].astype(int),
                         x_line[valid].astype(int)]
            )

    if np.all(profile == 0):
        return None

    fft_profile = np.fft.fft(profile)
    dominant_angle = np.angle(fft_profile[1])

    profile_norm = profile / (profile.mean() + 1e-8)
    eccentricity = float(np.std(profile_norm))

    theta = np.deg2rad(15.0 * eccentricity)
    phi = float(dominant_angle)
    
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    if norm > 0:
        return nx / norm, ny / norm, nz / norm
    return None

class ContourEllipseDetector(BaseDetector):
    """
    Детектор белых круговых меток на основе контурного анализа
    с субпиксельным уточнением центров
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_area = self.config.get("min_area", 25)
        self.max_area = self.config.get("max_area", 50000)
        self.circularity_threshold = self.config.get("circularity_threshold", 0.3)
        self.ellipse_aspect_ratio_max = self.config.get("ellipse_aspect_ratio_max", 2.0)
        self.refine_centers = self.config.get("refine_centers", True)
        self.morph_close_iterations = self.config.get("morph_close_iterations", 2)
        self.morph_open_iterations = self.config.get("morph_open_iterations", 1)
        self.max_center_shift = self.config.get("max_center_shift", 2)

        estimator_name = self.config.get("normal_estimator", "ellipse")
        if estimator_name == "ellipse":
            self._normal_estimator = estimate_normal_from_spot
        elif estimator_name == "radial":
            self._normal_estimator = estimate_normal_from_spot_alt
        else:
            self._normal_estimator = estimate_normal_from_spot

    def _advanced_binarization(self, gray: np.ndarray) -> np.ndarray:
        """Улучшенная бинаризация с морфологией"""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iterations)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iterations)
        return binary

    def _interpolate_center(self, gray: np.ndarray, center: tuple, window_size: int = 10) -> tuple:
        """Интерполяция центра с увеличением области"""
        x, y = center
        x, y = int(round(x)), int(round(y))
        half_win = window_size // 2
        h, w = gray.shape

        x_min = max(x - half_win, 0)
        x_max = min(x + half_win + 1, w)
        y_min = max(y - half_win, 0)
        y_max = min(y + half_win + 1, h)

        local_patch = gray[y_min:y_max, x_min:x_max]

        if local_patch.shape[0] < 3 or local_patch.shape[1] < 3:
            return center

        zoom_factor = 4
        resized = cv2.resize(local_patch, 
                           (local_patch.shape[1] * zoom_factor, local_patch.shape[0] * zoom_factor),
                           interpolation=cv2.INTER_CUBIC)

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resized)
        max_x, max_y = maxLoc

        refined_x = x_min + max_x / zoom_factor
        refined_y = y_min + max_y / zoom_factor

        return refined_x, refined_y
    
    def _refine_marker_center(self, gray: np.ndarray, contour: np.ndarray, approx_center: tuple) -> tuple:
        """Уточнение центра маркера с субпиксельной точностью"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = approx_center

        if not self.refine_centers:
            return cx, cy

        x, y, w, h = cv2.boundingRect(contour)
        margin = 15
        roi_x = max(0, x - margin)
        roi_y = max(0, y - margin)
        roi_w = min(gray.shape[1] - roi_x, w + 2 * margin)
        roi_h = min(gray.shape[0] - roi_y, h + 2 * margin)
        
        roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        offset_x, offset_y = roi_x, roi_y
        
        center_in_roi = np.array([[[cx - offset_x, cy - offset_y]]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
        
        try:
            gray_roi_float = np.float32(roi)
            harris_dst = cv2.cornerHarris(gray_roi_float, 2, 3, 0.04)
            harris_norm = cv2.normalize(harris_dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            cv2.cornerSubPix(harris_norm, center_in_roi, (10, 10), (-1, -1), criteria)
            
            refined_cx = center_in_roi[0, 0, 0] + offset_x
            refined_cy = center_in_roi[0, 0, 1] + offset_y
            
            if abs(refined_cx - cx) <= self.max_center_shift and abs(refined_cy - cy) <= self.max_center_shift:
                return refined_cx, refined_cy
            else:
                return cx, cy
        except:
            return cx, cy

    def _calculate_confidence(self, area: float, circularity: float, aspect_ratio: float, contour_points: int) -> float:
        """Вычисление уверенности в обнаружении маркера"""
        base_confidence = 0.5
        
        circularity_confidence = min(1.0, circularity / self.circularity_threshold) * 0.3
        
        aspect_ratio_confidence = (1.0 - min(1.0, (aspect_ratio - 1.0) / (self.ellipse_aspect_ratio_max - 1.0))) * 0.2
        
        contour_points_confidence = min(1.0, contour_points / 50.0) * 0.1
        
        confidence = base_confidence + circularity_confidence + aspect_ratio_confidence + contour_points_confidence
        return min(confidence, 1.0)

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Основной метод детекции маркеров"""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        binary = self._advanced_binarization(gray)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area or len(contour) < 5:
                continue

            try:
                ellipse = cv2.fitEllipse(contour)
                center_x, center_y = ellipse[0]
                width, height = ellipse[1]
                
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                
                if (circularity >= self.circularity_threshold and 
                    aspect_ratio <= self.ellipse_aspect_ratio_max):
                    
                    refined_center = self._refine_marker_center(gray, contour, (center_x, center_y))
                    
                    confidence = self._calculate_confidence(area, circularity, aspect_ratio, len(contour))
                    
                    radius = math.sqrt(area / math.pi)

                    x, y, w, h = cv2.boundingRect(contour)
                    margin = 10
                    x0 = max(0, x - margin)
                    y0 = max(0, y - margin)
                    x1 = min(gray.shape[1], x + w + margin)
                    y1 = min(gray.shape[0], y + h + margin)
                    spot_crop = gray[y0:y1, x0:x1]

                    nn = self._normal_estimator(spot_crop)
                    if nn is None:
                        nx, ny, nz = 0.0, 0.0, 1.0
                    else:
                        nx, ny, nz = nn
                    
                    results.append(DetectionResult(
                        center_x=float(refined_center[0]),
                        center_y=float(refined_center[1]),
                        normal_x=float(nx),
                        normal_y=float(ny),
                        normal_z=float(nz),
                        radius=float(radius),
                        confidence=confidence
                    ))
                    
            except Exception:
                continue

        return results