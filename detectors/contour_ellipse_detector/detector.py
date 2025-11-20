import cv2
import numpy as np
import math
from typing import List
from detectors import BaseDetector, DetectionResult


class ContourEllipseDetector(BaseDetector):
    """
    Детектор белых круговых меток на основе контурного анализа
    с субпиксельным уточнением центров
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_area = self.config.get("min_area", 100)
        self.max_area = self.config.get("max_area", 50000)
        self.circularity_threshold = self.config.get("circularity_threshold", 0.3)
        self.ellipse_aspect_ratio_max = self.config.get("ellipse_aspect_ratio_max", 2.0)
        self.refine_centers = self.config.get("refine_centers", True)
        self.morph_close_iterations = self.config.get("morph_close_iterations", 2)
        self.morph_open_iterations = self.config.get("morph_open_iterations", 1)
        self.max_center_shift = self.config.get("max_center_shift", 2)

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

        center_array = np.array([[cx, cy]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

        gray_float = np.float32(gray)
        harris_dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
        harris_norm = cv2.normalize(harris_dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        center_array = center_array.reshape(-1, 1, 2)
        cv2.cornerSubPix(harris_norm, center_array, (10, 10), (-1, -1), criteria)

        refined_cx, refined_cy = center_array[0, 0]

        if abs(refined_cx - cx) > self.max_center_shift or abs(refined_cy - cy) > self.max_center_shift:
            return cx, cy

        return refined_cx, refined_cy

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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
                    
                    results.append(DetectionResult(
                        center_x=float(refined_center[0]),
                        center_y=float(refined_center[1]),
                        normal_x=0.0,
                        normal_y=0.0,
                        normal_z=1.0,
                        radius=float(radius),
                        confidence=confidence
                    ))
                    
            except Exception:
                continue

        return results