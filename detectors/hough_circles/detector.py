import math

import cv2
import numpy as np
from typing import List
from detectors import BaseDetector, DetectionResult


class HoughCirclesDetector(BaseDetector):
    """Детектор на основе Hough Circles с субпиксельной точностью"""

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        # Применяем параметры из конфигурации
        dp = self.config.get('dp', 1.2)
        min_dist = self.config.get('min_dist', 50)
        param1 = self.config.get('param1', 100)
        param2 = self.config.get('param2', 30)
        min_radius = self.config.get('min_radius', 10)
        max_radius = self.config.get('max_radius', 100)
        refine_centers = self.config.get('refine_centers', True)

        # Конвертируем в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Детекция кругов
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp,
            minDist=min_dist, param1=param1,
            param2=param2, minRadius=min_radius,
            maxRadius=max_radius
        )

        results = []
        if circles is not None:
            # circles[0] содержит массив кругов в формате [x, y, radius]
            circles = circles[0]

            for circle in circles:
                x, y, r = circle

                # Уточняем центр с субпиксельной точностью
                if refine_centers:
                    x_refined, y_refined = self._refine_center(gray, x, y, r, math.ceil(2*r))
                else:
                    x_refined, y_refined = x, y

                results.append(DetectionResult(
                    center_x=float(x_refined),
                    center_y=float(y_refined),
                    normal_x=0.0,
                    normal_y=0.0,
                    normal_z=1.0,
                    radius=float(r),
                    confidence=self._calculate_confidence(gray, x_refined, y_refined, r)
                ))

        return results

    def _refine_center(self, gray: np.ndarray, x: float, y: float, r: float, window_size: int) -> tuple:
        """
        Уточняет центр круга с субпиксельной точностью
        используя метод моментов или градиентов
        """
        x_int, y_int = int(round(x)), int(round(y))
        half_window = window_size // 2

        # Определяем ROI вокруг предполагаемого центра
        y_start = max(0, y_int - half_window)
        y_end = min(gray.shape[0], y_int + half_window + 1)
        x_start = max(0, x_int - half_window)
        x_end = min(gray.shape[1], x_int + half_window + 1)

        roi = gray[y_start:y_end, x_start:x_end]

        if roi.size == 0:
            return x, y

        # Метод 1: Используем моменты для нахождения центра масс
        try:
            # Применяем порог чтобы выделить яркую область
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Вычисляем моменты
            moments = cv2.moments(binary)

            if moments["m00"] != 0:
                center_x = moments["m10"] / moments["m00"]
                center_y = moments["m01"] / moments["m00"]
                return x_start + center_x, y_start + center_y
        except:
            pass

        # Метод 2: Используем взвешенные градиенты
        try:
            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            if np.sum(magnitude) > 0:
                y_coords, x_coords = np.indices(roi.shape)
                weighted_x = np.sum(x_coords * magnitude) / np.sum(magnitude)
                weighted_y = np.sum(y_coords * magnitude) / np.sum(magnitude)
                return x_start + weighted_x, y_start + weighted_y
        except:
            pass

        # Если методы не сработали, возвращаем оригинальные координаты
        return x, y

    def _calculate_confidence(self, gray: np.ndarray, x: float, y: float, r: float) -> float:
        """
        Вычисляет уверенность в обнаружении круга
        """
        confidence = 0.7  # Базовая уверенность

        try:
            # Создаем маску круга
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(round(x)), int(round(y))), int(round(r)), 255, -1)

            # Вычисляем контраст на границе круга
            mask_border = np.zeros_like(gray)
            cv2.circle(mask_border, (int(round(x)), int(round(y))), int(round(r)), 255, 2)

            if np.sum(mask_border) > 0:
                border_pixels = gray[mask_border > 0]
                if len(border_pixels) > 0:
                    border_contrast = np.std(border_pixels)
                    confidence += min(0.3, border_contrast / 100.0)

        except:
            pass

        return min(confidence, 1.0)