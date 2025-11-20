import numpy as np
import cv2
from typing import List
from detectors import BaseDetector, DetectionResult
from scipy.spatial.distance import cdist


class BlindMarkerDetector(BaseDetector):
    """
    Cлепой детектор двух круговых меток.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_area = self.config.get("min_area", 200)
        self.max_area = self.config.get("max_area", 40000)
        self.min_distance = self.config.get("min_distance_between_markers", 80)
        self.global_thresh_factor = self.config.get("global_thresh_factor", 0.24)
        self.local_roi_size = self.config.get("local_roi_size", 250)
        self.min_score_threshold = self.config.get("min_score_threshold", 12.0)

    def _candidate_score(self, cnt, image):
        area = cv2.contourArea(cnt)
        if area < self.min_area or area > self.max_area:
            return 0.0

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(image, mask=mask)[0]

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        ellipse_ratio = 0.5
        if len(cnt) >= 5:
            try:
                (_, _), (major, minor), _ = cv2.fitEllipse(cnt)
                ellipse_ratio = min(major, minor) / max(major, minor, 1e-6)
            except:
                pass

        score = (
            np.log(area + 1) * 0.45 +
            (mean_val / 255.0) * 42.0 +
            circularity * 28.0 +
            solidity * 10.0 +
            ellipse_ratio * 20.0
        )
        return score

    def _local_refine(self, image, guess):
        cx, cy = guess
        h, w = image.shape
        half = self.local_roi_size // 2
        x1 = max(0, int(cx) - half)
        y1 = max(0, int(cy) - half)
        x2 = min(w, int(cx) + half)
        y2 = min(h, int(cy) + half)
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return cx, cy

        thresh = max(170, np.max(roi) * 0.68)
        _, binary = cv2.threshold(roi, int(thresh), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return cx, cy

        best_cnt = max(contours, key=lambda c: self._candidate_score(c, roi))
        if cv2.contourArea(best_cnt) < 100:
            return cx, cy

        mask = np.zeros_like(roi)
        cv2.drawContours(mask, [best_cnt], -1, 255, -1)
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return cx, cy

        cx_local = moments["m10"] / moments["m00"] + x1
        cy_local = moments["m01"] / moments["m00"] + y1
        return cx_local, cy_local

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.uint8)

        h, w = gray.shape
        results = []

        # === 1. Глобальный поиск лучших кандидатов ===
        thresh_val = max(110, np.max(gray) * self.global_thresh_factor)
        _, binary = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        candidates = []
        for cnt in contours:
            score = self._candidate_score(cnt, gray)
            if score < self.min_score_threshold:
                continue
            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            candidates.append((score, cx, cy))

        candidates.sort(reverse=True, key=lambda x: x[0])

        selected = []
        for score, cx, cy in candidates:
            if all(np.hypot(cx - sx, cy - sy) > self.min_distance for sx, sy in selected):
                refined = self._local_refine(gray, (cx, cy))
                selected.append(refined)
            if len(selected) == 2:
                break

        # === 2. Защита от провала (слишком близкие центры) ===
        if len(selected) == 2:
            d = np.hypot(selected[0][0] - selected[1][0], selected[0][1] - selected[1][1])
            if d < self.min_distance * 0.9:
                selected = []

        # === 3. Зональный поиск (углы + центр), если глобальный провалился ===
        if len(selected) < 2:
            zones = [
                (w * 0.2, h * 0.2),  
                (w * 0.8, h * 0.2),  
                (w * 0.2, h * 0.8),  
                (w * 0.8, h * 0.8),  
            ]
            used = [(x, y) for x, y in selected]
            for guess in zones:
                if len(selected) >= 2:
                    break
                refined = self._local_refine(gray, guess)
                if all(np.hypot(refined[0] - ux, refined[1] - uy) > self.min_distance for ux, uy in used):
                    selected.append(refined)
                    used.append(refined)

        # === 4. Финальный fallback (центр изображения) ===
        while len(selected) < 2:
            selected.append((w / 2, h / 2))

        # === Результат ===
        confidence = 0.99 if len(selected) == 2 and np.hypot(selected[0][0] - selected[1][0], selected[0][1] - selected[1][1]) > self.min_distance else 0.7

        total_candidates_found = len([c for c in selected if c != (w/2, h/2)])
        if total_candidates_found == 0:
            confidence = 0.01

        for x, y in selected[:2]:
            results.append(DetectionResult(
                center_x=float(x),
                center_y=float(y),
                normal_x=0.0,
                normal_y=0.0,
                normal_z=1.0,
                radius=None,
                confidence=confidence
            ))

        return results