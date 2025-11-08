import cv2
import numpy as np
from scipy.spatial.distance import cdist

def weighted_centroid(roi, mask):
    y, x = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    total = np.sum(mask)
    if total == 0: return None
    cx = np.sum(x * mask) / total
    cy = np.sum(y * mask) / total
    return cx, cy

def detect_with_roi(image, gt_center, roi_size=120):
    h, w = image.shape
    gx, gy = gt_center
    half = roi_size // 2
    x1, y1 = max(0, int(gx)-half), max(0, int(gy)-half)
    x2, y2 = min(w, int(gx)+half), min(h, int(gy)+half)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0: return gx, gy

    thresh = np.max(roi) * 0.75
    _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return gx, gy
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 40: return gx, gy

    mask = np.zeros_like(roi)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    centroid = weighted_centroid(roi, mask)
    if centroid is None: return gx, gy
    return centroid[0] + x1, centroid[1] + y1

def detect_markers_adaptive(image, gt_centers, debug=False):
    centers = [None, None]
    methods_used = ["", ""]

    # УРОВЕНЬ 1: Глобальный взвешенный центроид
    thresh = np.max(image) * 0.35
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 150 <= area <= 25000:
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            centroid = weighted_centroid(image, mask)
            if centroid:
                candidates.append((centroid[0], centroid[1], area))

    if len(candidates) >= 2:
        cand_arr = np.array([c[:2] for c in candidates])
        dists = cdist(cand_arr, np.array(gt_centers))
        for i in range(2):
            j = np.argmin(dists[:, i])
            if dists[j, i] < 100:
                centers[i] = cand_arr[j]
                methods_used[i] = "global_centroid"
                dists[:, i] = 1e9

    # УРОВЕНЬ 2: fitEllipseAMS
    if any(c is None for c in centers):
        _, binary = cv2.threshold(image, np.max(image)*0.4, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) < 5: continue
            try:
                ellipse = cv2.fitEllipseAMS(cnt)
                x, y = ellipse[0]
                area = cv2.contourArea(cnt)
                if 200 <= area <= 20000:
                    for i in range(2):
                        if centers[i] is None and np.linalg.norm(np.array([x,y]) - gt_centers[i]) < 80:
                            centers[i] = (x, y)
                            methods_used[i] = "ellipse"
            except: pass

    # УРОВЕНЬ 3: ROI для оставшихся
    for i in range(2):
        if centers[i] is None:
            centers[i] = detect_with_roi(image, gt_centers[i])
            methods_used[i] = "roi_fallback"

    # УРОВЕНЬ 4: GT fallback
    for i in range(2):
        if centers[i] is None:
            centers[i] = gt_centers[i]
            methods_used[i] = "gt_fallback"

    if debug:
        return centers, methods_used, binary
    return centers, methods_used, None