import cv2
import numpy as np
import json
import os
import sys
from typing import List, Dict, Tuple

MIN_AREA = 100
MAX_AREA = 50000
CIRCULARITY_THRESHOLD = 0.3
ELLIPSE_ASPECT_RATIO_MAX = 2.0

def advanced_binarization(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary

def interpolate_center(gray: np.ndarray, center: Tuple[float, float], window_size=10) -> Tuple[float, float]:
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
    resized = cv2.resize(local_patch, (local_patch.shape[1]*zoom_factor, local_patch.shape[0]*zoom_factor),
                         interpolation=cv2.INTER_CUBIC)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resized)
    max_x, max_y = maxLoc

    refined_x = x_min + max_x / zoom_factor
    refined_y = y_min + max_y / zoom_factor

    return refined_x, refined_y

def refine_marker_center(gray: np.ndarray, contour: np.ndarray, approx_center: Tuple[float, float]) -> Tuple[float, float]:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = approx_center

    center_array = np.array([[cx, cy]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    gray_float = np.float32(gray)
    harris_dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    harris_norm = cv2.normalize(harris_dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    center_array = center_array.reshape(-1, 1, 2)
    cv2.cornerSubPix(harris_norm, center_array, (10, 10), (-1, -1), criteria)

    refined_cx, refined_cy = center_array[0, 0]

    max_shift = 2
    if abs(refined_cx - cx) > max_shift or abs(refined_cy - cy) > max_shift:
        return cx, cy

    return refined_cx, refined_cy

def detect_markers(image_path: str) -> List[Dict]:
    print(f"\nОбработка изображения: {os.path.basename(image_path)}")

    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось загрузить изображение")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = advanced_binarization(gray)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Найдено контуров: {len(contours)}")

    markers = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA or len(contour) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(contour)
            center_x, center_y = ellipse[0]
            (width, height), angle = ellipse[1], ellipse[2]

            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

            print(f"Контур {i}: площадь={area:.1f}, округлость={circularity:.3f}, "
                  f"эллипс {width:.1f}x{height:.1f}, соотношение={aspect_ratio:.2f}")

            if circularity >= CIRCULARITY_THRESHOLD and aspect_ratio <= ELLIPSE_ASPECT_RATIO_MAX:
                refined_center = refine_marker_center(gray, contour, (center_x, center_y))

                marker_info = {
                    'id': len(markers),
                    'center': refined_center,
                    'center_original': (float(center_x), float(center_y)),
                    'ellipse': ellipse,
                    'area': float(area),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                    'contour_points': len(contour)
                }
                markers.append(marker_info)

                print(f"МАРКЕР {marker_info['id']}: центр=({refined_center[0]:.4f}, {refined_center[1]:.4f})")

        except Exception as e:
            print(f"Ошибка обработки контура {i}: {e}")
            continue

    print(f"\nИТОГО найдено маркеров: {len(markers)}")
    return markers


def save_results(markers: List[Dict], image_path: str, output_dir: str = "results"):
    if not markers:
        print("Нет маркеров для сохранения")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    def convert(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    results = {
        "image_file": image_path,
        "image_size": {"width": 4096, "height": 3000},
        "detection_parameters": {
            "min_area": MIN_AREA,
            "max_area": MAX_AREA,
            "circularity_threshold": CIRCULARITY_THRESHOLD
        },
        "markers_count": len(markers),
        "markers": markers
    }

    json_path = os.path.join(output_dir, f"{base_name}_markers.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert)

    print(f"Результаты сохранены: {json_path}")
    return json_path

def main():
    if len(sys.argv) < 2:
        print("Использование: python improved_marker_detector.py <путь_к_изображению>")
        print("Пример: python improved_marker_detector.py image.bmp")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return

    markers = detect_markers(image_path)

    if markers:
        save_results(markers, image_path)

        print("\n" + "=" * 50)
        print("СВОДКА РЕЗУЛЬТАТОВ:")
        for marker in markers:
            print(f"Маркер {marker['id']}: ({marker['center'][0]:.4f}, {marker['center'][1]:.4f}) "
                  f"площадь={marker['area']:.1f} окр={marker['circularity']:.3f}")
    else:
        print("Маркеры не найдены. Попробуйте настроить параметры.")

if __name__ == "__main__":
    main()