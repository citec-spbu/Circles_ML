import sys
sys.path.insert(1, '.')
from another_researches.data_loader import MarkerDataset
from another_researches.marker_detector import detect_markers_adaptive
from another_researches.evaluate_image import match_and_evaluate
import matplotlib.pyplot as plt
import numpy as np

dataset = MarkerDataset("1109ImagesWithoutOutliers")
sample = dataset[0]
img = sample['image']
gt_centers = sample['gt_centers']
filename = sample['filename']

print(f"\nОбрабатываем: {filename}")
print(f"GT центры: {gt_centers}")

centers, methods, binary = detect_markers_adaptive(img, gt_centers, debug=True)
matches, errors, mean_error = match_and_evaluate(centers, gt_centers, max_dist=100)

print(f"Найдено: {len(centers)} меток")
for i, (err, method) in enumerate(zip(errors, methods)):
    print(f"  Метка {i+1}: ошибка {err:.4f} px → метод: {method}")
print(f"Средняя ошибка: {mean_error:.4f} px")