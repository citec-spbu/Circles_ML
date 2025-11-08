import sys
sys.path.insert(1, '.')
from another_researches.data_loader import MarkerDataset
from another_researches.marker_detector import detect_markers_adaptive
from another_researches.evaluate_image import match_and_evaluate
import numpy as np

dataset = MarkerDataset("1109ImagesWithoutOutliers")
errors_per_image = []
methods_per_image = []

for sample in dataset.get_all():
    centers, methods, _ = detect_markers_adaptive(sample['image'], sample['gt_centers'])
    _, errs, mean_err = match_and_evaluate(centers, sample['gt_centers'])
    errors_per_image.append(mean_err)
    methods_per_image.append(methods)

errors = np.array(errors_per_image)
print(f"Средняя ошибка: {np.mean(errors):.4f} px")
print(f"Медиана: {np.median(errors):.4f} px")
print(f"95-й перцентиль: {np.percentile(errors, 95):.4f} px")
print(f"Доля < 0.01 px: {np.mean(errors < 0.01)*100:.1f}%")

import matplotlib.pyplot as plt
plt.hist(errors, bins=50, range=(0, 0.5))
plt.axvline(0.01, color='red', linestyle='--', label='0.01 px')
plt.title("Распределение ошибок")
plt.xlabel("Ошибка (px)")
plt.legend()
plt.show()