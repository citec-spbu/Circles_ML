import sys
sys.path.insert(1, '.')
from another_researches.data_loader import MarkerDataset

dataset = MarkerDataset("1109ImagesWithoutOutliers")

print("\nПример первой записи:")
sample = dataset[0]
print(f"Файл: {sample['filename']}")
print(f"Размер изображения: {sample['image'].shape}")
print(f"GT центры:\n{sample['gt_centers']}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(sample['image'], cmap='gray')
h, w = sample['image'].shape
for i, (x, y) in enumerate(sample['gt_centers']):
    plt.plot(x, y, 'ro', markersize=8)
    plt.text(x+10, y, f"GT {i+1}", color='red', fontsize=12)
plt.title(f"{sample['filename']}")
plt.axis('off')
plt.show()