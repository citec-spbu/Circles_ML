import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
from skimage.measure import regionprops, label
from skimage.filters import sobel, laplace
import torch
from torch.utils.data import Dataset


class UNetHeatmapDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        crop_4ch = sample['image']
        norm_coords = sample['norm_coords']
        filename = sample['filename']

        crop_tensor = torch.tensor(crop_4ch, dtype=torch.float32)
        coords_tensor = torch.tensor(norm_coords, dtype=torch.float64)

        return crop_tensor, coords_tensor, filename
    
def collect_dataset(df, image_folder, margin=10, target_crop_size=64):
    samples = []
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.bmp')]

    for image_path in tqdm(image_paths):
        filename = os.path.basename(image_path)
        row = df.loc[df['filename'] == filename]
        if row.empty:
            continue

        img = np.array(Image.open(image_path))
        image = img.astype(np.float32)

        image = cv2.GaussianBlur(image, (5, 5), sigmaX=0, sigmaY=0)

        binary_mask = (image > 60).astype(np.uint8)
        labeled_img = label(binary_mask)
        regions = regionprops(labeled_img)

        true_centers = [
            (row.iloc[0]['x1'], row.iloc[0]['y1']),
            (row.iloc[0]['x2'], row.iloc[0]['y2'])
        ]

        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox

            minr_margin = max(minr - margin, 0)
            minc_margin = max(minc - margin, 0)
            maxr_margin = min(maxr + margin, image.shape[0])
            maxc_margin = min(maxc + margin, image.shape[1])

            crop_full = image[minr_margin:maxr_margin, minc_margin:maxc_margin]

            if crop_full.shape[0] < 35 or crop_full.shape[1] < 35:
                continue

            crop_center_x = minc_margin + crop_full.shape[1] / 2.0
            crop_center_y = minr_margin + crop_full.shape[0] / 2.0

            dists = [np.sqrt((crop_center_x - cx) ** 2 + (crop_center_y - cy) ** 2) for cx, cy in true_centers]
            closest_idx = np.argmin(dists)
            true_x, true_y = true_centers[closest_idx]

            half_crop = target_crop_size // 2
            minc_crop = int(round(crop_center_x - half_crop))
            minr_crop = int(round(crop_center_y - half_crop))
            maxc_crop = minc_crop + target_crop_size
            maxr_crop = minr_crop + target_crop_size

            crop_extract = image[minr_crop:maxr_crop, minc_crop:maxc_crop]
            crop_extract_norm = crop_extract.astype(np.float32) / 255.0

            pad_top = max(0, -minr_crop)
            pad_bottom = max(0, maxr_crop - image.shape[0])
            pad_left = max(0, -minc_crop)
            pad_right = max(0, maxc_crop - image.shape[1])

            crop_64 = np.pad(crop_extract_norm, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0.0)

            x_norm = (true_x - minc_crop) / (target_crop_size - 1)
            y_norm = (true_y - minr_crop) / (target_crop_size - 1)
            x_norm = np.clip(x_norm, 0.0, 1.0)
            y_norm = np.clip(y_norm, 0.0, 1.0)
            norm_coords = np.array([x_norm, y_norm], dtype=np.float64)

            sobel_x = sobel(crop_64, axis=1)
            sobel_y = sobel(crop_64, axis=0)
            laplacian_img = laplace(crop_64)

            crop_4ch = np.stack([crop_64, sobel_x, sobel_y, laplacian_img], axis=0)
            
            sample_name = filename + '_' + str(i)
            samples.append({
                'image': crop_4ch,
                'norm_coords': norm_coords,
                'filename': sample_name
            })

    return samples