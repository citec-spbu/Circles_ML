import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

class MarkerDataset:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "imagesGood"
        self.excel_path = self.root_dir / "resultsCleared.xlsx"
        
        self.df = pd.read_excel(self.excel_path, header=None)
        self.df.columns = ['x1_gt', 'y1_gt', 'x2_gt', 'y2_gt', 'filename']
        
        self.df['filename'] = self.df['filename'].astype(str).str.strip()
        
        print(f"Загружено {len(self.df)} записей из {self.excel_path.name}")
        print(f"Папка с изображениями: {self.images_dir}")
        
        self._check_files()

    def _check_files(self):
        missing = []
        for fname in self.df['filename']:
            fpath = self.images_dir / fname
            if not fpath.exists():
                missing.append(fname)
        
        if missing:
            print(f"ОШИБКА: Не найдено {len(missing)} изображений!")
            print("Первые 5:")
            for f in missing[:5]:
                print(f"  - {f}")
            print(f"Всего в папке: {len(list(self.images_dir.iterdir()))} файлов")
        else:
            print("Все изображения найдены!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row['filename']
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить {img_path}")
        
        gt_centers = np.array([
            [row['x1_gt'], row['y1_gt']],
            [row['x2_gt'], row['y2_gt']]
        ], dtype=np.float32)
        
        return {
            'image': img,
            'gt_centers': gt_centers,
            'filename': row['filename'],
            'index': idx
        }

    def get_all(self):
        """Генератор по всем данным"""
        for idx in range(len(self)):
            yield self[idx]