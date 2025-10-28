import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.model import UNetHeatmapModel
from src.dataset import UNetHeatmapDataset, collect_ml_dataset_unet
from src.utils import evaluate_model

def main():
    csv_path = 'resultsCleared.csv'
    image_folder = 'imagesGood'

    df = pd.read_csv(csv_path)
    df['x1'] = df['x1'].str.replace(',', '.').astype(float)
    df['x2'] = df['x2'].str.replace(',', '.').astype(float)
    df['y1'] = df['y1'].str.replace(',', '.').astype(float)
    df['y2'] = df['y2'].str.replace(',', '.').astype(float)

    samples = collect_ml_dataset_unet(df, image_folder, margin=10, target_crop_size=64)

    if not samples:
        print("Не удалось загрузить данные.")
        return
    
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    train_dataset = UNetHeatmapDataset(train_samples)
    val_dataset = UNetHeatmapDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetHeatmapModel(crop_size=64, heatmap_size=64, dropout_p=0.0).to(device) 
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    criterion = nn.SmoothL1Loss()

    best_val_error = float('inf')
    best_model_path = 'best_model.pth'

    num_epochs = 300
    print("Training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for crops, coords, path in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            crops = crops.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()
            heatmap_pred = model(crops)

            H, W = heatmap_pred.shape[-2], heatmap_pred.shape[-1]
            x_coords_norm = torch.arange(W, dtype=torch.float64, device=device) / (W - 1)
            y_coords_norm = torch.arange(H, dtype=torch.float64, device=device) / (H - 1)
            X_coords_norm, Y_coords_norm = torch.meshgrid(x_coords_norm, y_coords_norm, indexing='xy')

            
            heatmap_flat_f64 = heatmap_pred.squeeze(1).flatten(start_dim=1).double()
            X_flat_norm_f64 = X_coords_norm.flatten()
            Y_flat_norm_f64 = Y_coords_norm.flatten()
            
            x_pred_centroid_norm = torch.sum(heatmap_flat_f64 * X_flat_norm_f64, dim=1)
            y_pred_centroid_norm = torch.sum(heatmap_flat_f64 * Y_flat_norm_f64, dim=1)
            coords_pred_centroid_norm = torch.stack([x_pred_centroid_norm, y_pred_centroid_norm], dim=1)

            loss = criterion(coords_pred_centroid_norm, coords)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}')

        avg_err, med_err, _ = evaluate_model(
            model, train_loader, device, crop_size=64
        )
        print(f'[Validation] Avg Err: {avg_err:.6f}, Med Err: {med_err:.6f}')

        if med_err < best_val_error:
            print(f"  ** New best validation error (Centroid): {med_err:.4f}, saving model to '{best_model_path}' **")
            best_val_error_cent = med_err
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(med_err)

if __name__ == "__main__":
    main()
