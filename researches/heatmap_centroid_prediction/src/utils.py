import numpy as np
import torch
import torch.nn as nn

def evaluate_model(model, dataloader, device, crop_size=64):
    model.eval()
    all_errors_centroid_pixels = []
    all_paths = []
    criterion = nn.SmoothL1Loss()
    with torch.no_grad():
        for crops, coords_norm, paths in dataloader:
            crops = crops.to(device)
            coords_norm = coords_norm.to(device)

            heatmap_pred = model(crops)

            H, W = heatmap_pred.shape[-2], heatmap_pred.shape[-1]
            x_coords_norm = torch.arange(W, dtype=torch.float32, device=device) / (W - 1)
            y_coords_norm = torch.arange(H, dtype=torch.float32, device=device) / (H - 1)
            X_coords_norm, Y_coords_norm = torch.meshgrid(x_coords_norm, y_coords_norm, indexing='xy')

            heatmap_flat_f64 = heatmap_pred.squeeze(1).flatten(start_dim=1).double()
            X_flat_norm_f64 = X_coords_norm.flatten().double() 
            Y_flat_norm_f64 = Y_coords_norm.flatten().double()

            x_pred_centroid_norm = torch.sum(heatmap_flat_f64 * X_flat_norm_f64, dim=1)
            y_pred_centroid_norm = torch.sum(heatmap_flat_f64 * Y_flat_norm_f64, dim=1)
            coords_pred_centroid_norm = torch.stack([x_pred_centroid_norm, y_pred_centroid_norm], dim=1)

            errors_centroid_norm = torch.sqrt(torch.sum((coords_pred_centroid_norm - coords_norm)**2, dim=1))

            errors_centroid_pixels = errors_centroid_norm * (crop_size - 1)
            all_errors_centroid_pixels.extend(errors_centroid_pixels.cpu().numpy())
            all_paths.extend(paths)

    error_dict = {"error": all_errors_centroid_pixels, "filename": all_paths}
    all_errors_centroid_pixels = np.array(all_errors_centroid_pixels)
    avg_error_centroid_px = np.mean(all_errors_centroid_pixels)
    median_error_centroid_px = np.median(all_errors_centroid_pixels)

    return avg_error_centroid_px, median_error_centroid_px, error_dict