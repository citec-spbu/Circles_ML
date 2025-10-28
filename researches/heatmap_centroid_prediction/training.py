import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.model import UNetHeatmapModel
from src.dataset import UNetHeatmapDataset, collect_ml_dataset_unet
from src.utils import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Model.")
    
    parser.add_argument('--data_csv', type=str, default='resultsCleared.csv',
                        help='Путь к csv-файлу с реальными координатами центров')
    parser.add_argument('--image_folder', type=str, default='imagesGood',
                        help='Путь к папке с bmp изображениями')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Тестовая часть данных')

    parser.add_argument('--crop_size', type=int, default=64,
                        help='Размер кропа пятен')
    parser.add_argument('--heatmap_size', type=int, default=64,
                        help='Размер heatmap выхода декодера')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='Dropout')

    parser.add_argument('--epochs', type=int, default=300,
                        help='Число эпох')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Начальный learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2-регуляризация')
    
    parser.add_argument('--save_dir', type=str, default='./runs/exp',
                        help='Директория сохранения чекпойнтов и логов')
    parser.add_argument('--checkpoint_metric', type=str, choices=['average_error', 'median_error'], default='median_error',
                        help='Метрика для валидации лучшей модели (медианная/средняя ошибка)')
                    
    parser.add_argument('--device', type=str, default=None,
                        help="Девайс ('cpu', 'cuda', 'cuda:0', и т.д.)")

    return parser.parse_args()

def setup_save_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if os.listdir(save_dir):
        print(f"Save directory '{save_dir}' is not empty.")
    return save_dir

def main():
    args = parse_args()
    save_dir = setup_save_dir(args.save_dir)
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Конфигурация сохранена в: {config_path}")

    print(f"Загрузка таргетов из '{args.data_csv}'")
    try:
        df = pd.read_csv(args.data_csv)
        df['x1'] = df['x1'].str.replace(',', '.').astype(float)
        df['x2'] = df['x2'].str.replace(',', '.').astype(float)
        df['y1'] = df['y1'].str.replace(',', '.').astype(float)
        df['y2'] = df['y2'].str.replace(',', '.').astype(float)
    except Exception as e:
        print(f"Проблема с csv: {e}")
        return

    print(f"Получение изображений из '{args.image_folder}'")
    try:
        samples = collect_ml_dataset_unet(
            df, args.image_folder, 
            margin=10,
            target_crop_size=args.crop_size
        )
        print(f"ПОлучено {len(samples)} изображений")
    except Exception as e:
        print(f"ПРоблема с загрузкой изображений: {e}")
        return

    if not samples:
        print("Данных не получено")
        return

    train_samples, val_samples = train_test_split(
        samples, test_size=args.test_size, random_state=args.random_seed
    )

    train_dataset = UNetHeatmapDataset(train_samples)
    val_dataset = UNetHeatmapDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется: {device}")

    try:
        model = UNetHeatmapModel(
            crop_size=args.crop_size, 
            heatmap_size=args.heatmap_size, 
            dropout_p=args.dropout_p
        ).to(device)    
    except Exception as e:
        print(f"Проблема с инициализацией модели: {e}")
        return

    optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.scheduler_factor, 
            patience=args.scheduler_patience, 
            min_lr=1e-7,
            verbose=True
        )

    best_val_metric = float('inf')
    best_epoch = -1
    
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')

    print(f"Начинаем обучение: {args.epochs} эпох")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(train_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (crops, coords_norm, paths) in enumerate(train_loader):
            crops = crops.to(device)
            coords_norm = coords_norm.to(device).double()

            optimizer.zero_grad()
            heatmap_pred = model(crops)

            H, W = heatmap_pred.shape[-2], heatmap_pred.shape[-1]
            x_coords_norm = torch.arange(W, dtype=torch.float64, device=device) / (W - 1) 
            y_coords_norm = torch.arange(H, dtype=torch.float64, device=device) / (H - 1)
            X_coords_norm, Y_coords_norm = torch.meshgrid(x_coords_norm, y_coords_norm, indexing='xy') 

            heatmap_flat_f64 = heatmap_pred.squeeze(1).flatten(start_dim=1).double() 
            X_flat_norm_f64 = X_coords_norm.flatten().double()
            Y_flat_norm_f64 = Y_coords_norm.flatten().double()

            x_pred_centroid_norm = torch.sum(heatmap_flat_f64 * X_flat_norm_f64, dim=1) 
            y_pred_centroid_norm = torch.sum(heatmap_flat_f64 * Y_flat_norm_f64, dim=1)
            coords_pred_centroid_norm = torch.stack([x_pred_centroid_norm, y_pred_centroid_norm], dim=1) 

            criterion = nn.SmoothL1Loss()
            loss = criterion(coords_pred_centroid_norm.float(), coords_norm.float()) 
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.6f}')

        avg_err, med_err, _ = evaluate_model(
                model, val_loader, device, crop_size=args.crop_size
            )
            
        print(f'[Validation] Avg Err: {avg_err:.4f}, Med Err: {med_err:.4f}')

        current_metric = None
        if args.checkpoint_metric == 'average_error':
            current_metric = avg_err
        elif args.checkpoint_metric == 'median_error':
            current_metric = med_err
            
        if current_metric < best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch + 1
            print(f"  ** Новый лучший: {args.checkpoint_metric} ({current_metric:.6f}) на эпохе {best_epoch}, модель сохраняется в '{best_model_path}' **")
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(current_metric)
            
        torch.save(model.state_dict(), last_model_path)

    print("Обучение завершено")
    print(f"Лучший {args.checkpoint_metric}: {best_val_metric:.4f} на эпохе {best_epoch}")
    print("="*40)

    try:
        best_model = UNetHeatmapModel(
            crop_size=args.crop_size, 
            heatmap_size=args.heatmap_size, 
            dropout_p=args.dropout_p
        ).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.eval()

        print("Оценка лучшей модели")
        final_avg_err, final_med_err, final_error_dict = evaluate_model(
            best_model, val_loader, device, crop_size=args.crop_size
        )

        print(f'СРедняя ошибка: {final_avg_err:.6f} px')
        print(f'Медианная ошибка: {final_med_err:.6f} px')
        
        errors_path = os.path.join(save_dir, 'final_validation_errors.json')
        with open(errors_path, 'w') as f:
            serializable_dict = {
                "error": [float(e) for e in final_error_dict["error"]],
                "filename": final_error_dict["filename"]
            }
            json.dump(serializable_dict, f, indent=4)
        print(f"СЛоварь ошибок сохранен в: '{errors_path}'")
            
    except Exception as e:
        print(f"Ошибка во время финальной валидации: {e}")

if __name__ == "__main__":
    main()
