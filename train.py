import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error
import logging
import os
from tqdm import tqdm


class MultiMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        loss_gsw = self.mse(preds[:, 0], targets[:, 0])
        loss_gbw = self.mse(preds[:, 1], targets[:, 1])
        total_loss = loss_gsw + loss_gbw  # You can weight them if needed
        return total_loss, loss_gsw, loss_gbw


def train_model(model, dataset, batch_size=8, epochs=100, patience=10, lr=1e-4, model_path="best_model.pt", log_path="train.log"):
    # Setup logging
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w')
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split dataset
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Loss, optimizer
    criterion = MultiMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_preds, train_targets = [], []

        for image, weather, label in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            image = image.to(device)
            weather = weather.to(device)
            targets = label.to(device)

            preds = model(image, weather)
            loss, _, _ = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(preds.detach().cpu())
            train_targets.append(targets.cpu())

        # Training metrics
        train_preds = torch.cat(train_preds).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_mse_total = mean_squared_error(train_targets, train_preds)
        train_mape_gsw = np.mean(np.abs((train_targets[:, 0] - train_preds[:, 0]) / (train_targets[:, 0] + 1e-8))) * 100
        train_mape_gbw = np.mean(np.abs((train_targets[:, 1] - train_preds[:, 1]) / (train_targets[:, 1] + 1e-8))) * 100

        # Validation
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []

        with torch.no_grad():
            for image, weather, label in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                image = image.to(device)
                weather = weather.to(device)
                targets = label.to(device)

                preds = model(image, weather)
                loss, _, _ = criterion(preds, targets)

                val_losses.append(loss.item())
                val_preds.append(preds.cpu())
                val_targets.append(targets.cpu())

        # Validation metrics
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_mse_total = mean_squared_error(val_targets, val_preds)
        val_mape_gsw = np.mean(np.abs((val_targets[:, 0] - val_preds[:, 0]) / (val_targets[:, 0] + 1e-8))) * 100
        val_mape_gbw = np.mean(np.abs((val_targets[:, 1] - val_preds[:, 1]) / (val_targets[:, 1] + 1e-8))) * 100

        # Logging
        log_msg = (
            f"Epoch {epoch} | Train Loss: {np.mean(train_losses):.4f}, "
            f"Train MAPE (GSW: {train_mape_gsw:.2f}%, GBW: {train_mape_gbw:.2f}%) | "
            f"Val Loss: {np.mean(val_losses):.4f}, "
            f"Val MAPE (GSW: {val_mape_gsw:.2f}%, GBW: {val_mape_gbw:.2f}%)"
        )
        print(log_msg)
        logging.info(log_msg)

        # Early stopping
        val_loss_avg = np.mean(val_losses)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            logging.info("✅ Model improved and saved.")
        else:
            patience_counter += 1
            logging.info(f"⏳ No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logging.info("⛔ Early stopping triggered.")
            break
