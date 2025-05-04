import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset


class StomataDataset(Dataset):
    """
    A PyTorch Dataset for loading RGB stomata images, physiological metadata,
    and filtered, standardized weather data aligned by timestamp.

    Args:
        image_dir (str): Directory containing image files (.jpg).
        meta_dir (str): Path to the CSV file with image-level metadata.
        weather_dir (str): Path to the CSV file with timestamped weather data.
        transform (callable, optional): Optional transform applied to images.
    """
    def __init__(self, image_dir, meta_dir, weather_dir, transform=None):
        self.image_dir = image_dir
        self.meta = pd.read_csv(meta_dir)
        self.meta_var = ['gsw', 'gbw', 'gtw', 'VPDleaf', 'E']

        self.weather = pd.read_csv(weather_dir)
        self.weather['DateTime'] = pd.to_datetime(self.weather['DateTime'])
        self.weather_var = self.weather.columns[1:]

        self.transform = transform

        # Compute per-feature mean and std for normalization
        self.weather_mean = self.weather[self.weather_var].mean()
        self.weather_std = self.weather[self.weather_var].std() + 1e-8  # Avoid divide-by-zero

        # Only keep images that have matching metadata
        self.image_paths = [
            path for path in glob(os.path.join(image_dir, '*.jpg'))
            if os.path.basename(path) in self.meta['Photo_name'].unique()
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform image
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Extract labels and timestamp
        meta_row = self.meta[self.meta['Photo_name'] == image_name]
        label = torch.tensor(
            [meta_row['gsw'].values[0], meta_row['gbw'].values[0]],
            dtype=torch.float32
        )
        image_datetime = pd.to_datetime(meta_row['DateTime'].values[0])

        # Get same-date weather data
        weather_data = self.weather[self.weather['DateTime'].dt.date == image_datetime.date()].copy()

        # Normalize weather variables
        weather_data[self.weather_var] = (weather_data[self.weather_var] - self.weather_mean) / self.weather_std

        # Mask and forward-fill weather values after timestamp
        mask = weather_data['DateTime'] > image_datetime
        for var in self.weather_var:
            last_valid = weather_data.loc[~mask, var].iloc[-1] if not weather_data.loc[~mask].empty else 0
            weather_data.loc[mask, var] = last_valid

        # Convert to tensor
        weather_tensor = torch.tensor(weather_data[self.weather_var].values, dtype=torch.float32)

        return image, weather_tensor, label
    

if __name__ == '__main__':
    image_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\raw\image'
    meta_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\processed\information.csv'
    weather_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\processed\weather_data.csv'
    dataset = StomataDataset(image_dir, meta_dir, weather_dir)
    print(len(dataset))