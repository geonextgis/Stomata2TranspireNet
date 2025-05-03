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
    A PyTorch Dataset for loading RGB images of plant stomata, their associated metadata, and corresponding weather data.

    Each sample includes:
    - An image (.jpg)
    - Weather data filtered to the same date, with values zeroed out after the image timestamp
    - Two physiological labels: stomatal conductance (gsw) and boundary layer conductance (gbw)

    Args:
        image_dir (str): Directory containing image files.
        meta_dir (str): Path to the CSV file with metadata including physiological variables.
        weather_dir (str): Path to the CSV file with timestamped weather data.
        transform (callable, optional): Transform to apply to images.
    """
    def __init__(self, image_dir, meta_dir, weather_dir, transform=None):
        self.image_dir = image_dir
        self.meta_dir = meta_dir
        self.weather_dir = weather_dir
        self.transform = transform

        # Collect all image paths with .jpg extension
        self.image_paths = glob(os.path.join(self.image_dir, '*.jpg'))

        # Load metadata and keep only images that have corresponding metadata entries
        self.meta = pd.read_csv(self.meta_dir)
        self.meta_var = ['gsw', 'gbw', 'gtw', 'VPDleaf', 'E']
        self.image_paths = [
            path for path in self.image_paths
            if os.path.basename(path) in self.meta['Photo_name'].unique()
        ]

        # Load weather data and convert the 'DateTime' column to datetime objects
        self.weather = pd.read_csv(self.weather_dir)
        self.weather['DateTime'] = pd.to_datetime(self.weather['DateTime'])
        self.weather_var = self.weather.columns[1:]

    def __len__(self):
        """Returns the number of valid images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns a single sample consisting of:
        - The image
        - Filtered and time-masked weather data
        - Two physiological labels: gsw and gbw
        """
        # Load and optionally transform the image
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Extract metadata row corresponding to the current image
        meta_row = self.meta[self.meta['Photo_name'] == image_name][self.meta_var]
        label_gsw = meta_row['gsw'].values[0]
        label_gbw = meta_row['gbw'].values[0]

        # Extract the timestamp of the image
        image_datetime = pd.to_datetime(self.meta[self.meta['Photo_name'] == image_name]['DateTime'].values[0])

        # Filter weather data to only include rows from the same date as the image
        weather_data = self.weather[self.weather['DateTime'].dt.date == image_datetime.date()].copy()

        # Zero out all weather values that occur after the image timestamp
        mask = weather_data['DateTime'] > image_datetime
        weather_data.loc[mask, self.weather_var] = 0

        return image, weather_data, label_gsw, label_gbw


if __name__ == '__main__':
    image_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\raw\image'
    meta_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\processed\information.csv'
    weather_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\processed\weather_data.csv'
    dataset = StomataDataset(image_dir, meta_dir, weather_dir)
    print(len(dataset))