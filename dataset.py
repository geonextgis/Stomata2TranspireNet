import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    PyTorch Dataset class for loading images along with their associated metadata and weather data.

    Each image is paired with:
    - Metadata (gsw, gbw, gtw, VPDleaf, E) loaded from a CSV file
    - Weather data with time-based filtering

    Args:
        image_dir (str): Directory path containing image files (.jpg).
        meta_dir (str): Path to the CSV file containing image metadata.
        weather_dir (str): Path to the CSV file containing weather data with a 'DateTime' column.
        transform (callable, optional): Optional transform to be applied on a sample image.
    """
    def __init__(self, image_dir, meta_dir, weather_dir, transform=None):
        self.image_dir = image_dir
        self.meta_dir = meta_dir
        self.weather_dir = weather_dir
        self.transform = transform

        self.image_paths = glob(os.path.join(self.image_dir, '*.jpg'))
        self.meta = pd.read_csv(self.meta_dir)
        self.meta_var = ['gsw', 'gbw', 'gtw', 'VPDleaf', 'E']
        self.image_paths = [path for path in self.image_paths if os.path.basename(path) in list(self.meta['Photo_name'].unique())]

        self.weather = pd.read_csv(self.weather_dir)
        self.weather['DateTime'] = pd.to_datetime(self.weather['DateTime'])
        self.weather_var = self.weather.columns[1:]
        
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Loads and returns a data sample from the dataset at the given index.

        Returns:
            tuple: (image, image_meta, weather_data, label)
                - image (Tensor or PIL Image): Transformed or raw image.
                - image_meta (np.ndarray): Metadata features excluding the label.
                - weather_data (pd.DataFrame): Filtered weather data associated with the image timestamp.
                - label (float): The target label (E - Transpiration Rate).
        """
        # Load image
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load metadata and extract label
        meta_row = self.meta[self.meta['Photo_name'] == image_name][self.meta_var]
        image_meta = meta_row.values.flatten()
        label = image_meta[-1]
        image_meta = image_meta[:-1]

        # Load weather data and apply temporal masking
        image_datetime = pd.to_datetime(self.meta[self.meta['Photo_name'] == image_name]['DateTime'].values[0])
        weather_data = self.weather[self.weather['DateTime'].dt.date == image_datetime.date()].copy()
        mask = weather_data['DateTime'] > image_datetime
        weather_data.loc[mask, self.weather_var] = 0

        # Weather
        image_date = pd.to_datetime(self.meta[self.meta['Photo_name']==image_name]['DateTime'].values[0])
        weather_data = self.weather[self.weather['DateTime'].dt.date == image_date.date()]
        mask = weather_data['DateTime'] > image_date
        weather_data.loc[mask, weather_data.columns[1:]] = 0

        return image, image_meta, weather_data, label


if __name__ == '__main__':
    image_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\raw\image'
    meta_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\processed\information.csv'
    weather_dir = r'D:\GITHUB\MSM-Research\Stomata2TranspireNet\data\processed\weather_data.csv'
    dataset = CustomDataset(image_dir, meta_dir, weather_dir)
    print(len(dataset))