# Stomata2TranspireNet 🌿💧

**Stomata2TranspireNet** is a deep learning-based pipeline to predict **leaf-level transpiration rates** from **microscopic images of stomata** and **environmental variables** like temperature, humidity, CO₂ concentration, and light intensity.

This project combines computer vision and tabular feature learning to bridge plant physiology with modern AI for better understanding and modeling of plant-water interactions.

---

## 🚀 Features

- 📷 **Stomatal Image Processing**: Preprocessing and augmentation of stomatal micrographs.
- 🌍 **Environmental Data Integration**: Incorporates key climate variables that influence transpiration.
- 🤖 **Multi-Modal Deep Learning**: Fusion of CNN and tabular input features for accurate prediction.
- 📊 **Evaluation Metrics**: Custom loss functions and metrics suited for regression on physiological data.
- 📁 **Modular Codebase**: Clean structure for experimentation and extension.

---

## 🧪 Use Cases

- Crop phenotyping and drought tolerance analysis
- High-throughput transpiration estimation
- Digital plant physiology and trait modeling

---

## 📂 Project Structure
Stomata2TranspireNet/ │ ├── data/ # Raw and processed data │ ├── images/ # Stomatal images │ ├── env_data/ # Environmental variables (CSV, JSON, etc.) │ └── processed/ # Combined & cleaned datasets │ ├── notebooks/ # Jupyter notebooks for exploration and training │ └── EDA.ipynb │ ├── src/ # Source code │ ├── models/ # CNN, MLP, fusion models │ │ └── transpire_net.py │ ├── dataloaders/ # Image & tabular data loaders │ ├── utils/ # Preprocessing, metrics, helpers │ └── train.py # Training pipeline │ ├── outputs/ # Saved models, logs, and predictions │ ├── models/ │ └── results/ │ ├── requirements.txt # Python dependencies ├── README.md └── LICENSE