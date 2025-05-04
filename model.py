import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CNNLSTMRegressor(nn.Module):
    """
    A PyTorch model that combines a CNN (ResNet18) for image feature extraction and 
    an LSTM for weather time series encoding, to predict two continuous values (e.g., gsw and gbw).

    Architecture:
    - CNN Backbone (ResNet18): Extracts spatial features from input image.
    - LSTM: Encodes sequential weather data (e.g., temperature, humidity over time).
    - Fully Connected Head: Merges both representations and outputs two regression values.

    Args:
        weather_input_size (int): Number of features in each time step of the weather sequence.
        lstm_hidden_size (int): Number of hidden units in the LSTM.

    Inputs:
        image (Tensor): Input image tensor of shape (B, 3, H, W), where B is batch size.
        weather_seq (Tensor): Weather time series tensor of shape (B, T, F),
                              where T is sequence length and F is number of features.

    Returns:
        Tensor: A tensor of shape (B, 2) containing two predicted values per sample.
    """
    def __init__(self, weather_input_size, lstm_hidden_size):
        super().__init__()
        # Load pretrained ResNet18 and remove classification head
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Replace final FC layer with identity to extract features
        cnn_out_dim = 512  # Output dimension of ResNet18 backbone

        # LSTM for weather time series input
        self.lstm = nn.LSTM(
            input_size=weather_input_size, 
            hidden_size=lstm_hidden_size, 
            batch_first=True
        )

        # Fully connected head to predict two values (e.g., gsw and gbw)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, image, weather_seq):
        """
        Forward pass of the model.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W).
            weather_seq (Tensor): Weather sequence of shape (B, T, F).

        Returns:
            Tensor: Output tensor of shape (B, 2), representing predicted gsw and gbw values.
        """
        # Extract image features using CNN
        img_feat = self.cnn(image)  # Shape: (B, 512)

        # Encode weather sequence with LSTM and get final hidden state
        _, (hn, _) = self.lstm(weather_seq)  # hn shape: (1, B, H)
        weather_feat = hn[-1]  # Take output from the last LSTM layer → shape: (B, H)

        # Concatenate image and weather features
        combined = torch.cat([img_feat, weather_feat], dim=1)  # Shape: (B, 512 + H)

        # Pass through fully connected head to get output
        return self.fc(combined)  # Shape: (B, 2)
