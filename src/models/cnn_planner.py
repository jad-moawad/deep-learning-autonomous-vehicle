import torch
import torch.nn as nn
from .base import BasePlanner, INPUT_MEAN, INPUT_STD


class CNNResidualBlock(nn.Module):
    """Residual block for CNN backbone"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class CNNPlanner(BasePlanner):
    """CNN-based planner for end-to-end driving from images"""
    
    def __init__(
        self,
        n_waypoints: int = 3,
        base_channels: int = 32,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        self.n_waypoints = n_waypoints
        
        # Register normalization parameters
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # CNN backbone
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            CNNResidualBlock(base_channels, base_channels*2, stride=2, dropout_rate=dropout_rate),
            CNNResidualBlock(base_channels*2, base_channels*4, stride=2, dropout_rate=dropout_rate),
            CNNResidualBlock(base_channels*4, base_channels*8, stride=2, dropout_rate=dropout_rate),
            
            nn.Conv2d(base_channels*8, base_channels*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  
        )
        
        feature_dim = base_channels * 16
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Specialized longitudinal network
        self.longitudinal_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  
        )
        
        # Output heads
        self.lateral_head = nn.Linear(128, n_waypoints)
        self.longitudinal_head = nn.Linear(64, n_waypoints)
        
        # Initialize
        nn.init.kaiming_normal_(self.lateral_head.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.longitudinal_head.weight, nonlinearity='relu')
        self.lateral_head.weight.data *= 0.1
        self.longitudinal_head.weight.data *= 0.05

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predicts waypoints from camera image
        
        Args:
            image: shape (B, 3, H, W) with values in [0, 1]
            
        Returns:
            waypoints: shape (B, n_waypoints, 2)
        """
        # Normalize
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Extract features
        features = self.cnn_backbone(x)
        waypoint_features = self.waypoint_head(features)
        
        # Predict waypoints
        lateral = self.lateral_head(waypoint_features)
        
        longitudinal_features = self.longitudinal_network(waypoint_features)
        longitudinal = self.longitudinal_head(longitudinal_features)
        
        waypoints = torch.stack([lateral, longitudinal], dim=2)
        
        return waypoints