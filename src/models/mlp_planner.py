import torch
import torch.nn as nn
from .base import BasePlanner


class ResidualBlock(nn.Module):
    """Residual block to help with gradient flow"""
    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        return x + self.layers(x)


class MLPPlanner(BasePlanner):
    """MLP-based trajectory planner using lane boundary information"""
    
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_size: int = 512,  
        dropout_rate: float = 0.2,  
    ):
        super().__init__()
        
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        # Feature extraction layers
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 64),  
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        self.centerline_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        self.width_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        
        # Main network
        combined_features = n_track * (64 + 32 + 16)
        
        self.encoder = nn.Sequential(
            nn.Linear(combined_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(hidden_size, dropout_rate),
            ResidualBlock(hidden_size, dropout_rate),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
        )
        
        # Output heads
        self.longitudinal_head = nn.Linear(hidden_size // 4, n_waypoints)
        self.lateral_head = nn.Linear(hidden_size // 4, n_waypoints)
        
        # Initialize
        nn.init.kaiming_normal_(self.lateral_head.weight, nonlinearity='relu')
        self.lateral_head.weight.data *= 0.5

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predicts waypoints from lane boundaries
        
        Args:
            track_left: shape (B, n_track, 2)
            track_right: shape (B, n_track, 2)
            
        Returns:
            waypoints: shape (B, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        all_features = []
        
        for i in range(self.n_track):
            left_point = track_left[:, i, :]
            right_point = track_right[:, i, :]
            
            # Extract features
            point_pair = torch.cat([left_point, right_point], dim=1)
            point_features = self.point_encoder(point_pair)
            
            centerline = (left_point + right_point) / 2
            centerline_features = self.centerline_encoder(centerline)
            
            width = torch.norm(right_point - left_point, dim=1, keepdim=True)
            width_features = self.width_encoder(width)
            
            combined = torch.cat([point_features, centerline_features, width_features], dim=1)
            all_features.append(combined)
        
        x = torch.cat(all_features, dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        longitudinal = self.longitudinal_head(decoded)
        lateral = self.lateral_head(decoded)
        
        waypoints = torch.stack([lateral, longitudinal], dim=2)
        return waypoints