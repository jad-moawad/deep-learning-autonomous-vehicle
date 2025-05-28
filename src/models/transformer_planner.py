import torch
import torch.nn as nn
from .base import BasePlanner


class TransformerPlanner(BasePlanner):
    """Transformer-based planner using Perceiver-like architecture"""
    
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        # Input encoding
        self.track_encoder = nn.Sequential(
            nn.Linear(4, d_model),  
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_track, d_model) * 0.02)
        
        # Learnable query embeddings
        self.waypoint_queries = nn.Parameter(torch.randn(n_waypoints, d_model) * 0.02)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        
        # Output projection
        self.waypoint_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),  
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predicts waypoints using attention over track features
        
        Args:
            track_left: shape (B, n_track, 2)
            track_right: shape (B, n_track, 2)
            
        Returns:
            waypoints: shape (B, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Combine tracks
        track_combined = torch.cat([track_left, track_right], dim=2)
        
        # Encode track points
        memory = self.track_encoder(track_combined)
        memory = memory + self.pos_encoding
        
        # Prepare queries
        queries = self.waypoint_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer
        decoded = self.transformer_decoder(queries, memory)
        
        # Project to waypoints
        waypoints = self.waypoint_projector(decoded)
        
        return waypoints