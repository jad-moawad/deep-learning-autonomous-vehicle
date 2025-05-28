import torch
import torch.nn as nn


class WeightedL1Loss(nn.Module):
    """
    Custom L1 loss that weights lateral and longitudinal errors differently
    """
    def __init__(self, lateral_weight=2.0, longitudinal_weight=1.5):
        super().__init__()
        self.lateral_weight = lateral_weight
        self.longitudinal_weight = longitudinal_weight
    
    def forward(self, pred, target, mask=None):
        """
        Calculate weighted L1 loss
        
        Args:
            pred: Predicted waypoints (B, n_waypoints, 2)
            target: Target waypoints (B, n_waypoints, 2)
            mask: Valid waypoint mask (B, n_waypoints)
        """
        # Calculate L1 loss per dimension
        loss = torch.abs(pred - target)
        
        # Apply weights
        loss[:, :, 0] *= self.lateral_weight    # Lateral (x)
        loss[:, :, 1] *= self.longitudinal_weight  # Longitudinal (z)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() * 2 + 1e-8)
        
        return loss.mean()


class HuberLoss(nn.Module):
    """
    Huber loss for robustness to outliers
    """
    def __init__(self, delta=1.0, lateral_weight=2.0, longitudinal_weight=1.5):
        super().__init__()
        self.delta = delta
        self.lateral_weight = lateral_weight
        self.longitudinal_weight = longitudinal_weight
    
    def forward(self, pred, target, mask=None):
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Huber loss calculation
        loss = torch.where(
            abs_diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * abs_diff - 0.5 * self.delta ** 2
        )
        
        # Apply weights
        loss[:, :, 0] *= self.lateral_weight
        loss[:, :, 1] *= self.longitudinal_weight
        
        # Apply mask
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() * 2 + 1e-8)
        
        return loss.mean()