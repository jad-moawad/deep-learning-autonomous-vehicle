import numpy as np
import torch


class PlannerMetric:
    """
    Computes trajectory prediction metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metric accumulator"""
        self.l1_errors = []
        self.total = 0
    
    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor, labels_mask: torch.Tensor):
        """
        Add predictions to metric computation
        
        Args:
            preds: Predicted waypoints (B, n_waypoints, 2)
            labels: Ground truth waypoints (B, n_waypoints, 2)
            labels_mask: Valid waypoint mask (B, n_waypoints)
        """
        error = (preds - labels).abs()
        error_masked = error * labels_mask[..., None]
        
        # Sum across batch and waypoints
        error_sum = error_masked.sum(dim=(0, 1)).cpu().numpy()
        
        self.l1_errors.append(error_sum)
        self.total += labels_mask.sum().item()
    
    def compute(self) -> dict:
        """
        Compute final metrics
        
        Returns:
            Dictionary containing:
            - l1_error: Total L1 error
            - longitudinal_error: Error in forward direction
            - lateral_error: Error in lateral direction
            - num_samples: Number of samples evaluated
        """
        if self.total == 0:
            return {
                "l1_error": 0.0,
                "longitudinal_error": 0.0,
                "lateral_error": 0.0,
                "num_samples": 0
            }
        
        error = np.stack(self.l1_errors, axis=0)
        lateral_error = error[:, 0].sum() / self.total
        longitudinal_error = error[:, 1].sum() / self.total
        l1_error = lateral_error + longitudinal_error
        
        return {
            "l1_error": float(l1_error),
            "longitudinal_error": float(longitudinal_error),
            "lateral_error": float(lateral_error),
            "num_samples": self.total
        }