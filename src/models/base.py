import torch
import torch.nn as nn

INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class BasePlanner(nn.Module):
    """Base class for all planner models"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")