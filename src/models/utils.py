from pathlib import Path
import torch
import torch.nn as nn

from .mlp_planner import MLPPlanner
from .transformer_planner import TransformerPlanner
from .cnn_planner import CNNPlanner


MODEL_REGISTRY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    checkpoint_path: str = None,
    device: str = "cpu",
    **model_kwargs,
) -> nn.Module:
    """
    Load a model by name with optional checkpoint
    
    Args:
        model_name: Name of the model architecture
        checkpoint_path: Path to saved weights
        device: Device to load model on
        **model_kwargs: Additional arguments for model constructor
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**model_kwargs)
    
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = model.to(device)
    return model


def save_model(model: nn.Module, save_path: str, model_name: str = None) -> Path:
    """
    Save model weights
    
    Args:
        model: Model to save
        save_path: Path to save weights
        model_name: Optional model name for validation
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate model type if name provided
    if model_name is not None:
        model_class = MODEL_REGISTRY.get(model_name)
        if model_class and not isinstance(model, model_class):
            raise ValueError(f"Model is not of type {model_name}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return save_path


def calculate_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024