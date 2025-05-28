from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import transforms
from .utils import Track


class DrivingDataset(Dataset):
    """
    Autonomous driving dataset for trajectory planning
    """
    
    def __init__(
        self,
        episode_path: str,
        transform_pipeline: str = "default",
    ):
        super().__init__()
        
        self.episode_path = Path(episode_path)
        
        # Load episode data
        info = np.load(self.episode_path / "info.npz", allow_pickle=True)
        
        self.track = Track(**info["track"].item())
        self.frames = {k: np.stack(v) for k, v in info["frames"].item().items()}
        self.transform = self.get_transform(transform_pipeline)
    
    def get_transform(self, transform_pipeline: str):
        """
        Creates a pipeline for processing data
        """
        if transform_pipeline == "default":
            # For vision-based models (includes images)
            xform = transforms.Compose([
                transforms.ImageLoader(self.episode_path),
                transforms.EgoTrackProcessor(self.track),
            ])
        elif transform_pipeline == "state_only":
            # For track-based models (no images)
            xform = transforms.EgoTrackProcessor(self.track)
        elif transform_pipeline == "augmented":
            # With data augmentation
            xform = transforms.Compose([
                transforms.ImageLoader(self.episode_path),
                transforms.EgoTrackProcessor(self.track),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            raise ValueError(f"Unknown transform pipeline: {transform_pipeline}")
        
        return xform
    
    def __len__(self):
        return len(self.frames["location"])
    
    def __getitem__(self, idx: int):
        sample = {"_idx": idx, "_frames": self.frames}
        sample = self.transform(sample)
        
        # Remove private keys
        for key in list(sample.keys()):
            if key.startswith("_"):
                sample.pop(key)
        
        return sample


def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader | Dataset:
    """
    Constructs the dataset/dataloader for training or evaluation
    
    Args:
        dataset_path: Path to dataset directory
        transform_pipeline: Data transformation pipeline to use
        return_dataloader: Return DataLoader if True, Dataset if False
        num_workers: Number of data loading workers
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader or Dataset
    """
    dataset_path = Path(dataset_path)
    scenes = [x for x in dataset_path.iterdir() if x.is_dir()]
    
    # Handle single scene
    if not scenes and dataset_path.is_dir():
        scenes = [dataset_path]
    
    # Load all episodes
    datasets = []
    for episode_path in sorted(scenes):
        datasets.append(DrivingDataset(episode_path, transform_pipeline=transform_pipeline))
    
    dataset = ConcatDataset(datasets)
    
    print(f"Loaded {len(dataset)} samples from {len(datasets)} episodes")
    
    if not return_dataloader:
        return dataset
    
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )