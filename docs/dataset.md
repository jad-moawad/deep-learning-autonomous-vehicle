# Dataset Documentation

## SuperTuxKart Driving Dataset

### Overview
This project uses a custom driving dataset collected from the SuperTuxKart racing game, which provides realistic driving scenarios with ground truth trajectory information.

### Dataset Structure

drive_data/
├── train/
│   ├── episode_001/
│   │   ├── 00000_im.jpg
│   │   ├── 00001_im.jpg
│   │   ├── ...
│   │   └── info.npz
│   ├── episode_002/
│   └── ...
└── val/
├── episode_001/
└── ...

### Data Format

Each episode contains:
- **Images**: RGB images (96×128) from the vehicle's front camera
- **info.npz**: NumPy archive containing:
  - `track`: Track geometry information
  - `frames`: Dictionary with per-frame data

### Frame Data Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `location` | (N, 3) | Vehicle position in world coordinates |
| `front` | (N, 3) | Point the vehicle is facing |
| `velocity` | (N, 3) | Vehicle velocity vector |
| `distance_down_track` | (N,) | Distance traveled along track |
| `P` | (N, 4, 4) | Projection matrix |
| `V` | (N, 4, 4) | View matrix |

### Track Information

The track object contains:
- `path_nodes`: Center line of the track
- `path_distance`: Cumulative distance along track
- `path_width`: Track width at each point

### Data Processing Pipeline

1. **Track Boundary Extraction**
   - Extract left and right boundaries from track center and width
   - Sample 10 points ahead of vehicle
   - Transform to ego-vehicle coordinates

2. **Waypoint Generation**
   - Use future vehicle positions as ground truth waypoints
   - Sample 3 waypoints with configurable skip distance
   - Apply same ego-vehicle transformation

3. **Image Preprocessing**
   - Normalize with dataset statistics
   - Mean: [0.2788, 0.2657, 0.2629]
   - Std: [0.2064, 0.1944, 0.2252]

### Data Augmentation

Available augmentation strategies:
- Random horizontal flip (with corresponding track flip)
- Gaussian noise on track boundaries
- Random brightness/contrast for images

### Usage Example

```python
from src.data import load_data

# Load training data
train_loader = load_data(
    "drive_data/train",
    transform_pipeline="default",  # or "state_only", "augmented"
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for batch in train_loader:
    images = batch['image']  # (B, 3, 96, 128)
    track_left = batch['track_left']  # (B, 10, 2)
    track_right = batch['track_right']  # (B, 10, 2)
    waypoints = batch['waypoints']  # (B, 3, 2)
    waypoints_mask = batch['waypoints_mask']  # (B, 3)

    ### Model Behaviors

**MLP Planner**
- Smooth trajectories with consistent curvature
- Occasionally overshoots on sharp turns
- Very stable and predictable behavior

**Transformer Planner**
- Best at maintaining optimal racing line
- Excellent speed control through corners
- Slightly higher computational cost

**CNN Planner**
- Most robust to visual variations (lighting, textures)
- Best lateral control despite visual complexity
- Occasional issues with depth perception on hills

### Failure Modes

1. **MLP**: Struggles with S-curves and chicanes
2. **Transformer**: Sensitive to incomplete track boundary information
3. **CNN**: Affected by shadows and lighting changes


