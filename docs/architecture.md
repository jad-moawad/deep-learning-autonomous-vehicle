# Architecture Details

This document provides detailed information about the neural network architectures implemented in this project.

## Table of Contents
- [MLP Planner](#mlp-planner)
- [Transformer Planner](#transformer-planner)
- [CNN Planner](#cnn-planner)
- [Design Decisions](#design-decisions)

## MLP Planner

### Overview
The MLP Planner is our baseline model that processes geometric track information to predict vehicle trajectories.

### Architecture Details

Input: Left/Right Track Boundaries (B, 10, 2) each
↓
[Feature Extraction]

Point Pair Encoder: (B, 10, 4) → (B, 10, 64)
Centerline Encoder: (B, 10, 2) → (B, 10, 32)
Width Encoder: (B, 10, 1) → (B, 10, 16)
↓
[Feature Aggregation]
Concatenate: (B, 10 * 112) → (B, 1120)
↓
[Main Network]
Linear(1120, 512) + LayerNorm + ReLU + Dropout(0.2)
Linear(512, 512) + LayerNorm + ReLU + Dropout(0.2)
ResidualBlock(512) x2
↓
[Decoder]
Linear(512, 256) + LayerNorm + ReLU + Dropout(0.2)
Linear(256, 128) + LayerNorm + ReLU
↓
[Output Heads]
Lateral Head: Linear(128, 3)
Longitudinal Head: Linear(128, 3)
↓
Output: Waypoints (B, 3, 2)


### Key Design Choices

1. **Multi-Stream Feature Extraction**: We process three complementary features:
   - Point pairs capture relative positions
   - Centerline provides trajectory direction
   - Width indicates available maneuvering space

2. **Residual Connections**: Help with gradient flow and enable deeper networks

3. **Separate Output Heads**: Different networks for lateral and longitudinal predictions as they have different characteristics

## Transformer Planner

### Overview
Inspired by the Perceiver architecture, this model uses attention mechanisms to aggregate track information flexibly.

### Architecture Details
Input: Left/Right Track Boundaries (B, 10, 2) each
↓
[Track Encoding]
Concatenate: (B, 10, 4)
Linear(4, 128) + LayerNorm + ReLU + Dropout(0.1)
↓
[Positional Encoding]
Add learned positional embeddings: (B, 10, 128)
↓
[Cross-Attention]
Queries: Learned waypoint embeddings (3, 128)
Keys/Values: Encoded track features (B, 10, 128)
↓
[Transformer Decoder]
3 layers of:

Multi-Head Cross-Attention (4 heads)
LayerNorm
Feed-Forward Network (512 hidden)
LayerNorm
↓
[Output Projection]
Linear(128, 64) + LayerNorm + ReLU
Linear(64, 2)
↓
Output: Waypoints (B, 3, 2)


### Key Design Choices

1. **Learned Query Embeddings**: Each waypoint has its own query embedding that learns to attend to relevant track features

2. **Cross-Attention Only**: Unlike full Transformers, we only use cross-attention between queries and track features

3. **Shallow Architecture**: 3 layers provide sufficient capacity while maintaining efficiency

## CNN Planner

### Overview
End-to-end model that directly processes camera images to predict trajectories.

### Architecture Details

Input: RGB Image (B, 3, 96, 128)
↓
[Normalization]
Subtract mean, divide by std
↓
[CNN Backbone]

Conv2d(3, 32, 7x7, stride=2) + BN + ReLU
ResidualBlock(32, 64, stride=2)
ResidualBlock(64, 128, stride=2)
ResidualBlock(128, 256, stride=2)
Conv2d(256, 512, 3x3) + BN + ReLU
AdaptiveAvgPool2d(1)
↓
[Waypoint Head]
Flatten: (B, 512)
Linear(512, 256) + LayerNorm + ReLU + Dropout(0.2)
Linear(256, 128) + LayerNorm + ReLU + Dropout(0.2)
↓
[Specialized Longitudinal Network]
Linear(128, 64) + LayerNorm + ReLU + Dropout(0.1)
↓
[Output Heads]
Lateral: Linear(128, 3)
Longitudinal: Linear(64, 3)
↓
Output: Waypoints (B, 3, 2)

### Key Design Choices

1. **Progressive Downsampling**: Reduces spatial dimensions while increasing channels

2. **Global Average Pooling**: Provides translation invariance and fixed-size features

3. **Specialized Longitudinal Network**: Additional processing for longitudinal predictions which require understanding of depth and speed

## Design Decisions

### Loss Function Design
We use a weighted L1 loss with different weights for lateral and longitudinal errors:
- Lateral weight: 2.0-2.5 (higher because steering accuracy is critical)
- Longitudinal weight: 1.5 (lower but still important for smooth driving)

### Initialization Strategy
- Weights initialized with Kaiming initialization for ReLU activations
- Output heads scaled down (×0.1 for lateral, ×0.05 for longitudinal) for training stability

### Regularization
- Dropout: 0.1-0.2 depending on model capacity
- Weight decay: 1e-4 to 1e-5
- Gradient clipping: 1.0 for all models