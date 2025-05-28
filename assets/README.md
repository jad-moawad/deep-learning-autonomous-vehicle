# Autonomous Vehicle Trajectory Planning with Deep Learning

A comprehensive study comparing neural network architectures for real-time trajectory prediction in autonomous driving scenarios. This project implements and evaluates three distinct approaches: MLP-based planning, Transformer-based planning with attention mechanisms, and end-to-end CNN planning from visual inputs.

<p align="center">
  <img src="assets/demo.gif" alt="Model driving demonstration" width="600">
</p>

## 🎯 Project Overview

This project explores the challenge of trajectory planning for autonomous vehicles by implementing three different neural network architectures:

1. **MLP Planner**: A baseline model that processes lane boundary information
2. **Transformer Planner**: An attention-based approach inspired by the Perceiver architecture
3. **CNN Planner**: An end-to-end vision model that directly processes camera images

Each model is designed to predict future waypoints that guide the vehicle along the track while maintaining lane discipline and appropriate speed.

## 🚀 Key Features

- **Multiple Architecture Implementations**: Compare traditional MLPs, modern Transformers, and CNN approaches
- **Real-time Performance**: Models optimized for low-latency inference suitable for autonomous driving
- **Comprehensive Evaluation**: Metrics for both longitudinal (speed) and lateral (steering) accuracy
- **Visualization Tools**: Generate videos of model performance in simulated environments
- **Modular Design**: Easy to extend with new architectures or loss functions

## 📊 Results

### Performance Metrics

| Model | Longitudinal Error ↓ | Lateral Error ↓ | Total L1 | Training Epochs |
|-------|---------------------|-----------------|----------|-----------------|
| **CNN Planner** | **0.218** | **0.267** | **0.485** | 14/20 |
| MLP Planner* | 0.19 | 0.53 | 0.72 | 25/30 |
| Transformer* | 0.16 | 0.49 | 0.65 | 18/20 |

*Results pending final training completion

### 🎯 Target Achievement

- **Required**: Longitudinal < 0.30, Lateral < 0.45
- **CNN Achieved**: Longitudinal = 0.218 (27% better), Lateral = 0.267 (41% better)


### 📈 Training Insights

The CNN model demonstrated:
- Rapid convergence in first 7 epochs
- Best performance at epoch 14
- Stable training without overfitting
- Superior performance despite end-to-end visual complexity

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/autonomous-vehicle-trajectory-planning.git
cd autonomous-vehicle-trajectory-planning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset
bash scripts/download_data.sh