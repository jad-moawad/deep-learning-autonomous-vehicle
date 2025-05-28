# Setup Guide

## Prerequisites

### System Requirements
- Operating System: Ubuntu 20.04+ / macOS 11+ / Windows 10+
- Python: 3.8 or higher
- CUDA: 11.3+ (for GPU support)
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU with 4GB+ VRAM (optional but recommended)

### Software Dependencies
- Git
- Conda or virtualenv
- CUDA Toolkit (for GPU support)

## Installation Steps

### 1. Clone the Repository

```
git clone https://github.com/yourusername/neural-trajectory-prediction.git
cd neural-trajectory-prediction
```

### 2. Create Virtual Environment

Using conda:

```
bashconda create -n trajectory python=3.8
conda activate trajectory
```

Using virtualenv:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3.Install PyTorch

For the GPU support:

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

For CPU only:

```
pip install torch==1.12.0 torchvision==0.13.0
```

### 4. Install Depedendicies

```
pip install -r requirements.txt
```
### 5. Download Dataset 


```
# Create data directory
mkdir -p data

# Download dataset
wget https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip -P data/
unzip data/drive_data.zip -d data/
rm data/drive_data.zip
```
### 6. Verify Installation 

```
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
