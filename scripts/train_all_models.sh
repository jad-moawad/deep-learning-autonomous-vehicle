#!/bin/bash
# Train all models with their respective configurations

echo "Training all trajectory planning models..."

# Train MLP Planner
echo "Training MLP Planner..."
python -m src.training.train --config configs/mlp_config.yaml

# Train Transformer Planner
echo "Training Transformer Planner..."
python -m src.training.train --config configs/transformer_config.yaml

# Train CNN Planner
echo "Training CNN Planner..."
python -m src.training.train --config configs/cnn_config.yaml

echo "All models trained successfully!"

# Evaluate all models
echo "Evaluating all models..."
python scripts/evaluate_models.py --model all

echo "Training and evaluation complete!"