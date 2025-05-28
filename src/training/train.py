import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from ..data import load_data
from ..models import load_model, save_model
from ..evaluation import PlannerMetric
from .losses import WeightedL1Loss


class Trainer:
    """Unified trainer for all planner models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = load_model(
            config['model']['name'],
            device=self.device,
            **config['model'].get('params', {})
        )
        
        # Create dataloaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Create loss function
        self.criterion = WeightedL1Loss(
            lateral_weight=config['training'].get('lateral_weight', 2.0),
            longitudinal_weight=config['training'].get('longitudinal_weight', 1.5)
        )
        
        # Create metric
        self.metric = PlannerMetric()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_lateral_error = float('inf')
        self.patience_counter = 0
    
    def _create_dataloader(self, split):
        """Create dataloader for given split"""
        data_config = self.config['data']
        
        # Determine transform pipeline based on model
        if self.config['model']['name'] == 'cnn_planner':
            transform_pipeline = 'default'  # includes images
        else:
            transform_pipeline = 'state_only'  # only track boundaries
        
        return load_data(
            data_config['dataset_path'] + f'/{split}',
            transform_pipeline=transform_pipeline,
            batch_size=data_config['batch_size'],
            shuffle=(split == 'train'),
            num_workers=data_config.get('num_workers', 4),
        )
    
    def _create_optimizer(self):
        """Create optimizer from config"""
        opt_config = self.config['optimizer']
        opt_type = opt_config['type'].lower()
        
        if opt_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'none').lower()
        
        if sched_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5),
                verbose=True
            )
        elif sched_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['optimizer']['lr'] / 10
            )
        else:
            return None
    
    def setup_logging(self):
        """Setup tensorboard logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config['model']['name']
        log_dir = Path(self.config['training']['log_dir']) / f"{model_name}_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir']) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.metric.reset()
        
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Model-specific forward pass
            if self.config['model']['name'] == 'cnn_planner':
                outputs = self.model(image=batch['image'])
            else:
                outputs = self.model(
                    track_left=batch['track_left'],
                    track_right=batch['track_right']
                )
            
            # Calculate loss
            loss = self.criterion(
                outputs,
                batch['waypoints'],
                batch['waypoints_mask']
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.metric.add(outputs, batch['waypoints'], batch['waypoints_mask'])
            running_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        metrics = self.metric.compute()
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/train', epoch_loss, self.current_epoch)
        for key, value in metrics.items():
            self.writer.add_scalar(f'Metrics/train/{key}', value, self.current_epoch)
        
        return epoch_loss, metrics
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        self.metric.reset()
        
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.config['model']['name'] == 'cnn_planner':
                    outputs = self.model(image=batch['image'])
                else:
                    outputs = self.model(
                        track_left=batch['track_left'],
                        track_right=batch['track_right']
                    )
                
                # Calculate loss
                loss = self.criterion(
                    outputs,
                    batch['waypoints'],
                    batch['waypoints_mask']
                )
                
                # Update metrics
                self.metric.add(outputs, batch['waypoints'], batch['waypoints_mask'])
                running_loss += loss.item()
        
        # Calculate validation metrics
        val_loss = running_loss / len(self.val_loader)
        metrics = self.metric.compute()
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
        for key, value in metrics.items():
            self.writer.add_scalar(f'Metrics/val/{key}', value, self.current_epoch)
        
        return val_loss, metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.pth"
        save_model(self.model, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            save_model(self.model, best_path)
    
    def train(self):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        patience = self.config['training'].get('early_stopping_patience', 10)
        
        print(f"Starting training for {epochs} epochs")
        print(f"Model: {self.config['model']['name']}")
        print(f"Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print metrics
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Longitudinal Error: {val_metrics['longitudinal_error']:.4f}")
            print(f"  Lateral Error: {val_metrics['lateral_error']:.4f}")
            
            # Check for improvement
            if val_metrics['lateral_error'] < self.best_lateral_error:
                self.best_lateral_error = val_metrics['lateral_error']
                print(f"  New best lateral error! Saving checkpoint.")
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break
        
        print(f"\nTraining complete!")
        print(f"Best lateral error: {self.best_lateral_error:.4f}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train trajectory planning models")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()