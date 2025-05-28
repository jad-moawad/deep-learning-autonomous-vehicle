#!/usr/bin/env python3
"""
Unified evaluation script for all trajectory planning models
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import load_model
from src.data import load_data
from src.evaluation import PlannerMetric, DriveEvaluator


def evaluate_offline_metrics(model, dataloader, device):
    """Evaluate model on offline metrics"""
    metric = PlannerMetric()
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Model-specific forward pass
            if hasattr(model, 'forward'):
                model_name = model.__class__.__name__
                if 'CNN' in model_name:
                    outputs = model(image=batch['image'])
                else:
                    outputs = model(
                        track_left=batch['track_left'],
                        track_right=batch['track_right']
                    )
            
            # Update metrics
            metric.add(outputs, batch['waypoints'], batch['waypoints_mask'])
    
    return metric.compute()


def evaluate_driving_performance(model, tracks, device, max_steps=100):
    """Evaluate model performance in simulation"""
    evaluator = DriveEvaluator(model, device=device)
    results = {}
    
    for track in tracks:
        print(f"Evaluating on track: {track}")
        distance, track_length = evaluator.evaluate(
            track_name=track,
            max_steps=max_steps,
            disable_tqdm=False
        )
        
        results[track] = {
            'distance': float(distance),
            'track_length': float(track_length),
            'completion': float(distance / track_length * 100)
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory planning models")
    parser.add_argument('--model', type=str, default='all', 
                       help='Model to evaluate (mlp_planner, transformer_planner, cnn_planner, all)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--data_path', type=str, default='data/drive_data/val',
                       help='Path to validation data')
    parser.add_argument('--tracks', nargs='+', 
                       default=['lighthouse', 'hacienda', 'snowmountain'],
                       help='Tracks to evaluate on')
    parser.add_argument('--metrics_only', action='store_true',
                       help='Only compute offline metrics')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Models to evaluate
    if args.model == 'all':
        models = ['mlp_planner', 'transformer_planner', 'cnn_planner']
    else:
        models = [args.model]
    
    # Results dictionary
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    for model_name in models:
        print(f"\nEvaluating {model_name}")
        
        # Load model
        checkpoint_path = Path(args.checkpoint_dir) / model_name / 'best_model.pth'
        try:
            model = load_model(model_name, checkpoint_path=checkpoint_path, device=device)
        except FileNotFoundError:
            print(f"Checkpoint not found for {model_name}, skipping...")
            continue
        
        # Create appropriate dataloader
        if model_name == 'cnn_planner':
            transform_pipeline = 'default'
        else:
            transform_pipeline = 'state_only'
        
        dataloader = load_data(
            args.data_path,
            transform_pipeline=transform_pipeline,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate offline metrics
        print("Computing offline metrics...")
        offline_metrics = evaluate_offline_metrics(model, dataloader, device)
        
        results = {
            'offline_metrics': offline_metrics,
            'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
        }
        
        # Evaluate driving performance if requested
        if not args.metrics_only:
            print("Evaluating driving performance...")
            try:
                driving_results = evaluate_driving_performance(
                    model, args.tracks, device
                )
                results['driving_performance'] = driving_results
            except Exception as e:
                print(f"Failed to evaluate driving performance: {e}")
                results['driving_performance'] = None
        
        all_results['models'][model_name] = results
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"  L1 Error: {offline_metrics['l1_error']:.4f}")
        print(f"  Longitudinal Error: {offline_metrics['longitudinal_error']:.4f}")
        print(f"  Lateral Error: {offline_metrics['lateral_error']:.4f}")
        print(f"  Model Size: {results['model_size_mb']:.2f} MB")
        
        if 'driving_performance' in results and results['driving_performance']:
            print("  Driving Performance:")
            for track, perf in results['driving_performance'].items():
                print(f"    {track}: {perf['completion']:.1f}% completed")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Generate summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Long. Error':<12} {'Lat. Error':<12} {'Size (MB)':<10}")
    print("-"*60)
    
    for model_name, results in all_results['models'].items():
        metrics = results['offline_metrics']
        print(f"{model_name:<20} "
              f"{metrics['longitudinal_error']:<12.4f} "
              f"{metrics['lateral_error']:<12.4f} "
              f"{results['model_size_mb']:<10.2f}")


if __name__ == "__main__":
    main()