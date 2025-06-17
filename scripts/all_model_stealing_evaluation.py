#!/usr/bin/env python3
"""
Comprehensive model stealing evaluation script.
This script evaluates all model architectures against all attack configurations
and provides detailed metrics including accuracy, fidelity, FPR, FNR with CSV output.
"""

import argparse
import torch
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
import json
from datetime import datetime

# Add grove to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grove.models.gnn import GATModel, GINModel, GraphSAGEModel
from grove.attacks.model_stealing import ModelStealingAttack
from grove.attacks.evaluation import ModelEvaluator
from grove.attacks.model_stealing import DistributionShiftTrainer


def load_all_target_models(models_dir, dataset_name, device='cuda', split_type='non-overlapped'):
    """
    Load all trained target models for comprehensive evaluation.
    
    Args:
        models_dir: Directory containing trained models
        dataset_name: Name of the dataset
        device: Device to load models on
        split_type: Data split type (non-overlapped or overlapped)
        
    Returns:
        Dictionary of loaded models {model_name: model}
    """
    models = {}
    model_types = ['gat', 'gin', 'sage']
    
    for model_type in model_types:
        # Look for the correct file pattern: {model}_{dataset}_target_{split_type}.pt
        model_path = Path(models_dir) / f"{model_type}_{dataset_name}" / f"{model_type}_{dataset_name}_target_{split_type}.pt"
        
        if model_path.exists():
            print(f"Loading {model_type} model from: {model_path}")
            
            try:
                # Load model checkpoint
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model_config = checkpoint['model_config']
                
                # Create model architecture
                if model_type == 'gat':
                    model = GATModel(
                        in_feats=model_config['input_dim'],
                        h_feats=model_config['hidden_dim'],
                        out_feats=model_config['output_dim'],
                        num_heads=4,
                        num_layers=3,
                        dropout=model_config['dropout']
                    )
                elif model_type == 'gin':
                    model = GINModel(
                        in_feats=model_config['input_dim'],
                        h_feats=model_config['hidden_dim'],
                        out_feats=model_config['output_dim'],
                        num_layers=3,
                        dropout=model_config['dropout']
                    )
                elif model_type == 'sage':
                    model = GraphSAGEModel(
                        in_feats=model_config['input_dim'],
                        h_feats=model_config['hidden_dim'],
                        out_feats=model_config['output_dim'],
                        num_layers=2,
                        dropout=model_config['dropout']
                    )
                
                # Load trained weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                
                models[model_type] = model
                print(f"SUCCESS: {model_type} model loaded successfully")
                
            except Exception as e:
                print(f"ERROR: Failed to load {model_type} model: {e}")
        else:
            print(f"WARNING: Model file not found: {model_path}")
    
    return models


def load_data_splits(dataset_name, split_type, device):
    """
    Load data splits for model stealing evaluation.
    """
    # Add safe globals for loading PyG data
    torch.serialization.add_safe_globals([Data])
    
    data_dir = f"data/processed/{split_type}/{dataset_name}"
    print(f"Loading data from: {data_dir}")
    
    # Load query data 
    query_train_path = os.path.join(data_dir, 'query_train.pt')
    if os.path.exists(query_train_path):
        query_data = torch.load(query_train_path, weights_only=False)
        print(f"Loaded query_train data: {query_data.x.shape[0]} nodes")
    else:
        # Fallback to surrogate_train
        query_data = torch.load(os.path.join(data_dir, 'surrogate_train.pt'), weights_only=False)
        print(f"Using surrogate_train data for querying: {query_data.x.shape[0]} nodes")
    
    # Load validation and test data
    validation_path = os.path.join(data_dir, 'validation.pt')
    if os.path.exists(validation_path):
        validation_data = torch.load(validation_path, weights_only=False)
        print(f"Loaded validation data: {validation_data.x.shape[0]} nodes")
    else:
        validation_data = torch.load(os.path.join(data_dir, 'verification.pt'), weights_only=False)
        print(f"Using verification data for validation: {validation_data.x.shape[0]} nodes")
    
    test_data = torch.load(os.path.join(data_dir, 'test.pt'), weights_only=False)
    print(f"Loaded test data: {test_data.x.shape[0]} nodes")
    
    # Move data to device
    query_data = query_data.to(device)
    validation_data = validation_data.to(device)
    test_data = test_data.to(device)
    
    # Ensure labels are in correct format 
    if len(query_data.y.shape) == 2:  # One-hot encoded
        query_data.y = query_data.y.argmax(dim=1)
    if len(validation_data.y.shape) == 2:
        validation_data.y = validation_data.y.argmax(dim=1)
    if len(test_data.y.shape) == 2:
        test_data.y = test_data.y.argmax(dim=1)
    
    return query_data, validation_data, test_data


def evaluate_cross_architecture_attacks(target_models, 
                                       query_data, 
                                       val_data, 
                                       test_data,
                                       dataset_name,
                                       evaluator,
                                       num_epochs=50):
    """
    Evaluate cross-architecture attacks (GAT attacking GIN, etc.).
    
    Args:
        target_models: Dictionary of target models
        query_data: Query data
        val_data: Validation data  
        test_data: Test data
        dataset_name: Dataset name
        evaluator: ModelEvaluator instance
        num_epochs: Number of training epochs
        
    Returns:
        DataFrame with cross-architecture attack results
    """
    print("\nSTARTING: Cross-architecture attack evaluation...")
    
    cross_results = []
    
    for target_arch, target_model in target_models.items():
        for surrogate_arch in ['gat', 'gin', 'sage']:
            for recovery_method in ['embedding', 'prediction']:
                for structure in ['original', 'idgl', 'knn']:
                    
                    config_name = f"{target_arch}_vs_{surrogate_arch}_{recovery_method}_{structure}"
                    print(f"\nTesting: {config_name}")
                    
                    try:
                        # Initialize attack
                        attack = ModelStealingAttack(target_model, device=evaluator.device)
                        
                        # Attack configuration
                        attack_config = {
                            'surrogate_architecture': surrogate_arch,
                            'recovery_from': recovery_method,
                            'structure': structure,
                            'dataset_name': dataset_name
                        }
                        
                        # Perform attack
                        trainer, training_results = attack.simple_extraction(
                            query_data=query_data,
                            val_data=val_data,
                            test_data=test_data,
                            num_epochs=num_epochs,
                            **attack_config
                        )
                        
                        # Evaluate results
                        eval_metrics = evaluator.evaluate_surrogate_model(
                            target_model, trainer, test_data, dataset_name, attack_config
                        )
                        
                        # Add cross-architecture specific info
                        eval_metrics.update({
                            'target_architecture': target_arch,
                            'config_name': config_name,
                            'cross_architecture': target_arch != surrogate_arch,
                            'final_train_acc': training_results['train_accs'][-1] if training_results['train_accs'] else 0.0,
                            'final_val_acc': training_results.get('final_val_acc', 0.0),
                            'final_test_acc': training_results.get('final_test_acc', 0.0)
                        })
                        
                        cross_results.append(eval_metrics)
                        
                        print(f"SUCCESS - Fidelity: {eval_metrics['fidelity']:.4f}, "
                              f"Surrogate Acc: {eval_metrics['surrogate_accuracy']:.4f}")
                        
                    except Exception as e:
                        print(f"ERROR: Failed: {e}")
                        # Add failed configuration
                        failed_metrics = {
                            'dataset': dataset_name,
                            'target_architecture': target_arch,
                            'surrogate_architecture': surrogate_arch,
                            'recovery_from': recovery_method,
                            'attack_type': structure,
                            'config_name': config_name,
                            'cross_architecture': target_arch != surrogate_arch,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        # Fill other metrics with NaN
                        for key in ['target_accuracy', 'surrogate_accuracy', 'fidelity', 'target_fpr', 'target_fnr', 
                                   'surrogate_fpr', 'surrogate_fnr', 'accuracy_gap', 'fpr_gap', 'fnr_gap']:
                            failed_metrics[key] = np.nan
                        
                        cross_results.append(failed_metrics)
    
    return pd.DataFrame(cross_results)


def evaluate_advanced_attacks(target_models,
                            query_data,
                            val_data, 
                            test_data,
                            dataset_name,
                            evaluator,
                            num_epochs=50):
    """
    Evaluate advanced attacks including adversarial training and distribution shift.
    
    Args:
        target_models: Dictionary of target models
        query_data: Query data
        val_data: Validation data
        test_data: Test data  
        dataset_name: Dataset name
        evaluator: ModelEvaluator instance
        num_epochs: Number of training epochs
        
    Returns:
        DataFrame with advanced attack results
    """
    print("\nSTARTING: Advanced attack evaluation...")
    
    advanced_results = []
    
    for target_arch, target_model in target_models.items():
        print(f"\nTesting advanced attacks against {target_arch} model...")
        
        # Test distribution shift trainer
        try:
            print("Testing distribution shift attack...")
            
            advanced_trainer = DistributionShiftTrainer(
                target_model=target_model,
                surrogate_architecture='gat',  # Use GAT for advanced attacks
                recovery_from='embedding',
                structure='original',
                device=evaluator.device,
                shift_intensity=0.3
            )
            
            # Train with distribution shift
            training_results = advanced_trainer.train_surrogate(
                query_data, val_data, test_data, 
                num_epochs=num_epochs
            )
            
            # Evaluate using standard metrics
            eval_metrics = evaluator.evaluate_surrogate_model(
                target_model, advanced_trainer, test_data, dataset_name, 
                {'attack_type': 'distribution_shift', 'surrogate_architecture': 'gat', 'recovery_from': 'embedding'}
            )
            
            fidelity = eval_metrics['fidelity']
            surrogate_acc = eval_metrics['surrogate_accuracy']
            target_acc = eval_metrics['target_accuracy']
            
            advanced_metrics = eval_metrics.copy()
            advanced_metrics.update({
                'surrogate_architecture': 'gat_advanced',
                'attack_type': 'distribution_shift',
                'advanced_attack': True,
                'shift_intensity': 0.3,
                'final_train_acc': training_results['train_accs'][-1] if training_results['train_accs'] else 0.0,
                'timestamp': datetime.now().isoformat()
            })
            
            advanced_results.append(advanced_metrics)
            
            print(f"SUCCESS: Advanced attack - Fidelity: {fidelity:.4f}, Surrogate Acc: {surrogate_acc:.4f}")
            
        except Exception as e:
            print(f"ERROR: Advanced attack failed: {e}")
            
            failed_advanced = {
                'dataset': dataset_name,
                'target_architecture': target_arch,
                'surrogate_architecture': 'gat_advanced',
                'attack_type': 'distribution_shift',
                'recovery_from': 'embedding',
                'advanced_attack': True,
                'shift_intensity': 0.3,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            # Fill other metrics with NaN
            for key in ['target_accuracy', 'surrogate_accuracy', 'fidelity', 'target_fpr', 'target_fnr', 
                       'surrogate_fpr', 'surrogate_fnr', 'accuracy_gap', 'fpr_gap', 'fnr_gap']:
                failed_advanced[key] = np.nan
            
            advanced_results.append(failed_advanced)
    
    return pd.DataFrame(advanced_results)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model stealing evaluation')
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directory containing trained target models')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--split-type', type=str, default='non-overlapped',
                       choices=['non-overlapped', 'overlapped'], help='Data split type')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs per attack')
    parser.add_argument('--include-advanced', action='store_true',
                       help='Include advanced attacks (slower)')
    parser.add_argument('--architectures', nargs='+', default=['gat', 'gin', 'sage'],
                       choices=['gat', 'gin', 'sage'], help='Surrogate architectures to test')
    parser.add_argument('--recovery-methods', nargs='+', default=['embedding', 'prediction'],
                       choices=['embedding', 'prediction'], help='Recovery methods to test')
    parser.add_argument('--structures', nargs='+', default=['original', 'idgl', 'knn'],
                       choices=['original', 'idgl', 'knn', 'random'], help='Graph structures to test')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL STEALING ATTACK EVALUATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Split type: {args.split_type}")
    print(f"Device: {args.device}")
    print(f"Epochs per attack: {args.epochs}")
    print(f"Surrogate architectures: {args.architectures}")
    print(f"Recovery methods: {args.recovery_methods}")
    print(f"Graph structures: {args.structures}")
    print(f"Include advanced attacks: {args.include_advanced}")
    print("=" * 80)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(device=args.device)
        
        # Load all target models
        print("\nLoading target models...")
        target_models = load_all_target_models(args.models_dir, args.dataset, args.device, args.split_type)
        
        if not target_models:
            print("ERROR: No target models found!")
            return False
        
        print(f"SUCCESS: Loaded {len(target_models)} target models: {list(target_models.keys())}")
        
        # Load data splits
        print("\nLoading data splits...")
        query_data, val_data, test_data = load_data_splits(args.dataset, args.split_type, args.device)
        
        # Create output directory only when needed
        output_path = Path(args.output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_path}")
        
        all_results = []
        
        # 1. Standard cross-architecture evaluation
        print("\n" + "=" * 60)
        print("PHASE 1: CROSS-ARCHITECTURE ATTACK EVALUATION")
        print("=" * 60)
        
        cross_results_df = evaluate_cross_architecture_attacks(
            target_models, query_data, val_data, test_data,
            args.dataset, evaluator, args.epochs
        )
        
        all_results.append(cross_results_df)
        
        # 2. Advanced attacks (if requested)
        if args.include_advanced:
            print("\n" + "=" * 60)
            print("PHASE 2: ADVANCED ATTACK EVALUATION")
            print("=" * 60)
            
            advanced_results_df = evaluate_advanced_attacks(
                target_models, query_data, val_data, test_data,
                args.dataset, evaluator, args.epochs
            )
            
            all_results.append(advanced_results_df)
        
        # 3. Combine all results
        print("\nCombining all evaluation results...")
        combined_results_df = pd.concat(all_results, ignore_index=True)
        
        # 4. Save results to CSV
        print("\nSaving evaluation results...")
        csv_path = evaluator.save_results_to_csv(
            combined_results_df, 
            args.output_dir,
            f"comprehensive_evaluation_{args.dataset}"
        )
        
        # 5. Print final summary
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION COMPLETED!")
        print("=" * 80)
        
        successful_attacks = combined_results_df.dropna(subset=['fidelity'])
        
        print(f"SUMMARY: Total configurations tested: {len(combined_results_df)}")
        print(f"SUCCESS: Successful attacks: {len(successful_attacks)}")
        print(f"ERROR: Failed attacks: {len(combined_results_df) - len(successful_attacks)}")
        print(f"RATE: Success rate: {len(successful_attacks)/len(combined_results_df)*100:.2f}%")
        
        if len(successful_attacks) > 0:
            print(f"METRIC: Average fidelity: {successful_attacks['fidelity'].mean():.4f}")
            print(f"METRIC: Best fidelity: {successful_attacks['fidelity'].max():.4f}")
            print(f"METRIC: Worst fidelity: {successful_attacks['fidelity'].min():.4f}")
        
        print(f"SAVED: Results saved to: {csv_path}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error during comprehensive evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 