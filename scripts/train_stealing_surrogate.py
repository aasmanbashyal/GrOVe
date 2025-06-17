#!/usr/bin/env python3
"""
Script for training surrogate models using model stealing attacks.
This script loads a trained target model and performs model stealing attacks.
"""

import argparse
import torch
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data

# Add grove to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grove.models.gnn import GATModel, GINModel, GraphSAGEModel
from grove.attacks.model_stealing import ModelStealingAttack
from grove.attacks.evaluation import ModelEvaluator
from grove.attacks.advanced_models import ModelPruner


def load_target_model(model_path, model_name, device='cuda'):
    """
    Load a trained target model from checkpoint.
    
    Args:
        model_path: Path to the saved model file
        model_name: Name of the model architecture
        device: Device to load model on
        
    Returns:
        Loaded target model
    """
    print(f"Loading target model from: {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    
    # Create model architecture
    if model_name == 'gat':
        model = GATModel(
            in_feats=model_config['input_dim'],
            h_feats=model_config['hidden_dim'],
            out_feats=model_config['output_dim'],
            num_heads=4,
            num_layers=3,
            dropout=model_config['dropout']
        )
    elif model_name == 'gin':
        model = GINModel(
            in_feats=model_config['input_dim'],
            h_feats=model_config['hidden_dim'],
            out_feats=model_config['output_dim'],
            num_layers=3,
            dropout=model_config['dropout']
        )
    elif model_name == 'sage':
        model = GraphSAGEModel(
            in_feats=model_config['input_dim'],
            h_feats=model_config['hidden_dim'],
            out_feats=model_config['output_dim'],
            num_layers=2,
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"SUCCESS Target model loaded successfully")
    print(f"Model config: {model_config}")
    
    return model


def load_data_splits(dataset_name, split_type, device):
    """
    Load data splits for model stealing.
    
    Args:
        dataset_name: Name of the dataset
        split_type: Type of split (non-overlapped/overlapped)
        device: Device to load data on
        
    Returns:
        Tuple of (query_data, validation_data, test_data)
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


def perform_advanced_attack(attack_type, target_model, query_data, validation_data, test_data, 
                          surrogate_architecture, args):
    """
    Perform advanced model stealing attacks.
    
    Args:
        attack_type: Type of advanced attack ('fine_tuning', 'double_extraction', 'pruning', 'distribution_shift')
        target_model: Target model to steal from
        query_data: Data for querying target
        validation_data: Validation data
        test_data: Test data
        surrogate_architecture: Architecture for surrogate model
        args: Command line arguments
        
    Returns:
        Tuple of (trainer/model, results, attack_info)
    """
    print(f"\n Performing {attack_type} attack...")
    
    if attack_type == 'distribution_shift':
        # Use the corrected distribution_shift method from ModelStealingAttack
        print("Using corrected distribution shift implementation...")
        attack = ModelStealingAttack(target_model, device=args.device)
        
        trainer, results = attack.distribution_shift(
            query_data=query_data,
            val_data=validation_data,
            test_data=test_data,
            surrogate_architecture=surrogate_architecture,
            recovery_from=args.recovery_from,
            structure=args.structure,
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            lr=0.001,
            shift_intensity=0.3
        )
        
        attack_info = {
            'attack_type': 'distribution_shift',
            'shift_intensity': 0.3,
            'adversarial_training': True,
            'using_corrected_implementation': True
        }
        
        print(f"Distribution shift completed using corrected implementation!")
        print(f"Final test accuracy: {results.get('final_test_acc', 0.0):.4f}")
        
        return trainer, results, attack_info
        
    elif attack_type == 'double_extraction':
        # Use the corrected double extraction method from ModelStealingAttack
        print("Using corrected double extraction implementation...")
        attack = ModelStealingAttack(target_model, device=args.device)
        
        trainer, results = attack.double_extraction(
            query_data=query_data,
            val_data=validation_data,
            test_data=test_data,
            surrogate_architecture=surrogate_architecture,
            recovery_from=args.recovery_from,
            structure=args.structure,
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            lr=0.001,
            experiment_id=42  # For reproducible data splitting
        )
        
        # Extract metrics from the combined results
        stage1_results = results.get('first_extraction', {})
        stage2_results = results.get('second_extraction', {})
        
        # The results are already properly structured by the corrected method
        # Just add any missing attack info fields
        attack_info = {
            'attack_type': 'double_extraction',
            'stage1_epochs': args.epochs,
            'stage2_epochs': args.epochs,
            'using_corrected_implementation': True,
            'proper_data_splitting': True
        }
        
        attack_info.update({
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'intermediate_acc': stage1_results.get('final_test_acc', 0.0),
            'final_acc': stage2_results.get('final_test_acc', 0.0),
            'final_fidelity': results.get('final_fidelity', 0.0)
        })
        
        print(f"Double extraction completed using corrected implementation!")
        print(f"Stage 1 accuracy: {attack_info['intermediate_acc']:.4f}")
        print(f"Stage 2 accuracy: {attack_info['final_acc']:.4f}")
        print(f"Final fidelity: {attack_info['final_fidelity']:.4f}")
        
        return trainer, results, attack_info
        
    elif attack_type == 'fine_tuning':
        # Use the corrected fine_tuning method from ModelStealingAttack
        print("Using corrected fine tuning implementation...")
        attack = ModelStealingAttack(target_model, device=args.device)
        
        trainer, results = attack.fine_tuning(
            query_data=query_data,
            val_data=validation_data,
            test_data=test_data,
            surrogate_architecture=surrogate_architecture,
            recovery_from=args.recovery_from,
            structure=args.structure,
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            lr=0.001,
            experiment_id=42,  # For reproducible data splitting
            fine_tune_ratio=0.4,  # 40% of data for fine-tuning
            fine_tune_lr_factor=0.1  # Reduced learning rate for fine-tuning
        )
        
        attack_info = {
            'attack_type': 'fine_tuning',
            'initial_epochs': args.epochs,
            'fine_tune_epochs': args.epochs // 2,
            'fine_tune_ratio': 0.4,
            'fine_tune_lr_factor': 0.1,
            'using_corrected_implementation': True,
            'proper_data_splitting': True
        }
        
        print(f"Fine tuning completed using corrected implementation!")
        print(f"Final test accuracy: {results.get('final_test_acc', 0.0):.4f}")
        
        return trainer, results, attack_info
        
    elif attack_type == 'pruning':
        # First train a basic surrogate
        print("Stage 1: Training basic surrogate...")
        attack = ModelStealingAttack(target_model, device=args.device)
        trainer, initial_results = attack.simple_extraction(
            query_data=query_data,
            val_data=validation_data,
            test_data=test_data,
            surrogate_architecture=surrogate_architecture,
            recovery_from=args.recovery_from,
            structure=args.structure,
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs
        )
        
        # Stage 2: Apply pruning with debugging
        print("Stage 2: Applying pruning...")
        import copy
        
        # Store original models for comparison
        original_surrogate = copy.deepcopy(trainer.surrogate_model)
        original_classifier = copy.deepcopy(trainer.classifier)
        
        # Evaluate original performance
        original_acc = trainer.evaluate_surrogate(test_data)
        print(f"DEBUG Original model accuracy: {original_acc:.4f}")
        
        # Apply pruning using the corrected method
        pruner = ModelPruner(pruning_ratio=args.pruning_ratio)
        
        # Use the specified pruning method
        print(f"Applying random pruning with ratio: {args.pruning_ratio}")
        pruned_surrogate = pruner.random_prune(trainer.surrogate_model)
        pruned_classifier = pruner.random_prune(trainer.classifier)
        
        # Verify actual pruning ratios
        actual_surrogate_ratio = pruner.verify_pruning_ratio(original_surrogate, pruned_surrogate)
        actual_classifier_ratio = pruner.verify_pruning_ratio(original_classifier, pruned_classifier)
        
        print(f"DEBUG Pruning Verification:")
        print(f"  Requested pruning ratio: {args.pruning_ratio}")
        print(f"  Actual surrogate pruning ratio: {actual_surrogate_ratio:.3f}")
        print(f"  Actual classifier pruning ratio: {actual_classifier_ratio:.3f}")
        
        # Update trainer with pruned models
        trainer.surrogate_model = pruned_surrogate
        trainer.classifier = pruned_classifier
        
        # Evaluate pruned performance
        pruned_acc = trainer.evaluate_surrogate(test_data)
        accuracy_drop = original_acc - pruned_acc
        
        print(f"DEBUG Performance Impact:")
        print(f"  Original accuracy: {original_acc:.4f}")
        print(f"  Pruned accuracy: {pruned_acc:.4f}")
        print(f"  Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop/original_acc*100:.1f}%)")

        attack_info = {
            'attack_type': 'pruning',
            'pruning_ratio': args.pruning_ratio,
            'pruning_method': 'random',
            'actual_surrogate_ratio': actual_surrogate_ratio,
            'actual_classifier_ratio': actual_classifier_ratio,
            'original_accuracy': original_acc,
            'pruned_accuracy': pruned_acc,
            'accuracy_drop': accuracy_drop
        }
        
        return trainer, initial_results, attack_info
        
    else:
        raise ValueError(f"Unknown advanced attack type: {attack_type}")


def save_stolen_embeddings(trainer, validation_data, embeddings_dir, model_name, dataset_name, structure, attack_type=None):
    """
    Save stolen surrogate embeddings in the correct format for visualization.
    
    Args:
        trainer: Trained surrogate model trainer
        validation_data: Validation data to generate embeddings from
        embeddings_dir: Directory to save embeddings
        model_name: Name of the model architecture
        dataset_name: Name of the dataset
        structure: Graph structure used
        attack_type: Type of advanced attack (if any)
    """
    trainer.surrogate_model.eval()
    
    with torch.no_grad():
        # Generate embeddings using surrogate model
        surrogate_embs = trainer.surrogate_model(validation_data).detach().cpu()
    
    # Create Data object for embeddings in the format expected by visualization script
    verification_data = Data(
        x=surrogate_embs,
        y=None,
        embedding_type='stolen_surrogate_verification'
    )
    
    # Save embeddings in the same format as regular model training
    embeddings_data = {
        'best_embeddings': {
            'verification': verification_data
        },
        'final_embeddings': {
            'verification': verification_data
        },
        'best_epoch': 0,  
        'best_val_acc': 0.0,  
        'final_train_acc': 0.0,  
        'final_val_acc': 0.0   
    }
    
    # Save embeddings
    embeddings_path = Path(embeddings_dir)
    embeddings_path.mkdir(parents=True, exist_ok=True)
    
    # Include attack type in filename if specified
    if attack_type:
        embed_save_path = embeddings_path / f"{model_name}_{dataset_name}_surrogate_{structure}_{attack_type}.pt"
    else:
        embed_save_path = embeddings_path / f"{model_name}_{dataset_name}_surrogate_{structure}.pt"
    
    torch.save(embeddings_data, embed_save_path)
    
    print(f"SUCCESS Saved stolen surrogate embeddings to: {embed_save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train surrogate model using model stealing attacks')
    parser.add_argument('--target-model-path', type=str, required=True,
                       help='Path to trained target model')
    parser.add_argument('--model', type=str, required=True, choices=['gat', 'gin', 'sage'],
                       help='Target model architecture')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--split-type', type=str, default='non-overlapped',
                       choices=['non-overlapped', 'overlapped'], help='Data split type')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for stolen models')
    parser.add_argument('--embeddings-dir', type=str, required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--surrogate-architecture', type=str, default=None,
                       choices=['gat', 'gin', 'sage'], help='Surrogate model architecture (default: same as target)')
    parser.add_argument('--recovery-from', type=str, default='embedding',
                       choices=['embedding', 'prediction'], help='What to recover from target')
    parser.add_argument('--structure', type=str, default='original',
                       choices=['original', 'idgl', 'knn', 'random'], 
                       help='Graph structure to use (original=Type I, others=Type II)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for surrogate model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-detailed-metrics', action='store_true',
                       help='Save detailed metrics including FPR/FNR to CSV')
    parser.add_argument('--advanced-attack', type=str, default=None,
                       choices=['fine_tuning', 'double_extraction', 'pruning', 'distribution_shift'],
                       help='Type of advanced attack to perform')
    parser.add_argument('--pruning-ratio', type=float, default=0.3,
                       help='Pruning ratio for pruning attack (default: 0.3)')
    
    args = parser.parse_args()
    
    # Set surrogate architecture to same as target if not specified
    if args.surrogate_architecture is None:
        args.surrogate_architecture = args.model
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("="*60)
    print("MODEL STEALING ATTACK")
    print("="*60)
    print(f"Target model: {args.model}")
    print(f"Surrogate architecture: {args.surrogate_architecture}")
    print(f"Recovery from: {args.recovery_from}")
    print(f"Structure: {args.structure} ({'Type I' if args.structure == 'original' else 'Type II'})")
    print(f"Dataset: {args.dataset}")
    print(f"Split type: {args.split_type}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Advanced attack: {args.advanced_attack or 'None (basic attack)'}")
    if args.advanced_attack == 'pruning':
        print(f"Pruning ratio: {args.pruning_ratio}")
    print(f"Save detailed metrics: {args.save_detailed_metrics}")
    print("="*60)
    
    try:
        # Load target model
        target_model = load_target_model(args.target_model_path, args.model, args.device)

        # Load data splits
        query_data, validation_data, test_data = load_data_splits(args.dataset, args.split_type, args.device)
        
        if args.advanced_attack:
            trainer, results, attack_info = perform_advanced_attack(
                args.advanced_attack, target_model, query_data, validation_data, test_data,
                args.surrogate_architecture, args
            )
        else:
            print(f"\n Initializing basic model stealing attack...")
            attack = ModelStealingAttack(target_model, device=args.device)
            
            print(f"\n Starting basic model stealing attack...")
            trainer, results = attack.simple_extraction(
                query_data=query_data,
                val_data=validation_data,
                test_data=test_data,
                surrogate_architecture=args.surrogate_architecture,
                recovery_from=args.recovery_from,
                structure=args.structure,
                dataset_name=args.dataset,
                hidden_dim=args.hidden_dim,
                num_epochs=args.epochs
            )
            attack_info = {'attack_type': 'basic'}

        
        # Comprehensive evaluation with detailed metrics
        print(f"\n Performing comprehensive evaluation...")
        if args.save_detailed_metrics:
            # Use ModelEvaluator for detailed metrics
            evaluator = ModelEvaluator(device=args.device)
            
            attack_config = {
                'surrogate_architecture': args.surrogate_architecture,
                'recovery_from': args.recovery_from,
                'structure': args.structure,
                'dataset_name': args.dataset,
                'advanced_attack': args.advanced_attack
            }
            attack_config.update(attack_info)
            
            # Get comprehensive evaluation results
            eval_results = evaluator.evaluate_surrogate_model(
                target_model, trainer, test_data, args.dataset, attack_config
            )
            
            # Add training and configuration metrics to the evaluation results
            eval_results.update({
                'target_model_architecture': args.model,
                'split_type': args.split_type,
                'epochs': args.epochs,
                'hidden_dim': args.hidden_dim,
                'final_train_acc': results.get('train_accs', [0.0])[-1] if results.get('train_accs') else 0.0,
                'final_val_acc': results.get('final_val_acc', 0.0),
                'final_test_acc': results.get('final_test_acc', 0.0)
            })
            eval_results.update(attack_info)
            
            # Convert to DataFrame and save using evaluator's method
            results_df = pd.DataFrame([eval_results])
            
            # Save using the comprehensive evaluator's method
            attack_suffix = f"_{args.advanced_attack}" if args.advanced_attack else ""
            evaluator.save_results_to_csv(
                results_df, 
                args.output_dir,
                f"{args.structure}_{args.model}_{args.dataset}{attack_suffix}"
            )
            
        else:
            # Basic evaluation (original functionality)
            attack = ModelStealingAttack(target_model, device=args.device)
            eval_results = attack.evaluate_attack_success(trainer, test_data)
        
        # Save stolen models
        attack_suffix = f"_{args.advanced_attack}" if args.advanced_attack else ""
        output_path = Path(args.output_dir) / f"{args.structure}_{args.model}_{args.dataset}{attack_suffix}"
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_models(str(output_path))
        
        # Save stolen embeddings
        save_stolen_embeddings(trainer, validation_data, args.embeddings_dir, args.model, 
                             args.dataset, args.structure, args.advanced_attack)
        
        # Print final results
        print(f"\n" + "="*60)
        print(f"MODEL STEALING ATTACK COMPLETED!")
        print(f"="*60)
        attack_type_desc = f"Advanced ({args.advanced_attack})" if args.advanced_attack else "Basic"
        print(f"Attack Type: {attack_type_desc} - {'Type I (original structure)' if args.structure == 'original' else f'Type II ({args.structure} structure)'}")
        print(f"Target Accuracy: {eval_results.get('target_accuracy', 0.0):.4f}")
        print(f"Surrogate Accuracy: {eval_results.get('surrogate_accuracy', 0.0):.4f}")
        print(f"Fidelity (Agreement): {eval_results.get('fidelity', 0.0):.4f}")
        print(f"Accuracy Gap: {eval_results.get('accuracy_gap', 0.0):.4f}")
        
        if args.save_detailed_metrics:
            print(f"Target FPR: {eval_results.get('target_fpr', 0.0):.4f}")
            print(f"Target FNR: {eval_results.get('target_fnr', 0.0):.4f}")
            print(f"Surrogate FPR: {eval_results.get('surrogate_fpr', 0.0):.4f}")
            print(f"Surrogate FNR: {eval_results.get('surrogate_fnr', 0.0):.4f}")
            print(f"FPR Gap: {eval_results.get('fpr_gap', 0.0):.4f}")
            print(f"FNR Gap: {eval_results.get('fnr_gap', 0.0):.4f}")
            print(f"Embedding Similarity: {eval_results.get('embedding_cosine_similarity', 0.0):.4f}")
        
        print(f"Final Training Accuracy: {results.get('train_accs', [0.0])[-1] if results.get('train_accs') else 0.0:.4f}")
        print(f"Final Validation Accuracy: {results.get('final_val_acc', 0.0):.4f}")
        print(f"Final Test Accuracy: {results.get('final_test_acc', 0.0):.4f}")
        
        # Print attack-specific info
        if args.advanced_attack:
            print(f"\nAdvanced Attack Details:")
            for key, value in attack_info.items():
                if key != 'attack_type':
                    print(f"  {key}: {value}")
        
        print(f"="*60)
        print(f"SUCCESS Stolen models saved to: {output_path}")
        print(f"SUCCESS Stolen embeddings saved to: {args.embeddings_dir}")
        if args.save_detailed_metrics:
            print(f"SUCCESS Detailed metrics saved to CSV")
        print(f"="*60)
        
        return True
        
    except Exception as e:
        print(f" Error during model stealing attack: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)