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
from datetime import datetime

# Add grove to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grove.models.gnn import GATModel, GINModel, GraphSAGEModel
from grove.attacks.model_stealing import ModelStealingAttack
from grove.attacks.evaluation import ModelEvaluator


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
    
    print(f"✅ Target model loaded successfully")
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


def save_stolen_embeddings(trainer, validation_data, embeddings_dir, model_name, dataset_name, structure):
    """
    Save stolen surrogate embeddings in the correct format for visualization.
    
    Args:
        trainer: Trained surrogate model trainer
        validation_data: Validation data to generate embeddings from
        embeddings_dir: Directory to save embeddings
        model_name: Name of the model architecture
        dataset_name: Name of the dataset
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
    
    embed_save_path = embeddings_path / f"{model_name}_{dataset_name}_surrogate_{structure}.pt"
    torch.save(embeddings_data, embed_save_path)
    
    print(f"✅ Saved stolen surrogate embeddings to: {embed_save_path}")


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
    print(f"Save detailed metrics: {args.save_detailed_metrics}")
    print("="*60)
    
    try:
        # Load target model
        target_model = load_target_model(args.target_model_path, args.model, args.device)

        # Load data splits
        query_data, validation_data, test_data = load_data_splits(args.dataset, args.split_type, args.device)
        
        # Initialize model stealing attack
        print(f"\n🎯 Initializing model stealing attack...")
        attack = ModelStealingAttack(target_model, device=args.device)
        
        # Perform model stealing attack
        print(f"\n🚀 Starting model stealing attack...")
        
        # Use attack.simple_extraction() with structure parameter
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

        
        # Comprehensive evaluation with detailed metrics
        print(f"\n📊 Performing comprehensive evaluation...")
        if args.save_detailed_metrics:
            # Use ModelEvaluator for detailed metrics
            evaluator = ModelEvaluator(device=args.device)
            
            attack_config = {
                'surrogate_architecture': args.surrogate_architecture,
                'recovery_from': args.recovery_from,
                'structure': args.structure,
                'dataset_name': args.dataset
            }
            
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
                'final_train_acc': results['train_accs'][-1] if results['train_accs'] else 0.0,
                'final_val_acc': results.get('final_val_acc', 0.0),
                'final_test_acc': results.get('final_test_acc', 0.0)
            })
            
            # Convert to DataFrame and save using evaluator's method
            results_df = pd.DataFrame([eval_results])
            
            # Save using the comprehensive evaluator's method
            evaluator.save_results_to_csv(
                results_df, 
                args.output_dir,
                f"{args.structure}_{args.model}_{args.dataset}"
            )
            
        else:
            # Basic evaluation (original functionality)
            eval_results = attack.evaluate_attack_success(trainer, test_data)
        
        # Save stolen models
        output_path = Path(args.output_dir) / f"{args.structure}_{args.model}_{args.dataset}"
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_models(str(output_path))
        
        # Save stolen embeddings
        save_stolen_embeddings(trainer, validation_data, args.embeddings_dir, args.model, args.dataset, args.structure)
        
        # Print final results
        print(f"\n" + "="*60)
        print(f"MODEL STEALING ATTACK COMPLETED!")
        print(f"="*60)
        print(f"Attack Type: {'Type I (original structure)' if args.structure == 'original' else f'Type II ({args.structure} structure)'}")
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
        
        print(f"Final Training Accuracy: {results['train_accs'][-1]:.4f}")
        print(f"Final Validation Accuracy: {results['final_val_acc']:.4f}")
        print(f"Final Test Accuracy: {results['final_test_acc']:.4f}")
        print(f"="*60)
        print(f"✅ Stolen models saved to: {output_path}")
        print(f"✅ Stolen embeddings saved to: {args.embeddings_dir}")
        if args.save_detailed_metrics:
            print(f"✅ Detailed metrics saved to CSV")
        print(f"="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model stealing attack: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 