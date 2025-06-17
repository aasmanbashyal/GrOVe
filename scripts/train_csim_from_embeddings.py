#!/usr/bin/env python3
"""
Train Csim similarity model using pre-saved embeddings for GNN ownership verification.

"""

import argparse
import torch
import os
import sys
import numpy as np
from pathlib import Path

# Add grove to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grove.verification.similarity_model import  CsimManager


def find_embedding_files(embeddings_dir, model_name, dataset_name, split_type='non-overlapped'):
    """
    Find embedding files for a specific model and dataset.
    
    Args:
        embeddings_dir: Base embeddings directory
        model_name: Model architecture (gat, gin, sage)
        dataset_name: Dataset name
        split_type: Data split type
        
    Returns:
        Dictionary with paths to target, surrogate, and independent embeddings
    """
    base_path = Path(embeddings_dir) / split_type / f"{model_name}_{dataset_name}"
    
    if not base_path.exists():
        raise ValueError(f"Embeddings directory not found: {base_path}")
    
    # Find target embedding
    target_pattern = f"{model_name}_{dataset_name}_target.pt"
    target_path = base_path / target_pattern
    
    if not target_path.exists():
        raise ValueError(f"Target embedding not found: {target_path}")
    
    # Find surrogate embeddings (different attack structures)
    surrogate_patterns = [
        f"{model_name}_{dataset_name}_surrogate_original.pt",
        f"{model_name}_{dataset_name}_surrogate_idgl.pt", 
    ]
    
    surrogate_paths = []
    for pattern in surrogate_patterns:
        surrogate_path = base_path / pattern
        if surrogate_path.exists():
            surrogate_paths.append(str(surrogate_path))
    
    if not surrogate_paths:
        raise ValueError(f"No surrogate embeddings found in {base_path}")
    
    independent_models = [
        "gat", "gin", "sage"
    ]
    independent_paths = []
    for independent_model in independent_models:
        # Find independent embeddings
        independent_pattern = f"{model_name}_{dataset_name}_independent_{independent_model}.pt"
        independent_path = base_path / independent_pattern
        if independent_path.exists():
            independent_paths.append(str(independent_path))
        else:
            # Use target as placeholder for independent (this is just for demo)
            print(f"Warning: Using target embedding as independent placeholder")
            independent_paths.append(str(target_path))
    
        # Use target as placeholder for independent (this is just for demo)
        print(f"Warning: Using target embedding as independent placeholder")
        independent_paths.append(str(target_path))
    
    return {
        'target': str(target_path),
        'surrogates': surrogate_paths,
        'independents': independent_paths
    }


def main():
    parser = argparse.ArgumentParser(description='Train Csim using saved embeddings')
    parser.add_argument('--embeddings-dir', type=str, default='embeddings',
                       help='Directory containing saved embeddings')
    parser.add_argument('--model', type=str, required=True, choices=['gat', 'gin', 'sage'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--split-type', type=str, default='non-overlapped',
                       choices=['non-overlapped', 'overlapped'], help='Data split type')
    parser.add_argument('--output-dir', type=str, default='models/csim',
                       help='Output directory for trained Csim models')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--use-grid-search', action='store_true', default=True,
                       help='Use grid search for hyperparameter tuning')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("="*60)
    print("CSIM TRAINING FROM SAVED EMBEDDINGS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split type: {args.split_type}")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    
    try:
        # Find embedding files
        embedding_files = find_embedding_files(
            args.embeddings_dir, args.model, args.dataset, args.split_type
        )
        
        print(f"\n Found embedding files:")
        print(f"Target: {embedding_files['target']}")
        print(f"Surrogates: {len(embedding_files['surrogates'])} files")
        for i, path in enumerate(embedding_files['surrogates']):
            print(f"  {i+1}. {path}")
        print(f"Independents: {len(embedding_files['independents'])} files")
        for i, path in enumerate(embedding_files['independents']):
            print(f"  {i+1}. {path}")
        
        # Create Csim manager
        manager = CsimManager(base_save_dir=args.output_dir)
        
        # Train Csim using saved embeddings
        target_model_name = f"{args.model}_{args.dataset}_target"
        
        print(f"\n Training Csim for: {target_model_name}")
        csim = manager.train_csim_from_embeddings(
            target_model_name=target_model_name,
            target_embedding_path=embedding_files['target'],
            surrogate_embedding_paths=embedding_files['surrogates'],
            independent_embedding_paths=embedding_files['independents'],
            device=args.device,
            use_grid_search=args.use_grid_search
        )
        
        print(f"\nSUCCESS Csim training completed successfully!")
        

        print(f"\n" + "="*60)
        print(f"CSIM TRAINING SUMMARY")
        print(f"="*60)
        print(f"SUCCESS Successfully trained Csim for {target_model_name}")
        print(f" Training data:")
        print(f"   - Target model: {args.model}")
        print(f"   - Dataset: {args.dataset}")
        print(f"   - Surrogate models: {len(embedding_files['surrogates'])}")
        print(f"   - Independent models: {len(embedding_files['independents'])}")
        print(f"💾 Csim saved to: {args.output_dir}/csim_{target_model_name}.pkl")
        print(f"="*60)
        
        return True
        
    except Exception as e:
        print(f" Error during Csim training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 