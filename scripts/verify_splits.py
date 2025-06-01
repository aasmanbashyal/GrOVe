import os
import torch
import argparse
from collections import Counter
from torch_geometric.data import Data
from torch.serialization import add_safe_globals
import numpy as np

# Add PyTorch Geometric Data class to safe globals
add_safe_globals([Data])

def verify_splits(data_dir):
    """
    Verify the data splits and print statistics.
    """
    print(f"\nVerifying splits in: {data_dir}")
    print("\nSplit Statistics:")
    print("-" * 50)
    
    # Updated list of split files for the new strategy
    split_files = ['target_train.pt', 'independent_train.pt', 'query_train.pt', 'test.pt', 'validation.pt']
    
    # For compatibility, also check old names
    fallback_files = ['surrogate_train.pt', 'verification.pt']
    
    total_nodes = 0
    split_nodes = {}
    
    # Check main split files
    for split_file in split_files:
        file_path = os.path.join(data_dir, split_file)
        if os.path.exists(file_path):
            data = torch.load(file_path)
            nodes = data.x.size(0)
            edges = data.edge_index.size(1)
            split_nodes[split_file] = nodes
            
            print(f"\n{split_file}:")
            print(f"Number of nodes: {nodes}")
            print(f"Number of edges: {edges}")
            
            if hasattr(data, 'y'):
                # Handle both one-hot encoded and raw labels
                if len(data.y.shape) == 2:  # One-hot encoded
                    labels = data.y.argmax(dim=1).numpy()
                else:  # Raw labels
                    labels = data.y.numpy()
                label_counts = Counter(labels)
                print("Label distribution:")
                for label, count in sorted(label_counts.items()):
                    print(f"  Class {label}: {count} nodes")
        else:
            print(f"Warning: {split_file} not found")
    
    # Check fallback files if main files don't exist
    for split_file in fallback_files:
        file_path = os.path.join(data_dir, split_file)
        if os.path.exists(file_path):
            data = torch.load(file_path)
            nodes = data.x.size(0)
            edges = data.edge_index.size(1)
            
            print(f"\n{split_file} (fallback):")
            print(f"Number of nodes: {nodes}")
            print(f"Number of edges: {edges}")
    
    # Load indices if available
    indices_path = os.path.join(data_dir, "split_indices.pt")
    if os.path.exists(indices_path):
        indices = torch.load(indices_path)
        print(f"\nSplit Indices Summary:")
        print("-" * 30)
        
        # Calculate total nodes from indices
        all_indices = set()
        for split_name, split_indices in indices.items():
            all_indices.update(split_indices.tolist())
            percentage = len(split_indices) / len(all_indices) * 100 if len(all_indices) > 0 else 0
            print(f"{split_name}: {len(split_indices)} nodes")
        
        total_unique_nodes = len(all_indices)
        print(f"\nTotal unique nodes across all splits: {total_unique_nodes}")
        
        # Check for overlaps in non-overlapped splits
        if 'target_train' in indices and 'query_train' in indices:
            target_set = set(indices['target_train'].tolist())
            query_set = set(indices['query_train'].tolist())
            overlap = target_set.intersection(query_set)
            
            print(f"\nData Split Verification:")
            print(f"Target/Independent training data: {len(target_set)} nodes ({len(target_set)/total_unique_nodes*100:.1f}%)")
            print(f"Query training data: {len(query_set)} nodes ({len(query_set)/total_unique_nodes*100:.1f}%)")
            print(f"Overlap between target and query: {len(overlap)} nodes")
            
            if len(overlap) == 0:
                print("✅ Non-overlapped splits verified - no overlap between training sets")
            else:
                print("❌ Warning: Overlap detected in supposedly non-overlapped splits")
        
        # Show percentages for all splits
        print(f"\nPercentage breakdown:")
        for split_name, split_indices in indices.items():
            percentage = len(split_indices) / total_unique_nodes * 100
            print(f"  {split_name}: {percentage:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Verify data splits')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing split data files')
    args = parser.parse_args()
    
    verify_splits(args.data_dir)

if __name__ == "__main__":
    main() 