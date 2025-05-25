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
    
    split_files = ['target_train.pt', 'surrogate_train.pt', 'test.pt', 'verification.pt']
    
    for split_file in split_files:
        file_path = os.path.join(data_dir, split_file)
        if not os.path.exists(file_path):
            print(f"Warning: {split_file} not found")
            continue
            
        data = torch.load(file_path)
        print(f"\n{split_file}:")
        print(f"Number of nodes: {data.x.size(0)}")
        print(f"Number of edges: {data.edge_index.size(1)}")
        
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

def main():
    parser = argparse.ArgumentParser(description='Verify data splits')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the split data')
    args = parser.parse_args()
    verify_splits(args.data_dir)

if __name__ == '__main__':
    main() 