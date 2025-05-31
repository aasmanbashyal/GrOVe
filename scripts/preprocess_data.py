import argparse
import torch
import os
from grove.utils.data_processing import GraphDataProcessor
from grove.utils.data_loader import load_npz_graph_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--overlapped', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    print(f"Overlapped splits: {args.overlapped}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")

    data = load_npz_graph_dataset(args.dataset, "data/raw")
    data = data.to('cpu')
    processor = GraphDataProcessor(random_seed=args.seed)
    splits = processor.split_graph(data, data.x, data.y, overlapped=args.overlapped)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save all split data
    for split_name, subgraph in splits.items():
        if split_name != 'indices':
            save_path = os.path.join(args.output_dir, f"{split_name}.pt")
            torch.save(subgraph, save_path)
            print(f"Saved {split_name}: {subgraph.x.shape[0]} nodes, {subgraph.edge_index.shape[1]} edges")
    
    # Save indices
    torch.save(splits['indices'], os.path.join(args.output_dir, "split_indices.pt"))
    
    # Print summary
    print(f"\nDataset split summary:")
    total_nodes = data.x.shape[0]
    for split_name, indices in splits['indices'].items():
        percentage = len(indices) / total_nodes * 100
        print(f"  {split_name}: {len(indices)} nodes ({percentage:.1f}%)")
    
    print(f"\nAll splits saved to: {args.output_dir}")

if __name__ == "__main__":
    main()