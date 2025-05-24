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

    data = load_npz_graph_dataset(args.dataset, "data/raw")
    data = data.to('cpu')
    processor = GraphDataProcessor(random_seed=args.seed)
    splits = processor.split_graph(data, data.x, data.y, overlapped=args.overlapped)

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, subgraph in splits.items():
        if split_name != 'indices':
            torch.save(subgraph, os.path.join(args.output_dir, f"{split_name}.pt"))
    torch.save(splits['indices'], os.path.join(args.output_dir, "split_indices.pt"))

if __name__ == "__main__":
    main()