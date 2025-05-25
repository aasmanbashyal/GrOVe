"""
Graph data processing and model training utilities for GrOVe.
"""

import torch
from torch_geometric.data import Data
import logging

# Setup logging
logger = logging.getLogger(__name__)

class GraphDataProcessor:
    """
    Utility class for processing graph data and preparing it for model training.
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the data processor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
    
    def split_graph(self, graph, node_features, node_labels, overlapped: bool = False):
        """
        Split the graph into training and evaluation splits with overlapped or non-overlapped options.
        
        Args:
            graph: PyTorch Geometric data object
            node_features: Node features tensor
            node_labels: Node labels tensor
            overlapped: Whether to use overlapped splits (default False).
            
        Non-overlapped (default):
            target_train: 40% of nodes
            surrogate_train: 40% (disjoint)
            test: 10%
            verification: 10%
            
        Overlapped:
            target_train: 45% of nodes
            surrogate_train: 25% of target_train (11.25% total)
            test: 45% of nodes
            verification: 10% of nodes
            
        Returns:
            Dictionary containing split subgraphs and indices.
        """
        # Get total number of nodes
        num_nodes = node_features.size(0)
        
        # Create shuffled indices
        indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(self.random_seed))
        
        if overlapped:
            # Overlapped splits: verification (10%), test (45%), target (45%), surrogate (25% of target)
            val_size = int(num_nodes * 0.1)
            test_size = int(num_nodes * 0.45)
            target_size = num_nodes - val_size - test_size
            surrogate_size = int(target_size * 0.25)
            
            verification_indices = indices[:val_size]
            test_indices = indices[val_size: val_size + test_size]
            target_indices = indices[val_size + test_size: val_size + test_size + target_size]
            surrogate_indices = target_indices[:surrogate_size]
        else:
            # Non-overlapped splits: target (40%), surrogate (40%), test (10%), verification (10%)
            target_size = int(num_nodes * 0.4)
            surrogate_size = int(num_nodes * 0.4)
            test_size = int(num_nodes * 0.1)
            
            target_indices = indices[:target_size]
            surrogate_indices = indices[target_size: target_size + surrogate_size]
            test_indices = indices[target_size + surrogate_size: target_size + surrogate_size + test_size]
            verification_indices = indices[target_size + surrogate_size + test_size:]
        
        # Create subgraphs for each split
        splits = {
            'target_train': self._create_subgraph(graph, target_indices),
            'surrogate_train': self._create_subgraph(graph, surrogate_indices),
            'test': self._create_subgraph(graph, test_indices),
            'verification': self._create_subgraph(graph, verification_indices)
        }
        
        # Store indices for reference
        splits['indices'] = {
            'target_train': target_indices,
            'surrogate_train': surrogate_indices,
            'test': test_indices,
            'verification': verification_indices
        }
        
        return splits
    
    def _create_subgraph(self, graph, node_indices):
        """
        Create a subgraph from the selected node indices.
        
        Args:
            graph: PyTorch Geometric data object
            node_indices: Indices of nodes to include in subgraph
            
        Returns:
            PyTorch Geometric data object containing the subgraph
        """
        # Create a mapping from original node indices to new indices
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Get edges where both nodes are in the selected set
        edge_index = graph.edge_index
        mask = torch.tensor([(edge_index[0, i].item() in node_mapping and 
                            edge_index[1, i].item() in node_mapping) 
                           for i in range(edge_index.size(1))])
        
        # Create new edge index with remapped node indices
        new_edge_index = edge_index[:, mask]
        new_edge_index = torch.tensor([[node_mapping[new_edge_index[0, i].item()],
                                      node_mapping[new_edge_index[1, i].item()]]
                                     for i in range(new_edge_index.size(1))]).t()
        
        # Handle node features and labels
        x = graph.x[node_indices]
        y = None
        
        if hasattr(graph, 'y'):
            y = graph.y[node_indices]
            # Convert to one-hot encoding if needed
            if len(y.shape) == 1:
                num_classes = y.max().item() + 1
                y = torch.nn.functional.one_hot(y, num_classes).float()
        
        # Create new data object
        subgraph = Data(
            x=x,
            edge_index=new_edge_index,
            y=y
        )
        
        return subgraph
    