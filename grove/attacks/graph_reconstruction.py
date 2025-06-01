"""
Graph reconstruction methods for model stealing attacks.
Implements IDGL and other graph structure inference techniques for Type II attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
from typing import Optional
from sklearn.neighbors import kneighbors_graph
from scipy import sparse


class GraphLearner(nn.Module):
    """
    Graph learner module for learning adjacency matrices.
    Adapted from IDGL implementation with proper multi-perspective learning.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 topk: Optional[int] = None,
                 epsilon: float = 0.65,
                 num_pers: int = 8,
                 metric_type: str = 'weighted_cosine',
                 device: str = 'cuda'):
        """
        Initialize graph learner.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for graph learning
            topk: Top-k connections to keep (if None, use epsilon)
            epsilon: Threshold for edge connections
            num_pers: Number of perspectives for multi-perspective learning
            metric_type: Type of metric ('weighted_cosine', 'kernel', etc.)
            device: Device to use
        """
        super(GraphLearner, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.epsilon = epsilon
        self.num_pers = num_pers
        self.metric_type = metric_type
        self.device = device
        
        if metric_type == 'weighted_cosine':
            # Multi-perspective weight tensor as in original IDGL
            self.weight_tensor = nn.Parameter(torch.Tensor(num_pers, input_dim))
            print(f'[ Multi-perspective {metric_type} GraphLearner: {num_pers} ]')
        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        elif metric_type == 'cosine':
            pass  # No learnable parameters for simple cosine
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset parameters."""
        if self.metric_type == 'weighted_cosine':
            nn.init.xavier_uniform_(self.weight_tensor)
        elif self.metric_type == 'kernel':
            nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Learn adjacency matrix from node features.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            node_mask: Optional mask for nodes
            
        Returns:
            Learned adjacency matrix
        """
        if self.metric_type == 'weighted_cosine':
            return self._weighted_cosine_similarity(features, node_mask)
        elif self.metric_type == 'kernel':
            return self._kernel_similarity(features, node_mask)
        elif self.metric_type == 'cosine':
            return self._cosine_similarity(features, node_mask)
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")
    
    def _weighted_cosine_similarity(self, features: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-perspective weighted cosine similarity as in original IDGL."""
        # Multi-perspective transformation
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)  # [num_pers, 1, input_dim]
        
        # Apply multi-perspective transformation
        context_fc = features.unsqueeze(0) * expand_weight_tensor  # [num_pers, num_nodes, input_dim]
        
        # Normalize features for each perspective
        context_norm = F.normalize(context_fc, p=2, dim=-1)  # [num_pers, num_nodes, input_dim]
        
        # Compute similarity matrix for each perspective and average
        similarity = torch.matmul(context_norm, context_norm.transpose(-1, -2))  # [num_pers, num_nodes, num_nodes]
        similarity = similarity.mean(0)  # Average over perspectives: [num_nodes, num_nodes]
        
        # Apply threshold or top-k
        if self.topk is not None:
            # Keep only top-k connections
            values, indices = torch.topk(similarity, self.topk, dim=1)
            adj = torch.zeros_like(similarity)
            adj.scatter_(1, indices, values)
        else:
            # Apply threshold
            adj = torch.where(similarity > self.epsilon, similarity, torch.zeros_like(similarity))
        
        # Make symmetric
        adj = (adj + adj.t()) / 2
        
        # Apply mask if provided
        if node_mask is not None:
            mask_matrix = node_mask.unsqueeze(1) * node_mask.unsqueeze(0)
            adj = adj * mask_matrix
        
        return adj
    
    def _kernel_similarity(self, features: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute kernel-based similarity."""
        # Transform features
        feat_transformed = torch.mm(features, self.weight)
        
        # Compute RBF kernel
        dist = torch.cdist(feat_transformed, feat_transformed, p=2)
        similarity = torch.exp(-0.5 * dist * (self.precision_inv_dis ** 2))
        
        # Apply threshold or top-k
        if self.topk is not None:
            values, indices = torch.topk(similarity, self.topk, dim=1)
            adj = torch.zeros_like(similarity)
            adj.scatter_(1, indices, values)
        else:
            adj = torch.where(similarity > self.epsilon, similarity, torch.zeros_like(similarity))
        
        # Make symmetric
        adj = (adj + adj.t()) / 2
        
        # Apply mask if provided
        if node_mask is not None:
            mask_matrix = node_mask.unsqueeze(1) * node_mask.unsqueeze(0)
            adj = adj * mask_matrix
        
        return adj
    
    def _cosine_similarity(self, features: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute simple cosine similarity."""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(features_norm, features_norm.t())
        
        # Apply threshold or top-k
        if self.topk is not None:
            values, indices = torch.topk(similarity, self.topk, dim=1)
            adj = torch.zeros_like(similarity)
            adj.scatter_(1, indices, values)
        else:
            adj = torch.where(similarity > self.epsilon, similarity, torch.zeros_like(similarity))
        
        # Apply mask if provided
        if node_mask is not None:
            mask_matrix = node_mask.unsqueeze(1) * node_mask.unsqueeze(0)
            adj = adj * mask_matrix
        
        return adj


class IDGLReconstructor:
    """
    IDGL-based graph structure reconstruction for Type II attacks.
    Implements proper iterative deep graph learning with convergence checking.
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 hidden_dim: int = 256,
                 epsilon: float = 0.65,
                 num_pers: int = 8,
                 metric_type: str = 'weighted_cosine',
                 max_iter: int = 10,
                 eps_adj: float = 0.0,
                 threshold_type: str = 'adaptive',
                 graph_skip_conn: float = 0.4,
                 update_adj_ratio: Optional[float] = 0.1):
        """
        Initialize IDGL reconstructor.
        
        Args:
            device: Device to use
            hidden_dim: Hidden dimension for graph learning
            epsilon: Threshold for connections
            num_pers: Number of perspectives for multi-perspective learning
            metric_type: Similarity metric type
            max_iter: Maximum iterations for iterative learning
            eps_adj: Convergence threshold for adjacency matrix
            threshold_type: How to determine final adjacency ('adaptive', 'fixed')
            graph_skip_conn: Skip connection weight for adjacency updates
            update_adj_ratio: Ratio for adjacency matrix updates
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.num_pers = num_pers
        self.metric_type = metric_type
        self.max_iter = max_iter
        self.eps_adj = eps_adj
        self.threshold_type = threshold_type
        self.graph_skip_conn = graph_skip_conn
        self.update_adj_ratio = update_adj_ratio
        
    def _diff(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Compute difference metric for convergence checking."""
        return torch.norm(X - Y, p='fro') / torch.norm(Z, p='fro')
    
    def _add_graph_regularization(self, adj: torch.Tensor, features: torch.Tensor, 
                                 smoothness_ratio: float = 0.4, 
                                 degree_ratio: float = 0.1) -> torch.Tensor:
        """Add graph regularization terms."""
        # Smoothness regularization
        smoothness_loss = torch.trace(torch.mm(torch.mm(features.t(), adj), features))
        
        # Degree regularization (encourage sparse connections)
        degree_loss = torch.sum(adj)
        
        return smoothness_ratio * smoothness_loss + degree_ratio * degree_loss

    def reconstruct_graph(self, data: Data, dataset_name: str = 'unknown') -> Data:
        """
        Reconstruct graph structure using iterative deep graph learning.
        
        Args:
            data: Original data with features
            dataset_name: Name of dataset (affects thresholding)
            
        Returns:
            Data with reconstructed graph structure
        """
        features = data.x.to(self.device)
        num_nodes = features.shape[0]
        
        # Initialize graph learners for iterative learning
        graph_learner1 = GraphLearner(
            input_dim=features.shape[1],
            hidden_dim=self.hidden_dim,
            epsilon=self.epsilon,
            num_pers=self.num_pers,
            metric_type=self.metric_type,
            device=self.device
        ).to(self.device)
        
        graph_learner2 = GraphLearner(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            epsilon=self.epsilon,
            num_pers=self.num_pers,
            metric_type=self.metric_type,
            device=self.device
        ).to(self.device)
        
        # Simple GNN encoder for iterative learning
        encoder = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ).to(self.device)
        
        with torch.no_grad():
            # Initial adjacency matrix learning
            raw_adj = graph_learner1(features)
            
            # Normalize adjacency matrix
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=1e-8)
            
            # Add self-loops
            adj = adj + torch.eye(num_nodes, device=self.device)
            first_adj = adj.clone()
            
            # Initial node embeddings
            node_vec = torch.relu(torch.mm(adj, torch.mm(features, encoder[0].weight.t()) + encoder[0].bias))
            node_vec = F.dropout(node_vec, 0.5, training=False)
            init_node_vec = node_vec.clone()
            
            # Iterative graph learning
            for iter_ in range(self.max_iter):
                pre_raw_adj = raw_adj.clone()
                
                # Learn new adjacency matrix from current node embeddings
                raw_adj_new = graph_learner2(node_vec)
                
                # Normalize new adjacency matrix
                adj_new = raw_adj_new / torch.clamp(torch.sum(raw_adj_new, dim=-1, keepdim=True), min=1e-8)
                
                # Apply update ratio if specified
                if self.update_adj_ratio is not None:
                    adj = self.update_adj_ratio * adj_new + (1 - self.update_adj_ratio) * first_adj
                    raw_adj = self.update_adj_ratio * raw_adj_new + (1 - self.update_adj_ratio) * raw_adj
                else:
                    adj = adj_new
                    raw_adj = raw_adj_new
                
                # Update node embeddings - use correct layer index (3 for second Linear layer)
                node_vec = torch.relu(torch.mm(adj, torch.mm(init_node_vec, encoder[3].weight.t()) + encoder[3].bias))
                node_vec = F.dropout(node_vec, 0.5, training=False)
                
                # Check convergence
                if iter_ > 0:
                    diff = self._diff(raw_adj, pre_raw_adj, raw_adj)
                    if diff.item() <= self.eps_adj:
                        print(f"IDGL converged at iteration {iter_} with diff {diff.item():.6f}")
                        break
        
        # Apply dataset-specific thresholding 
        learned_adj_np = raw_adj.detach().cpu().numpy()
        
        if self.threshold_type == 'adaptive':
            if dataset_name in ['acm', 'amazon']:
                binary_adj = (learned_adj_np > 0.9).astype(np.int32)
            elif dataset_name in ['coauthor']:
                binary_adj = (learned_adj_np >= 0.999).astype(np.int32)
            else:
                binary_adj = (learned_adj_np > 0.999).astype(np.int32)
        else:
            # Fixed threshold
            binary_adj = (learned_adj_np > self.epsilon).astype(np.int32)
        
        # Convert to edge_index format
        edge_index, _ = dense_to_sparse(torch.from_numpy(binary_adj.astype(np.float32)))
        
        # Create new data object with reconstructed structure
        reconstructed_data = Data(
            x=data.x,
            edge_index=edge_index.to(data.x.device),
            y=data.y
        )
        
        return reconstructed_data


class KNNReconstructor:
    """
    K-nearest neighbors graph reconstruction.
    Simple baseline for graph structure inference.
    """
    
    def __init__(self, k: int = 10, metric: str = 'cosine'):
        """
        Initialize KNN reconstructor.
        
        Args:
            k: Number of nearest neighbors
            metric: Distance metric ('cosine', 'euclidean', etc.)
        """
        self.k = k
        self.metric = metric
    
    def reconstruct_graph(self, data: Data) -> Data:
        """
        Reconstruct graph using KNN.
        
        Args:
            data: Original data with features
            
        Returns:
            Data with KNN-based graph structure
        """
        features = data.x.cpu().numpy()
        
        # Build KNN graph
        knn_graph = kneighbors_graph(
            features, 
            n_neighbors=self.k, 
            metric=self.metric,
            mode='connectivity',
            include_self=False
        )
        
        # Make symmetric
        knn_graph = (knn_graph + knn_graph.T) / 2
        knn_graph.data = np.ones_like(knn_graph.data)
        
        # Convert to edge_index
        coo = knn_graph.tocoo()
        edge_index = torch.stack([
            torch.from_numpy(coo.row),
            torch.from_numpy(coo.col)
        ], dim=0).long()
        
        # Create new data object
        reconstructed_data = Data(
            x=data.x,
            edge_index=edge_index.to(data.x.device),
            y=data.y
        )
        
        return reconstructed_data


class RandomGraphReconstructor:
    """
    Random graph reconstruction for baseline comparison.
    """
    
    def __init__(self, edge_prob: float = 0.1, preserve_degree: bool = True):
        """
        Initialize random graph reconstructor.
        
        Args:
            edge_prob: Probability of edge existence
            preserve_degree: Whether to preserve original degree distribution
        """
        self.edge_prob = edge_prob
        self.preserve_degree = preserve_degree
    
    def reconstruct_graph(self, data: Data) -> Data:
        """
        Reconstruct graph with random structure.
        
        Args:
            data: Original data
            
        Returns:
            Data with random graph structure
        """
        num_nodes = data.x.shape[0]
        
        if self.preserve_degree:
            # Preserve original number of edges
            original_num_edges = data.edge_index.shape[1]
            
            # Sample random edges
            all_possible_edges = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    all_possible_edges.append([i, j])
            
            # Randomly select edges
            selected_indices = np.random.choice(
                len(all_possible_edges), 
                size=min(original_num_edges // 2, len(all_possible_edges)), 
                replace=False
            )
            
            selected_edges = [all_possible_edges[i] for i in selected_indices]
            
            # Create edge_index (undirected)
            edge_list = []
            for edge in selected_edges:
                edge_list.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            # Use edge probability
            adj_matrix = torch.rand(num_nodes, num_nodes) < self.edge_prob
            adj_matrix = adj_matrix.triu(diagonal=1)  # Upper triangular
            adj_matrix = adj_matrix + adj_matrix.t()  # Make symmetric
            
            edge_index, _ = dense_to_sparse(adj_matrix.float())
        
        # Create new data object
        reconstructed_data = Data(
            x=data.x,
            edge_index=edge_index.to(data.x.device),
            y=data.y
        )
        
        return reconstructed_data


def estimate_graph_structure(data: Data, 
                           method: str = 'original',
                           dataset_name: str = 'unknown',
                           **kwargs) -> Data:
    """
    Estimate graph structure using specified method.
    
    Args:
        data: Original data
        method: Reconstruction method ('original', 'idgl', 'knn', 'random')
        dataset_name: Name of dataset
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Data with estimated graph structure
    """
    if method == 'original':
        return data
    
    elif method == 'idgl':
        reconstructor = IDGLReconstructor(**kwargs)
        return reconstructor.reconstruct_graph(data, dataset_name)
    
    elif method == 'knn':
        reconstructor = KNNReconstructor(**kwargs)
        return reconstructor.reconstruct_graph(data)
    
    elif method == 'random':
        reconstructor = RandomGraphReconstructor(**kwargs)
        return reconstructor.reconstruct_graph(data)
    
    else:
        raise ValueError(f"Unknown reconstruction method: {method}") 