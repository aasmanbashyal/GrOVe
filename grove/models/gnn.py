"""
Graph Neural Network models for GrOVe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GINConv
import os


class BaseGNNModel(nn.Module):
    """
    Base class for GNN models.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, dropout=0.5, name=None):
        """
        Initialize the base GNN model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            dropout: Dropout probability
            name: Name of the model
        """
        super(BaseGNNModel, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.dropout = dropout
        self.name = name or self.__class__.__name__
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        raise NotImplementedError("Forward method must be implemented by subclasses")
    
    def save(self, path):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'in_feats': self.in_feats,
            'h_feats': self.h_feats,
            'out_feats': self.out_feats,
            'dropout': self.dropout,
            'name': self.name
        }, path)
        
        
    @classmethod
    def load(cls, path, device=None):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            in_feats=checkpoint['in_feats'],
            h_feats=checkpoint['h_feats'],
            out_feats=checkpoint['out_feats'],
            dropout=checkpoint['dropout'],
            name=checkpoint.get('name')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model

class GATModel(BaseGNNModel):
    """
    Graph Attention Network model.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, num_heads=4, dropout=0.5, name=None):
        """
        Initialize the GAT model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            name: Name of the model
        """
        super(GATModel, self).__init__(in_feats, h_feats, out_feats, dropout, name)
        self.num_heads = num_heads
        
        # Multi-head attention layers
        self.gat1 = GATConv(
            in_feats, h_feats, heads=num_heads, dropout=dropout
        )
        
        self.gat2 = GATConv(
            h_feats * num_heads, h_feats, heads=num_heads, dropout=dropout
        )
        
        # Output layer (combine multiple heads)
        self.gat3 = GATConv(
            h_feats * num_heads, out_feats, heads=1, concat=False, dropout=dropout
        )
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Sample neighbors for each layer
        edge_index1 = self._sample_neighbors(edge_index, 10)
        edge_index2 = self._sample_neighbors(edge_index, 10)
        edge_index3 = self._sample_neighbors(edge_index, 10)
        
        # First GAT layer with multi-head attention
        h = self.gat1(x, edge_index1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second GAT layer
        h = self.gat2(h, edge_index2)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Third GAT layer
        h = self.gat3(h, edge_index3)
        
        # Store embeddings
        embeddings = h
        
        # Apply softmax for classification
        predictions = F.log_softmax(h, dim=1)
        
        return embeddings, predictions
    
    def _sample_neighbors(self, edge_index, num_neighbors):
        """
        Sample a fixed number of neighbors for each node.
        
        Args:
            edge_index: Edge index tensor
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Sampled edge index tensor
        """
        if not self.training:
            return edge_index
            
        # Get unique nodes (both sources and targets)
        nodes = torch.unique(edge_index)
        
        # Pre-allocate tensors for efficiency
        max_edges = len(nodes) * num_neighbors
        new_edges = torch.zeros((2, max_edges), dtype=edge_index.dtype, device=edge_index.device)
        edge_count = 0
        
        for node in nodes:
            # Get both incoming and outgoing edges
            mask = (edge_index[0] == node) | (edge_index[1] == node)
            neighbors = torch.cat([
                edge_index[1][edge_index[0] == node],  # outgoing neighbors
                edge_index[0][edge_index[1] == node]   # incoming neighbors
            ])
            
            # Sample if we have more neighbors than needed
            if len(neighbors) > num_neighbors:
                indices = torch.randperm(len(neighbors))[:num_neighbors]
                neighbors = neighbors[indices]
            elif len(neighbors) < num_neighbors:
                # Sample with replacement if we don't have enough neighbors
                indices = torch.randint(0, len(neighbors), (num_neighbors,))
                neighbors = neighbors[indices]
            
            # Add edges to new edge index
            for neighbor in neighbors:
                new_edges[0, edge_count] = node
                new_edges[1, edge_count] = neighbor
                edge_count += 1
        
        # Trim the tensor to actual size
        return new_edges[:, :edge_count]

class GINModel(BaseGNNModel):
    """
    Graph Isomorphism Network model.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, dropout=0.5, name=None):
        """
        Initialize the GIN model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            dropout: Dropout probability
            name: Name of the model
        """
        super(GINModel, self).__init__(in_feats, h_feats, out_feats, dropout, name)
        
        # MLP for each GIN layer
        self.mlp1 = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        
        self.mlp3 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, out_feats)
        )
        
        # Graph convolution layers
        self.gin1 = GINConv(self.mlp1)
        self.gin2 = GINConv(self.mlp2)
        self.gin3 = GINConv(self.mlp3)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Sample neighbors for each layer
        edge_index1 = self._sample_neighbors(edge_index, 10)
        edge_index2 = self._sample_neighbors(edge_index, 10)
        edge_index3 = self._sample_neighbors(edge_index, 10)
        
        # First GIN layer
        h = self.gin1(x, edge_index1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second GIN layer
        h = self.gin2(h, edge_index2)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Third GIN layer
        h = self.gin3(h, edge_index3)
        
        # Store embeddings (before softmax)
        embeddings = h
        
        # Apply softmax for classification
        predictions = F.log_softmax(h, dim=1)
        
        return embeddings, predictions
    
    def _sample_neighbors(self, edge_index, num_neighbors):
        """
        Sample a fixed number of neighbors for each node.
        
        Args:
            edge_index: Edge index tensor
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Sampled edge index tensor
        """
        if not self.training:
            return edge_index
            
        # Get unique nodes (both sources and targets)
        nodes = torch.unique(edge_index)
        
        # Pre-allocate tensors for efficiency
        max_edges = len(nodes) * num_neighbors
        new_edges = torch.zeros((2, max_edges), dtype=edge_index.dtype, device=edge_index.device)
        edge_count = 0
        
        for node in nodes:
            # Get both incoming and outgoing edges
            mask = (edge_index[0] == node) | (edge_index[1] == node)
            neighbors = torch.cat([
                edge_index[1][edge_index[0] == node],  # outgoing neighbors
                edge_index[0][edge_index[1] == node]   # incoming neighbors
            ])
            
            # Sample if we have more neighbors than needed
            if len(neighbors) > num_neighbors:
                indices = torch.randperm(len(neighbors))[:num_neighbors]
                neighbors = neighbors[indices]
            elif len(neighbors) < num_neighbors:
                # Sample with replacement if we don't have enough neighbors
                indices = torch.randint(0, len(neighbors), (num_neighbors,))
                neighbors = neighbors[indices]
            
            # Add edges to new edge index
            for neighbor in neighbors:
                new_edges[0, edge_count] = node
                new_edges[1, edge_count] = neighbor
                edge_count += 1
        
        # Trim the tensor to actual size
        return new_edges[:, :edge_count]

class GraphSAGEModel(BaseGNNModel):
    """
    GraphSAGE model.
    """
    
    def __init__(self, in_feats, h_feats, out_feats, dropout=0.5, aggregator_type='mean', name=None):
        """
        Initialize the GraphSAGE model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension
            out_feats: Output feature dimension
            dropout: Dropout probability
            aggregator_type: Aggregator type ('mean', 'gcn', 'pool', 'lstm')
            name: Name of the model
        """
        super(GraphSAGEModel, self).__init__(in_feats, h_feats, out_feats, dropout, name)
        self.aggregator_type = aggregator_type
        
        # GraphSAGE layers
        self.sage1 = SAGEConv(in_feats, h_feats, normalize=True, aggr=aggregator_type)
        self.sage2 = SAGEConv(h_feats, out_feats, normalize=True, aggr=aggregator_type)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Sample neighbors for each layer
        edge_index1 = self._sample_neighbors(edge_index, 25)  # First layer: 25 neighbors
        edge_index2 = self._sample_neighbors(edge_index, 10)  # Second layer: 10 neighbors
        
        # First GraphSAGE layer
        h = self.sage1(x, edge_index1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second GraphSAGE layer
        h = self.sage2(h, edge_index2)
        
        # Store embeddings (before softmax)
        embeddings = h
        
        # Apply softmax for classification
        predictions = F.log_softmax(h, dim=1)
        
        return embeddings, predictions
    
    def _sample_neighbors(self, edge_index, num_neighbors):
        """
        Sample a fixed number of neighbors for each node.
        
        Args:
            edge_index: Edge index tensor
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Sampled edge index tensor
        """
        if not self.training:
            return edge_index
            
        # Get unique nodes (both sources and targets)
        nodes = torch.unique(edge_index)
        
        # Pre-allocate tensors for efficiency
        max_edges = len(nodes) * num_neighbors
        new_edges = torch.zeros((2, max_edges), dtype=edge_index.dtype, device=edge_index.device)
        edge_count = 0
        
        for node in nodes:
            # Get both incoming and outgoing edges
            mask = (edge_index[0] == node) | (edge_index[1] == node)
            neighbors = torch.cat([
                edge_index[1][edge_index[0] == node],  # outgoing neighbors
                edge_index[0][edge_index[1] == node]   # incoming neighbors
            ])
            
            # Sample if we have more neighbors than needed
            if len(neighbors) > num_neighbors:
                indices = torch.randperm(len(neighbors))[:num_neighbors]
                neighbors = neighbors[indices]
            elif len(neighbors) < num_neighbors:
                # Sample with replacement if we don't have enough neighbors
                indices = torch.randint(0, len(neighbors), (num_neighbors,))
                neighbors = neighbors[indices]
            
            # Add edges to new edge index
            for neighbor in neighbors:
                new_edges[0, edge_count] = node
                new_edges[1, edge_count] = neighbor
                edge_count += 1
        
        # Trim the tensor to actual size
        return new_edges[:, :edge_count]


