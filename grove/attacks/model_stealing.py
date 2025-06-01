#!/usr/bin/env python3
"""
Model stealing attacks for Graph Neural Networks.
Supports Type I (original structure) and Type II (reconstructed structure) attacks.

Type I Attack: Query and train on original structure
Type II Attack: Query and train on reconstructed structure (IDGL, KNN, etc.)

Based on "GrOVe: Ownership Verification of Graph Neural Networks using Embeddings"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np

from ..models.gnn import GATModel, GINModel, GraphSAGEModel
from .graph_reconstruction import estimate_graph_structure


class SurrogateEmbeddingModel(nn.Module):
    """
    Surrogate model that outputs embeddings to match target model embeddings.
    """
    
    def __init__(self, base_model: Union[GATModel, GINModel, GraphSAGEModel], output_dim: int):
        """
        Initialize surrogate embedding model.
        
        Args:
            base_model: Base GNN model (GAT, GIN, or GraphSAGE)
            output_dim: Dimension of target embeddings to match
        """
        super(SurrogateEmbeddingModel, self).__init__()
        self.base_model = base_model
        self.output_dim = output_dim
        
        # Override the classifier to output embeddings of target dimension
        self.base_model.classifier = nn.Linear(base_model.h_feats, output_dim)
        
    def forward(self, data):
        """
        Forward pass returning embeddings.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            embeddings: Node embeddings matching target dimension
        """
        embeddings, _ = self.base_model(data)
        # Use the modified classifier to get target-dimensional embeddings
        target_embeddings = self.base_model.classifier(embeddings)
        return target_embeddings

class SurrogateModelTrainer:
    """
    Trainer class for surrogate models using model stealing attacks.
    """
    
    def __init__(self, 
                 target_model: nn.Module,
                 surrogate_architecture: str = 'gat',
                 recovery_from: str = 'embedding',
                 structure: str = 'original',
                 device: str = 'cuda',
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: Optional[int] = None,
                 dropout: float = 0.5,
                 dataset_name: str = 'unknown',
                 idgl_params: Optional[Dict[str, Any]] = None):
        """
        Initialize surrogate model trainer.
        
        Args:
            target_model: The target model to steal
            surrogate_architecture: Architecture for surrogate ('gat', 'gin', 'sage')
            recovery_from: What to recover from target ('embedding', 'prediction', 'projection')
            structure: Graph structure to use ('original', 'idgl', 'knn', 'random')
            device: Device to run on
            hidden_dim: Hidden dimension for surrogate model
            num_heads: Number of attention heads for GAT
            num_layers: Number of layers (if None, uses architecture defaults)
            dropout: Dropout rate
            dataset_name: Name of dataset (affects IDGL thresholding)
            idgl_params: Additional parameters for IDGL reconstruction
        """
        self.target_model = target_model.to(device)
        self.target_model.eval()
        self.surrogate_architecture = surrogate_architecture.lower()
        self.recovery_from = recovery_from.lower()
        self.structure = structure.lower()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.dataset_name = dataset_name
        self.idgl_params = idgl_params or {}
        
        # Will be set during training
        self.surrogate_model = None
        self.classifier = None
        self.detached_classifier = None
        self._projection_layer = None
        
    def _apply_structure_modification(self, data: Data) -> Data:
        """
        Apply structure modification based on attack type.
        
        Args:
            data: Original data
            
        Returns:
            Data with modified structure (Type I: original, Type II: reconstructed)
        """
        if self.structure == 'original':
            # Type I attack: use original structure
            return data.to(self.device)
        else:
            # Type II attack: reconstruct structure
            print(f"🔄 Reconstructing graph structure using {self.structure} method...")
            
            # Prepare parameters for reconstruction
            reconstruction_kwargs = {}
            if self.structure == 'idgl':
                # Pass IDGL-specific parameters
                reconstruction_kwargs.update({
                    'device': self.device,
                    'hidden_dim': self.idgl_params.get('hidden_dim', 256),
                    'epsilon': self.idgl_params.get('epsilon', 0.65),
                    'num_pers': self.idgl_params.get('num_pers', 8),
                    'metric_type': self.idgl_params.get('metric_type', 'weighted_cosine'),
                    'max_iter': self.idgl_params.get('max_iter', 10),
                    'eps_adj': self.idgl_params.get('eps_adj', 0.0),
                    'threshold_type': self.idgl_params.get('threshold_type', 'adaptive')
                })
            elif self.structure == 'knn':
                reconstruction_kwargs.update({
                    'k': self.idgl_params.get('k', 10),
                    'metric': self.idgl_params.get('metric', 'cosine')
                })
            elif self.structure == 'random':
                reconstruction_kwargs.update({
                    'edge_prob': self.idgl_params.get('edge_prob', 0.1),
                    'preserve_degree': self.idgl_params.get('preserve_degree', True)
                })
            
            try:
                reconstructed_data = estimate_graph_structure(
                    data, 
                    method=self.structure,
                    dataset_name=self.dataset_name,
                    **reconstruction_kwargs
                )
                
                print(f"Original edges: {data.edge_index.shape[1]}, "
                      f"Reconstructed edges: {reconstructed_data.edge_index.shape[1]}")
                
                return reconstructed_data.to(self.device)
                
            except Exception as e:
                print(f"⚠️  Graph reconstruction failed: {e}")
                print("Falling back to original structure...")
                return data.to(self.device)
    
    def _query_target_model(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the target model to get responses.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Tuple of (predictions, embeddings)
        """
        data = data.to(self.device)
        self.target_model.eval()
        with torch.no_grad():
            try:
                embeddings, predictions = self.target_model(data)
                return predictions.to(self.device), embeddings.to(self.device)
            except Exception as e:
                print(f"⚠️  Target model query failed: {e}")
                # Return dummy outputs with correct shapes
                num_nodes = data.x.shape[0]
                dummy_embeddings = torch.zeros(num_nodes, self.hidden_dim, device=self.device)
                dummy_predictions = torch.zeros(num_nodes, 10, device=self.device)  # Assume 10 classes
                return dummy_predictions, dummy_embeddings
    
    def _create_surrogate_model(self, in_feats: int, out_feats: int, target_output_dim: int) -> Tuple[nn.Module, nn.Module]:
        """
        Create surrogate model and classifier.
        
        Args:
            in_feats: Input feature dimension
            out_feats: Output class dimension
            target_output_dim: Dimension of target model output to match
            
        Returns:
            Tuple of (surrogate_model, classifier)
        """
        # Create base surrogate model
        if self.surrogate_architecture == 'gat':
            base_model = GATModel(
                in_feats=in_feats,
                h_feats=self.hidden_dim,
                out_feats=out_feats,
                num_heads=self.num_heads,
                num_layers=self.num_layers or 3,
                dropout=self.dropout
            )
        elif self.surrogate_architecture == 'gin':
            base_model = GINModel(
                in_feats=in_feats,
                h_feats=self.hidden_dim,
                out_feats=out_feats,
                num_layers=self.num_layers or 3,
                dropout=self.dropout
            )
        elif self.surrogate_architecture == 'sage':
            base_model = GraphSAGEModel(
                in_feats=in_feats,
                h_feats=self.hidden_dim,
                out_feats=out_feats,
                num_layers=self.num_layers or 2,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown surrogate architecture: {self.surrogate_architecture}")
        
        # Create surrogate embedding model
        surrogate_model = SurrogateEmbeddingModel(base_model, target_output_dim)
        
        # Create classifier for final predictions
        classifier = nn.Linear(target_output_dim, out_feats)
        
        return surrogate_model.to(self.device), classifier.to(self.device)
    
    def _apply_projection(self, embeddings: torch.Tensor, labels: torch.Tensor, transform_name: str = 'tsne') -> torch.Tensor:
        """
        Apply dimensionality reduction projection (simplified version).
        
        Args:
            embeddings: Input embeddings
            labels: Node labels
            transform_name: Type of projection ('tsne', 'pca', etc.)
            
        Returns:
            Projected embeddings
        """
        # Initialize projection layer if not exists
        if self._projection_layer is None:
            projection_dim = min(50, embeddings.shape[1])
            self._projection_layer = nn.Linear(embeddings.shape[1], projection_dim).to(self.device)
            # Initialize with Xavier uniform
            nn.init.xavier_uniform_(self._projection_layer.weight)
            nn.init.zeros_(self._projection_layer.bias)
        
        with torch.no_grad():
            projected = self._projection_layer(embeddings)
        
        return projected
    
    def _train_detached_classifier(self, embeddings: torch.Tensor, labels: torch.Tensor) -> MLPClassifier:
        """
        Train a detached classifier on embeddings.
        
        Args:
            embeddings: Node embeddings
            labels: Node labels
            
        Returns:
            Trained MLPClassifier
        """
        X = embeddings.detach().cpu().numpy()
        y = labels.cpu().numpy()
        
        # Handle edge case where we have very few samples
        if len(X) < 10:
            print("⚠️  Too few samples for detached classifier training")
            # Create a dummy classifier
            clf = MLPClassifier(random_state=42, max_iter=10)
            # Fit on dummy data
            dummy_X = np.random.randn(10, X.shape[1])
            dummy_y = np.random.randint(0, max(2, len(np.unique(y))), 10)
            clf.fit(dummy_X, dummy_y)
            return clf
        
        # Split for training with stratification if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            # Fallback without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train MLP classifier
        batch_size = min(1024, max(32, len(X_train) // 4))
        clf = MLPClassifier(
            random_state=42, 
            max_iter=300, 
            batch_size=batch_size,
            early_stopping=True,
            validation_fraction=0.1 if len(X_train) > 100 else 0.0
        )
        
        try:
            clf.fit(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            print(f"Detached classifier test accuracy: {test_acc:.4f}")
        except Exception as e:
            print(f"⚠️  Detached classifier training failed: {e}")
        
        return clf
    
    def train_surrogate(self, 
                       query_data: Data,
                       val_data: Data,
                       test_data: Data,
                       num_epochs: int = 200,
                       lr: float = 0.001,
                       batch_size: int = 1024,
                       log_every: int = 10,
                       eval_every: int = 20) -> Dict[str, Any]:
        """
        Train surrogate model using model stealing attack.
        
        Algorithm (matching oldgrove):
        - Type I: Query target with ORIGINAL, train surrogate with ORIGINAL
        - Type II: Query target with RECONSTRUCTED, train surrogate with RECONSTRUCTED
        
        Args:
            query_data: Data to query target model with
            val_data: Validation data
            test_data: Test data
            num_epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            log_every: Log every N epochs
            eval_every: Evaluate every N epochs
            
        Returns:
            Dictionary with training results
        """
        print(f"Starting model stealing attack with {self.surrogate_architecture} surrogate")
        print(f"Recovery from: {self.recovery_from}")
        print(f"Attack type: {'Type I (original structure)' if self.structure == 'original' else f'Type II ({self.structure} structure)'}")
        
        # Apply structure modification for both querying and training
        modified_query_data = self._apply_structure_modification(query_data)
        modified_val_data = self._apply_structure_modification(val_data)
        modified_test_data = self._apply_structure_modification(test_data)
        
        # Query target model with the structure that will be used for training
        # Type I: Query with original structure, Type II: Query with reconstructed structure
        print(f"Querying target model with {'original' if self.structure == 'original' else 'reconstructed'} structure...")
        query_preds, query_embs = self._query_target_model(modified_query_data)
        
        # Determine what to recover from target
        if self.recovery_from == 'prediction':
            target_response = query_preds
            print(f"Using predictions as target response (dim: {target_response.shape[1]})")
        elif self.recovery_from == 'embedding':
            target_response = query_embs
            print(f"Using embeddings as target response (dim: {target_response.shape[1]})")
        elif self.recovery_from == 'projection':
            target_response = self._apply_projection(query_embs, modified_query_data.y)
            print(f"Using projections as target response (dim: {target_response.shape[1]})")
        else:
            raise ValueError(f"Unknown recovery_from: {self.recovery_from}")
        
        # Create surrogate model
        in_feats = modified_query_data.x.shape[1]
        out_feats = query_preds.shape[1]
        target_output_dim = target_response.shape[1]
        
        self.surrogate_model, self.classifier = self._create_surrogate_model(
            in_feats, out_feats, target_output_dim
        )
        
        # Setup optimizers
        surrogate_optimizer = optim.Adam(self.surrogate_model.parameters(), lr=lr)
        classifier_optimizer = optim.SGD(self.classifier.parameters(), lr=0.01)
        
        # Loss functions
        embedding_loss_fn = nn.MSELoss()
        classification_loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        print("Starting surrogate training...")
        results = {
            'train_losses': [],
            'train_accs': [],
            'val_accs': [],
            'test_accs': []
        }
        
        for epoch in range(num_epochs):
            self.surrogate_model.train()
            self.classifier.train()
            
            try:
                # Forward pass 
                surrogate_embs = self.surrogate_model(modified_query_data)
                
                # Embedding loss (MSE with target response)
                embedding_loss = torch.sqrt(embedding_loss_fn(surrogate_embs, target_response))
                
                # Update surrogate model
                surrogate_optimizer.zero_grad()
                embedding_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.surrogate_model.parameters(), max_norm=1.0)
                surrogate_optimizer.step()
                
                # Classification loss
                with torch.no_grad():
                    surrogate_embs_detached = self.surrogate_model(modified_query_data)
                
                logits = self.classifier(surrogate_embs_detached.detach())
                classification_loss = classification_loss_fn(logits, modified_query_data.y.long())
                
                # Update classifier
                classifier_optimizer.zero_grad()
                classification_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                classifier_optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    train_acc = (logits.argmax(dim=1) == modified_query_data.y).float().mean().item()
                
                results['train_losses'].append(embedding_loss.item())
                results['train_accs'].append(train_acc)
                
                # Logging
                if epoch % log_every == 0:
                    print(f'Epoch {epoch:05d} | Loss {embedding_loss.item():.4f} | '
                          f'Train Acc {train_acc:.4f} | Class Loss {classification_loss.item():.4f}')
                
                # Evaluation
                if epoch % eval_every == 0 and epoch > 0:
                    val_acc = self._evaluate_surrogate(modified_val_data)
                    test_acc = self._evaluate_surrogate(modified_test_data)
                    results['val_accs'].append(val_acc)
                    results['test_accs'].append(test_acc)
                    print(f'Val Acc {val_acc:.4f} | Test Acc {test_acc:.4f}')
                    
            except Exception as e:
                print(f"⚠️  Training error at epoch {epoch}: {e}")
                # Continue training with dummy values
                results['train_losses'].append(float('inf'))
                results['train_accs'].append(0.0)
        
        # Train detached classifier for final evaluation
        print("Training detached classifier...")
        try:
            with torch.no_grad():
                final_embs = self.surrogate_model(modified_query_data)
            self.detached_classifier = self._train_detached_classifier(final_embs, modified_query_data.y)
        except Exception as e:
            print(f"⚠️  Detached classifier training failed: {e}")
        
        # Final evaluation
        final_val_acc = self._evaluate_surrogate(modified_val_data)
        final_test_acc = self._evaluate_surrogate(modified_test_data)
        
        print(f"Final Val Acc: {final_val_acc:.4f}")
        print(f"Final Test Acc: {final_test_acc:.4f}")
        
        results['final_val_acc'] = final_val_acc
        results['final_test_acc'] = final_test_acc
        
        return results
    
    def evaluate_surrogate(self, data: Data) -> float:
        """
        Public method to evaluate surrogate model on data.
        
        Args:
            data: Data to evaluate on
            
        Returns:
            Accuracy
        """
        return self._evaluate_surrogate(data)
    
    def _evaluate_surrogate(self, data: Data) -> float:
        """
        Evaluate surrogate model on data.
        
        Args:
            data: Data to evaluate on
            
        Returns:
            Accuracy
        """
        if self.surrogate_model is None or self.classifier is None:
            return 0.0
            
        self.surrogate_model.eval()
        self.classifier.eval()
        
        try:
            with torch.no_grad():
                data = data.to(self.device)
                embeddings = self.surrogate_model(data)
                logits = self.classifier(embeddings)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == data.y).float().mean().item()
            return accuracy
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
            return 0.0
    
    def _evaluate_surrogate_on_original(self, data: Data) -> float:
        """
        Evaluate surrogate model on original structure (without modification).
        Used for final evaluation to compare with target model fairly.
        
        Args:
            data: Data to evaluate on (original structure)
            
        Returns:
            Accuracy
        """
        if self.surrogate_model is None or self.classifier is None:
            return 0.0
            
        self.surrogate_model.eval()
        self.classifier.eval()
        
        try:
            with torch.no_grad():
                data = data.to(self.device)
                embeddings = self.surrogate_model(data)
                logits = self.classifier(embeddings)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == data.y).float().mean().item()
            return accuracy
        except Exception as e:
            print(f"⚠️  Evaluation on original structure failed: {e}")
            return 0.0
    
    def compute_fidelity(self, data: Data) -> float:
        """
        Compute fidelity between target and surrogate model predictions.
        Fidelity measures how well surrogate mimics target on original structure.
        
        Args:
            data: Data to compute fidelity on (original structure)
            
        Returns:
            Fidelity score (agreement between target and surrogate predictions)
        """
        if self.surrogate_model is None or self.classifier is None:
            return 0.0
            
        try:
            data = data.to(self.device)
            
            # Get target predictions (on original structure)
            target_preds, _ = self._query_target_model(data)
            target_preds = F.softmax(target_preds, dim=1)
            
            # Get surrogate predictions (on original structure)
            self.surrogate_model.eval()
            self.classifier.eval()
            with torch.no_grad():
                surrogate_embs = self.surrogate_model(data)
                surrogate_logits = self.classifier(surrogate_embs)
                surrogate_preds = F.softmax(surrogate_logits, dim=1)
            
            # Compute fidelity (agreement between predictions)
            fidelity = (target_preds.argmax(dim=1) == surrogate_preds.argmax(dim=1)).float().mean().item()
            
            return fidelity
        except Exception as e:
            print(f"⚠️  Fidelity computation failed: {e}")
            return 0.0
    
    def save_models(self, save_path: str):
        """
        Save trained surrogate models.
        
        Args:
            save_path: Directory to save models
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save surrogate model
            if self.surrogate_model is not None:
                torch.save(self.surrogate_model.state_dict(), 
                          os.path.join(save_path, 'surrogate_model.pth'))
            
            # Save classifier
            if self.classifier is not None:
                torch.save(self.classifier.state_dict(), 
                          os.path.join(save_path, 'classifier.pth'))
            
            # Save detached classifier
            if self.detached_classifier is not None:
                import joblib
                joblib.dump(self.detached_classifier, 
                           os.path.join(save_path, 'detached_classifier.pkl'))
            
            print(f"Models saved to {save_path}")
        except Exception as e:
            print(f"⚠️  Failed to save models: {e}")


class ModelStealingAttack:
    """
    Main class for conducting model stealing attacks.
    """
    
    def __init__(self, target_model: nn.Module, device: str = 'cuda'):
        """
        Initialize model stealing attack.
        
        Args:
            target_model: Target model to attack
            device: Device to run on
        """
        self.target_model = target_model
        self.device = device
        
    def simple_extraction(self, 
                         query_data: Data,
                         val_data: Data,
                         test_data: Data,
                         surrogate_architecture: str = 'gat',
                         recovery_from: str = 'embedding',
                         structure: str = 'original',
                         dataset_name: str = 'unknown',
                         num_epochs: int = 100,
                         lr: float = 0.001,
                         **kwargs) -> Tuple[SurrogateModelTrainer, Dict[str, Any]]:
        """
        Perform simple model extraction attack.
        
        Args:
            query_data: Data to query target with
            val_data: Validation data
            test_data: Test data  
            surrogate_architecture: Surrogate architecture ('gat', 'gin', 'sage')
            recovery_from: What to recover ('embedding', 'prediction', 'projection')
            structure: Graph structure to use ('original', 'idgl', 'knn', 'random')
            dataset_name: Name of dataset (affects IDGL thresholding)
            num_epochs: Number of training epochs
            lr: Learning rate
            **kwargs: Additional arguments for trainer
            
        Returns:
            Tuple of (trained_trainer, results)
        """
        trainer = SurrogateModelTrainer(
            target_model=self.target_model,
            surrogate_architecture=surrogate_architecture,
            recovery_from=recovery_from,
            structure=structure,
            dataset_name=dataset_name,
            device=self.device,
            **kwargs
        )
        
        results = trainer.train_surrogate(query_data, val_data, test_data, 
                                        num_epochs=num_epochs, lr=lr)
        
        return trainer, results
    
    def double_extraction(self,
                         query_data: Data,
                         val_data: Data, 
                         test_data: Data,
                         surrogate_architecture: str = 'gat',
                         recovery_from: str = 'embedding',
                         structure: str = 'original',
                         dataset_name: str = 'unknown',
                         num_epochs: int = 100,
                         lr: float = 0.001,
                         **kwargs) -> Tuple[SurrogateModelTrainer, Dict[str, Any]]:
        """
        Perform double extraction attack (train surrogate, then train another surrogate on first).
        
        Args:
            query_data: Data to query target with
            val_data: Validation data
            test_data: Test data
            surrogate_architecture: Surrogate architecture
            recovery_from: What to recover
            structure: Graph structure to use ('original', 'idgl', 'knn', 'random')
            dataset_name: Name of dataset (affects IDGL thresholding)
            num_epochs: Number of training epochs
            lr: Learning rate
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (final_trainer, results)
        """
        # Split query data in half with proper edge handling
        num_nodes = query_data.x.shape[0]
        split_idx = num_nodes // 2
        
        # Create node indices for each subset
        nodes_1 = torch.arange(split_idx)
        nodes_2 = torch.arange(split_idx, num_nodes)
        
        # Extract subgraphs properly
        edge_index_1, _ = subgraph(nodes_1, query_data.edge_index, relabel_nodes=True)
        edge_index_2, _ = subgraph(nodes_2, query_data.edge_index, relabel_nodes=True)
        
        # Create first subset
        query_data_1 = Data(
            x=query_data.x[:split_idx],
            edge_index=edge_index_1,
            y=query_data.y[:split_idx]
        ).to(self.device)
        
        # Create second subset  
        query_data_2 = Data(
            x=query_data.x[split_idx:],
            edge_index=edge_index_2,
            y=query_data.y[split_idx:]
        ).to(self.device)
        
        # First extraction
        print("=== First Extraction ===")
        trainer_1 = SurrogateModelTrainer(
            target_model=self.target_model,
            surrogate_architecture=surrogate_architecture,
            recovery_from=recovery_from,
            structure=structure,
            dataset_name=dataset_name,
            device=self.device,
            **kwargs
        )
        
        results_1 = trainer_1.train_surrogate(query_data_1, val_data, test_data, 
                                              num_epochs=num_epochs, lr=lr)
        
        # Second extraction using first surrogate as target
        print("=== Second Extraction ===")
        
        # Create wrapper for first surrogate to match target model interface
        class SurrogateWrapper(nn.Module):
            def __init__(self, surrogate_model, classifier):
                super().__init__()
                self.surrogate_model = surrogate_model
                self.classifier = classifier
                
            def forward(self, data):
                embeddings = self.surrogate_model(data)
                predictions = self.classifier(embeddings)
                return embeddings, predictions
        
        surrogate_as_target = SurrogateWrapper(trainer_1.surrogate_model, trainer_1.classifier)
        
        trainer_2 = SurrogateModelTrainer(
            target_model=surrogate_as_target,
            surrogate_architecture=surrogate_architecture,
            recovery_from=recovery_from,
            structure=structure,
            dataset_name=dataset_name,
            device=self.device,
            **kwargs
        )
        
        results_2 = trainer_2.train_surrogate(query_data_2, val_data, test_data, 
                                             num_epochs=num_epochs, lr=lr)
        
        # Combine results
        combined_results = {
            'first_extraction': results_1,
            'second_extraction': results_2,
            'final_val_acc': results_2['final_val_acc'],
            'final_test_acc': results_2['final_test_acc']
        }
        
        return trainer_2, combined_results
    
    def evaluate_attack_success(self, 
                               trainer: SurrogateModelTrainer,
                               test_data: Data) -> Dict[str, float]:
        """
        Evaluate the success of the model stealing attack.
        Following oldgrove: both models evaluated on original test structure.
        
        Args:
            trainer: Trained surrogate model trainer
            test_data: Test data (original structure)
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get target model performance on original structure
            trainer.target_model.eval()
            with torch.no_grad():
                target_embs, target_preds = trainer.target_model(test_data.to(trainer.device))
                target_acc = (target_preds.argmax(dim=1) == test_data.y.to(trainer.device)).float().mean().item()
            
            # Get surrogate model performance on original structure (for fair comparison)
            surrogate_acc = trainer._evaluate_surrogate_on_original(test_data)
            
            # Compute fidelity on original structure
            fidelity = trainer.compute_fidelity(test_data)
            
            results = {
                'target_accuracy': target_acc,
                'surrogate_accuracy': surrogate_acc,
                'fidelity': fidelity,
                'accuracy_gap': abs(target_acc - surrogate_acc)
            }
            
            print("=== Attack Evaluation ===")
            print(f"Target Accuracy: {target_acc:.4f}")
            print(f"Surrogate Accuracy: {surrogate_acc:.4f}")
            print(f"Fidelity: {fidelity:.4f}")
            print(f"Accuracy Gap: {results['accuracy_gap']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"⚠️  Attack evaluation failed: {e}")
            return {
                'target_accuracy': 0.0,
                'surrogate_accuracy': 0.0,
                'fidelity': 0.0,
                'accuracy_gap': 0.0
            } 