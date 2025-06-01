"""
Advanced model implementations for sophisticated model stealing attacks.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  


class DistributionShiftGAT(nn.Module):
    """
    GAT model with distribution shift capabilities using adversarial training.
    """
    
    def __init__(self, 
                 in_feats: int,
                 h_feats: int, 
                 output_dim: int,
                 num_classes: int,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """
        Initialize distribution shift GAT model.
        
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden feature dimension  
            output_dim: Output embedding dimension
            num_classes: Number of classes
            num_heads: Number of attention heads
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super(DistributionShiftGAT, self).__init__()
        
        self.num_layers = num_layers
        self.h_feats = h_feats
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        # GAT layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GATConv(in_feats, h_feats, heads=num_heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(h_feats * num_heads, h_feats, heads=num_heads, dropout=dropout)
            )
        
        # Output layer
        self.layers.append(
            GATConv(h_feats * num_heads, output_dim, heads=1, concat=False, dropout=dropout)
        )
        
        # Additional layers for distribution alignment
        self.distribution_projector = nn.Sequential(
            nn.Linear(output_dim, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_feats, output_dim)
        )
        
    def forward(self, data):
        """
        Forward pass with optional distribution shift.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply GAT layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  # Not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply distribution projector for alignment
        if self.training:
            x = self.distribution_projector(x)
        
        return x


class EmbeddingDiscriminator(nn.Module):
    """
    Discriminator network for adversarial training in distribution shift attacks.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 500):
        """
        Initialize discriminator.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
        """
        super(EmbeddingDiscriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input embeddings
            
        Returns:
            Discriminator output (probability that input is real)
        """
        return self.network(x)


class AdvancedSurrogateTrainer:
    """
    Advanced trainer for surrogate models with adversarial training capabilities.
    """
    
    def __init__(self,
                 target_model: nn.Module,
                 surrogate_architecture: str = 'gat',
                 use_distribution_shift: bool = False,
                 device: str = 'cuda'):
        """
        Initialize advanced trainer.
        
        Args:
            target_model: Target model to steal
            surrogate_architecture: Architecture for surrogate
            use_distribution_shift: Whether to use adversarial distribution alignment
            device: Device to use
        """
        self.target_model = target_model.to(device)
        self.surrogate_architecture = surrogate_architecture
        self.use_distribution_shift = use_distribution_shift
        self.device = device
        
        # Will be initialized during training
        self.surrogate_model = None
        self.classifier = None
        self.discriminator = None
        
    def _create_advanced_surrogate(self, in_feats: int, out_feats: int, target_emb_dim: int):
        """
        Create advanced surrogate model with optional distribution shift.
        
        Args:
            in_feats: Input features
            out_feats: Output classes
            target_emb_dim: Target embedding dimension
            
        Returns:
            Tuple of (surrogate_model, classifier, discriminator)
        """
        if self.surrogate_architecture == 'gat':
            if self.use_distribution_shift:
                surrogate = DistributionShiftGAT(
                    in_feats=in_feats,
                    h_feats=256,
                    output_dim=target_emb_dim,
                    num_classes=out_feats,
                    num_heads=4,
                    num_layers=3,
                    dropout=0.5
                )
            else:
                # Use regular GAT with embedding output
                from .model_stealing import SurrogateEmbeddingModel
                from ..models.gnn import GATModel
                base_model = GATModel(in_feats, 256, out_feats, num_heads=4, num_layers=3)
                surrogate = SurrogateEmbeddingModel(base_model, target_emb_dim)
        else:
            raise NotImplementedError(f"Advanced {self.surrogate_architecture} not implemented yet")
        
        # Create classifier
        classifier = nn.Linear(target_emb_dim, out_feats)
        
        # Create discriminator if using distribution shift
        discriminator = None
        if self.use_distribution_shift:
            discriminator = EmbeddingDiscriminator(target_emb_dim, hidden_dim=500)
        
        return (surrogate.to(self.device), 
                classifier.to(self.device),
                discriminator.to(self.device) if discriminator else None)
    
    def train_with_adversarial_loss(self,
                                   query_data,
                                   val_data,
                                   test_data,
                                   num_epochs: int = 100,
                                   lr: float = 0.001,
                                   adversarial_weight: float = 0.1,
                                   log_every: int = 10):
        """
        Train surrogate with adversarial distribution alignment.
        
        Args:
            query_data: Data to query target with
            val_data: Validation data
            test_data: Test data
            num_epochs: Number of epochs
            lr: Learning rate
            adversarial_weight: Weight for adversarial loss
            log_every: Log frequency
            
        Returns:
            Training results
        """
        print("Starting adversarial training with distribution shift...")
        
        # Query target model
        with torch.no_grad():
            target_embs, target_preds = self.target_model(query_data)
        
        # Create models
        in_feats = query_data.x.shape[1]
        out_feats = target_preds.shape[1]
        target_emb_dim = target_embs.shape[1]
        
        self.surrogate_model, self.classifier, self.discriminator = self._create_advanced_surrogate(
            in_feats, out_feats, target_emb_dim
        )
        
        # Optimizers
        surrogate_optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=lr)
        classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.01)
        
        if self.discriminator:
            discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
            generator_optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.0001)
        
        # Loss functions
        embedding_loss_fn = nn.MSELoss()
        classification_loss_fn = nn.CrossEntropyLoss()
        
        results = {'train_losses': [], 'train_accs': [], 'disc_losses': [], 'gen_losses': []}
        
        for epoch in range(num_epochs):
            self.surrogate_model.train()
            self.classifier.train()
            if self.discriminator:
                self.discriminator.train()
            
            # Forward pass
            surrogate_embs = self.surrogate_model(query_data)
            
            # Embedding loss
            embedding_loss = torch.sqrt(embedding_loss_fn(surrogate_embs, target_embs))
            
            # Update surrogate for embedding matching
            surrogate_optimizer.zero_grad()
            embedding_loss.backward(retain_graph=True)
            surrogate_optimizer.step()
            
            # Classification loss
            with torch.no_grad():
                surrogate_embs_detached = self.surrogate_model(query_data)
            
            logits = self.classifier(surrogate_embs_detached.detach())
            classification_loss = classification_loss_fn(logits, query_data.y.long())
            
            classifier_optimizer.zero_grad()
            classification_loss.backward()
            classifier_optimizer.step()
            
            # Adversarial training for distribution alignment
            disc_loss, gen_loss = 0.0, 0.0
            if self.discriminator and epoch % 5 == 0:  # Every 5 epochs
                # Train discriminator
                for _ in range(5):  # Multiple discriminator updates
                    self.surrogate_model.eval()
                    
                    # Real samples (Gaussian noise)
                    real_samples = torch.randn_like(surrogate_embs).to(self.device)
                    real_labels = self.discriminator(real_samples)
                    
                    # Fake samples (surrogate embeddings)
                    fake_samples = self.surrogate_model(query_data)
                    fake_labels = self.discriminator(fake_samples)
                    
                    # Discriminator loss (maximize log likelihood)
                    disc_loss = -torch.mean(torch.log(real_labels + 1e-8) + 
                                          torch.log(1 - fake_labels + 1e-8))
                    
                    discriminator_optimizer.zero_grad()
                    disc_loss.backward()
                    discriminator_optimizer.step()
                
                # Train generator (surrogate model)
                self.surrogate_model.train()
                fake_samples = self.surrogate_model(query_data)
                fake_labels = self.discriminator(fake_samples)
                
                # Generator loss (fool discriminator)
                gen_loss = -torch.mean(torch.log(fake_labels + 1e-8))
                
                generator_optimizer.zero_grad()
                gen_loss.backward()
                generator_optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                train_acc = (logits.argmax(dim=1) == query_data.y).float().mean().item()
            
            results['train_losses'].append(embedding_loss.item())
            results['train_accs'].append(train_acc)
            results['disc_losses'].append(disc_loss.item() if isinstance(disc_loss, torch.Tensor) else disc_loss)
            results['gen_losses'].append(gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss)
            
            if epoch % log_every == 0:
                print(f'Epoch {epoch:05d} | Emb Loss {embedding_loss.item():.4f} | '
                      f'Train Acc {train_acc:.4f} | Class Loss {classification_loss.item():.4f}')
                if self.discriminator:
                    print(f'              | Disc Loss {disc_loss:.4f} | Gen Loss {gen_loss:.4f}')
        
        return results
    
    def evaluate_surrogate(self, data):
        """Evaluate surrogate model."""
        self.surrogate_model.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            embeddings = self.surrogate_model(data)
            logits = self.classifier(embeddings)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == data.y).float().mean().item()
        
        return accuracy


class ModelPruner:
    """
    Class for pruning models as part of advanced attacks.
    """
    
    def __init__(self, pruning_ratio: float = 0.1):
        """
        Initialize pruner.
        
        Args:
            pruning_ratio: Fraction of parameters to prune
        """
        self.pruning_ratio = pruning_ratio
    
    def random_prune(self, model: nn.Module) -> nn.Module:
        """
        Apply random pruning.
        This should cause significant degradation at high ratios.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model (copy)
        """
        import copy
        
        pruned_model = copy.deepcopy(model)
        
        # Apply pruning to  parameters 
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:  # Only prune trainable parameters
                bitmask = torch.rand_like(param) > self.pruning_ratio
                with torch.no_grad():
                    param.copy_(torch.mul(param, bitmask.float()))
        
        return pruned_model
    
    def verify_pruning_ratio(self, original_model: nn.Module, pruned_model: nn.Module) -> float:
        """
        Verify what percentage of weights were actually set to zero.
        
        Args:
            original_model: Original model before pruning
            pruned_model: Model after pruning
            
        Returns:
            Actual pruning ratio (fraction of weights set to zero)
        """
        total_params = 0
        zero_params = 0
        
        for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), 
            pruned_model.named_parameters()
        ):
            if param1.requires_grad:
                total_params += param2.numel()
                zero_params += (param2 == 0).sum().item()
        
        actual_ratio = zero_params / total_params if total_params > 0 else 0
        return actual_ratio

class FineTuner:
    """
    Fine-tuning capabilities for stolen models.
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize fine-tuner."""
        self.device = device
    
    def fine_tune_on_target_task(self,
                                model: nn.Module,
                                fine_tune_data,
                                num_epochs: int = 50,
                                lr: float = 0.0001,
                                freeze_backbone: bool = True):
        """
        Fine-tune model on a target downstream task.
        
        Args:
            model: Model to fine-tune
            fine_tune_data: Data for fine-tuning
            num_epochs: Number of epochs
            lr: Learning rate
            freeze_backbone: Whether to freeze backbone and only train classifier
            
        Returns:
            Fine-tuned model
        """
        model = model.to(self.device)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            if hasattr(model, 'surrogate_model'):
                # For surrogate models
                embeddings = model.surrogate_model(fine_tune_data)
                outputs = model.classifier(embeddings)
            else:
                # For regular models
                embeddings, outputs = model(fine_tune_data)
            
            loss = criterion(outputs, fine_tune_data.y.long())
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                acc = (outputs.argmax(dim=1) == fine_tune_data.y).float().mean()
                print(f'Fine-tune epoch {epoch}, loss: {loss:.4f}, acc: {acc:.4f}')
        
        return model 