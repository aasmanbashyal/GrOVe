import argparse
from copy import deepcopy
from grove.models.gnn import GATModel, GINModel, GraphSAGEModel
import torch 
import os 
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.data import Data


def class_indices(data, name):
    """Read class IDs from either integer labels or the saved one-hot labels."""
    labels = data.y
    if labels is None or labels.ndim not in (1, 2):
        raise ValueError(f"{name} labels must be class IDs or one-hot rows")
    if labels.shape[0] != data.num_nodes or labels.numel() == 0:
        raise ValueError(f"{name} must have one label per node")
    if labels.ndim == 2 and labels.shape[1] > 0:
        if not torch.all((labels == 0) | (labels == 1)) or not torch.all(labels.sum(dim=1) == 1):
            raise ValueError(f"{name} labels must be one-hot rows")
        labels = labels.argmax(dim=1)
    if not torch.isfinite(labels).all() or (labels < 0).any() or not torch.equal(labels, labels.long()):
        raise ValueError(f"{name} labels must be nonnegative integer class IDs")
    return labels.long()


def split_training_graph(data, validation_fraction=0.2, seed=42):
    """Hold out nodes from the role's training graph, keeping induced edges only.

    The split seed is independent of model initialization. Ownership-verification
    and test graphs are never used to select a checkpoint.
    """
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if data.num_nodes < 2:
        raise ValueError("training graph needs at least two nodes for a validation split")
    count = min(data.num_nodes - 1, max(1, int(data.num_nodes * validation_fraction)))
    indices = torch.randperm(data.num_nodes, generator=torch.Generator().manual_seed(seed))
    return data.subgraph(indices[count:]), data.subgraph(indices[:count])


def _class_count(data, name):
    labels = class_indices(data, name)
    return data.y.shape[1] if data.y.ndim == 2 else int(labels.max()) + 1

def set_random_seed(seed, use_cuda=True):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        use_cuda: Whether to set CUDA seed as well
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

def shuffle_data(data, seed):
    """
    Shuffle the data according to the given seed.
    
    Args:
        data: PyTorch Geometric data object
        seed: Random seed for shuffling
    """
    # Set random seed for shuffling
    np.random.seed(seed)
    
    # Get number of nodes
    num_nodes = data.x.size(0)
    
    # Create permutation indices
    perm = np.random.permutation(num_nodes)
    
    # Shuffle node features
    data.x = data.x[perm]
    
    # Shuffle labels if they exist
    if hasattr(data, 'y'):
        data.y = data.y[perm]
    
    # Update edge indices according to the new node ordering
    if hasattr(data, 'edge_index'):
        # Create mapping from old indices to new indices
        idx_map = {j: i for i, j in enumerate(perm)}
        
        # Update edge indices
        new_edge_index = torch.zeros_like(data.edge_index)
        for i in range(data.edge_index.size(1)):
            new_edge_index[0, i] = idx_map[data.edge_index[0, i].item()]
            new_edge_index[1, i] = idx_map[data.edge_index[1, i].item()]
        data.edge_index = new_edge_index
    
    return data


def train_model(model_name, dataset_name, output_dir, embeddings_dir=None, device='cuda', epochs=200, lr=0.001, 
                batch_size=32, hidden_dim=256, dropout=0.5, early_stopping=10, 
                seed=42, model_role='target', independent_model='gat', split_type='non-overlapped',
                validation_fraction=0.2, validation_seed=42):
    
    
    # Set random seed for reproducibility
    set_random_seed(seed, use_cuda=True if device == 'cuda' else False)
    
    if epochs < 1 or early_stopping < 1:
        raise ValueError("epochs and early_stopping must be positive")
    
    # Load data
    data_dir = f"data/processed/{split_type}/{dataset_name}"
    print(f"Loading data from: {data_dir}")
    
    target_train = torch.load(os.path.join(data_dir, 'target_train.pt'), map_location='cpu', weights_only=False)
    
    # Load independent_train if it exists, otherwise use target_train
    independent_train_path = os.path.join(data_dir, 'independent_train.pt')
    if os.path.exists(independent_train_path):
        independent_train = torch.load(independent_train_path, map_location='cpu', weights_only=False)
        print(f"Loaded independent_train data: {independent_train.x.shape[0]} nodes")
    else:
        independent_train = target_train
        print(f"Using target_train data for independent model (same data, different seed)")
    
    # Load query_train if it exists, otherwise use surrogate_train
    query_train_path = os.path.join(data_dir, 'query_train.pt')
    if os.path.exists(query_train_path):
        query_train = torch.load(query_train_path, map_location='cpu', weights_only=False)
        print(f"Loaded query_train data: {query_train.x.shape[0]} nodes")
    else:
        query_train = torch.load(os.path.join(data_dir, 'surrogate_train.pt'), map_location='cpu', weights_only=False)
        print(f"Using surrogate_train data for query (fallback)")
    
    # These files are aliases in preprocessing and are reserved for ownership
    # verification, not early stopping.
    validation_path = os.path.join(data_dir, 'validation.pt')
    if os.path.exists(validation_path):
        verification = torch.load(validation_path, map_location='cpu', weights_only=False)
        print(f"Loaded ownership verification data: {verification.x.shape[0]} nodes")
    else:
        verification = torch.load(os.path.join(data_dir, 'verification.pt'), map_location='cpu', weights_only=False)
        print("Using verification.pt for ownership embeddings")

    # Debug prints for data shapes
    print(f"\nData shapes:")
    print(f"target_train.x shape: {target_train.x.shape}")
    print(f"target_train.y shape: {target_train.y.shape if hasattr(target_train, 'y') else 'No y attribute'}")
    print(f"target_train.edge_index shape: {target_train.edge_index.shape}")
    print(f"Model role: {model_role} with seed: {seed}")
    
    role_data = {'target': target_train, 'independent': independent_train, 'surrogate': query_train}
    if model_role not in role_data:
        raise ValueError(f"Invalid model role: {model_role}")
    selected_data = role_data[model_role]
    
    # Initialize model
    input_dim = target_train.x.shape[1]
    output_dim = max(_class_count(target_train, 'target training'),
                     _class_count(selected_data, f'{model_role} training'))
    print(f"\nModel configuration:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Output dimension: {output_dim}")
    
    if model_name == 'gat':
        # Three-layer GAT model with 4 attention heads in first and second layers
        model = GATModel(
            in_feats=input_dim, 
            h_feats=hidden_dim, 
            out_feats=output_dim, 
            num_heads=4, 
            num_layers=3,  
            dropout=dropout
        )
        print("Initialized 3-layer GAT model with 4 attention heads in first and second layers")
    elif model_name == 'gin':
        # Three-layer GIN model with neighborhood sample size of 10
        model = GINModel(
            in_feats=input_dim, 
            h_feats=hidden_dim, 
            out_feats=output_dim, 
            num_layers=3,  
            dropout=dropout
        )
        print("Initialized 3-layer GIN model with neighborhood sample size of 10")
    elif model_name == 'sage':
        # Two-layer GraphSAGE model with neighborhood sample sizes of 25 and 10
        model = GraphSAGEModel(
            in_feats=input_dim, 
            h_feats=hidden_dim, 
            out_feats=output_dim, 
            num_layers=2,  
            aggregator_type='mean',  
            dropout=dropout
        )
        print("Initialized 2-layer GraphSAGE model with neighborhood sample sizes of 25 and 10")
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    # Apply initialization depending on model role
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            if model_role == 'independent':
                # Use Kaiming initialization for independent model
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            else:
                # Use Xavier/Glorot initialization for target and surrogate
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)
    
    # Apply the initialization to model weights
    model.apply(init_weights)
    
    model = model.to(device)
    
    selected_data = selected_data.clone()
    selected_data.y = class_indices(selected_data, f'{model_role} training')
    train_data, validation = split_training_graph(selected_data, validation_fraction, validation_seed)
    print(f"Training/early-stopping split: {train_data.num_nodes}/{validation.num_nodes} nodes")
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader([validation], batch_size=batch_size, shuffle=False)
    verification_loader = DataLoader([verification], batch_size=batch_size, shuffle=False)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Create progress bar
    pbar = tqdm(range(epochs), desc='Training Progress')
    
    for epoch in pbar:
        # Training
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            embeddings, out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(out, 1)
            labels = batch.y
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                embeddings, out = model(batch)
                loss = criterion(out, batch.y)
                total_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(out, 1)
                labels = batch.y
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
            
        avg_val_loss = total_val_loss / len(validation_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'train_acc': f'{train_accuracy:.4f}',
            'val_acc': f'{val_accuracy:.4f}',
            'patience': f'{patience_counter}/{early_stopping}'
        })
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
                print(f'Best validation accuracy: {best_val_acc:.4f}')
                break
    
    if best_model_state is None:
        raise RuntimeError("training did not produce a finite validation loss")
    model.load_state_dict(best_model_state)

    # Test data is read only after checkpoint selection and restoration.
    test = torch.load(os.path.join(data_dir, 'test.pt'), map_location='cpu', weights_only=False)
    test.y = class_indices(test, 'test')
    if int(test.y.max()) >= output_dim:
        raise ValueError("test labels contain a class absent from the training class schema")
    test_loader = DataLoader([test], batch_size=batch_size, shuffle=False)

    # Report metrics and embeddings from the same selected checkpoint.
    model.eval()
    final_verification_embeddings = []
    final_train_correct = 0
    final_train_total = 0
    final_val_correct = 0
    final_val_total = 0
    final_test_correct = 0
    final_test_total = 0
    
    with torch.no_grad():
        # Calculate final accuracies for reporting
        for batch in train_loader:
            batch = batch.to(device)
            _, out = model(batch)
            
            # Calculate final accuracy
            _, predicted = torch.max(out, 1)
            labels = batch.y
            final_train_correct += (predicted == labels).sum().item()
            final_train_total += labels.size(0)
        
        for batch in validation_loader:
            batch = batch.to(device)
            _, out = model(batch)
            
            # Calculate final accuracy
            _, predicted = torch.max(out, 1)
            labels = batch.y
            final_val_correct += (predicted == labels).sum().item()
            final_val_total += labels.size(0)

        for batch in test_loader:
            batch = batch.to(device)
            _, out = model(batch)
            predicted = out.argmax(dim=1)
            final_test_correct += (predicted == batch.y).sum().item()
            final_test_total += batch.y.size(0)
        
        # Generate final verification embeddings
        for batch in verification_loader:
            batch = batch.to(device)
            embeddings, _ = model(batch)
            final_verification_embeddings.append(embeddings.detach())
    
    final_train_accuracy = final_train_correct / final_train_total if final_train_total > 0 else 0
    final_val_accuracy = final_val_correct / final_val_total if final_val_total > 0 else 0
    final_test_accuracy = final_test_correct / final_test_total if final_test_total > 0 else 0
    
    final_verification_embeddings = torch.cat(final_verification_embeddings, dim=0).cpu()
    best_verification_embeddings = final_verification_embeddings.clone()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics to CSV file in model dir
    import pandas as pd

    from datetime import datetime

    # Prepare metrics dictionary
    metrics = {
        'final_train_acc': final_train_accuracy,
        'final_val_acc': final_val_accuracy,
        'final_test_acc': final_test_accuracy,
        'validation_fraction': validation_fraction,
        'validation_seed': validation_seed,
        'epochs': epochs,
        'hidden_dim': hidden_dim,
        'model_role': model_role,
        'model_name': model_name,
        'dataset': dataset_name,
        'split_type': split_type,
        'timestamp': datetime.now().isoformat()
    }
    if model_role == 'independent':
        metrics_csv_path = Path(output_dir) / f"{model_name}_{dataset_name}_{model_role}_{split_type}_{independent_model}_metrics.csv"
    else:
        metrics_csv_path = Path(output_dir) / f"{model_name}_{dataset_name}_{model_role}_{split_type}_metrics.csv"

    # If file exists, append; else, create with header
    try:
        df_existing = pd.read_csv(metrics_csv_path)
        df_new = pd.DataFrame([metrics])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(metrics_csv_path, index=False)
    except Exception:
        # File does not exist or is unreadable, create new
        df_new = pd.DataFrame([metrics])
        df_new.to_csv(metrics_csv_path, index=False)

    # Print training summary
    print("\nTraining Summary:")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Last epoch training loss: {train_losses[-1]:.4f}")
    print(f"Last epoch validation loss: {val_losses[-1]:.4f}")
    print(f"Final training accuracy: {final_train_accuracy:.4f}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    
    # Save the best model and verification embeddings
    if best_model_state is not None:
        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if embeddings_dir is not None:
            embeddings_path = Path(embeddings_dir)
            embeddings_path.mkdir(parents=True, exist_ok=True)
        else:
            embeddings_path = output_path
        
        # Save model
        if model_role == 'independent':
            model_save_path = output_path / f"{model_name}_{dataset_name}_{model_role}_{independent_model}_{split_type}.pt"
        else:   
            model_save_path = output_path / f"{model_name}_{dataset_name}_{model_role}_{split_type}.pt"
        torch.save({
            'model_state_dict': best_model_state,
            'model_config': {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'dropout': dropout
            },
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'validation_fraction': validation_fraction,
                'validation_seed': validation_seed,
                'train_nodes': train_data.num_nodes,
                'validation_nodes': validation.num_nodes,
                'final_test_acc': final_test_accuracy,
            }
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Save verification embeddings only
        if model_role == 'independent':
            embeddings_save_path = embeddings_path / f"{model_name}_{dataset_name}_{model_role}_{independent_model}.pt"
        else:       
            embeddings_save_path = embeddings_path / f"{model_name}_{dataset_name}_{model_role}.pt"
        
        # Create Data objects for verification embeddings
        best_verification_data = Data(
            x=best_verification_embeddings,
            y=None,  
            embedding_type='best_verification'
        )
        
        final_verification_data = Data(
            x=final_verification_embeddings,
            y=None,  
            embedding_type='final_verification'
        )
        
        # Save in PyG Data format - only verification embeddings
        torch.save({
            'best_embeddings': {
                'verification': best_verification_data
            },
            'final_embeddings': {
                'verification': final_verification_data
            },
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'final_train_acc': final_train_accuracy,
            'final_val_acc': final_val_accuracy,
            'final_test_acc': final_test_accuracy,
        }, embeddings_save_path)
        print(f"Verification embeddings saved to {embeddings_save_path}")
        print(f"Verification embeddings shape: {best_verification_data.x.shape}")

def main():
    parser = argparse.ArgumentParser(description='Train a Grove GNN model')
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'gin', 'sage'], help='Model to train')
    parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train on')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for model checkpoints')
    parser.add_argument('--embeddings-dir', type=str, default=None, help='Output directory for embeddings (defaults to output-dir if not specified)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to train on')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument("--early-stopping", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--model-role", type=str, default="target", choices=["target", "independent", "surrogate"], help="Model role (default: target)")
    parser.add_argument('--independent-model', type=str, default='gat', choices=['gat', 'gin', 'sage'], help='Model For Independent Training')
    parser.add_argument("--split-type", type=str, default="non-overlapped", choices=["non-overlapped", "overlapped"], help="Split type (default: non-overlapped)")
    parser.add_argument('--validation-fraction', type=float, default=0.2, help='Fraction held out from the role training graph for early stopping')
    parser.add_argument('--validation-seed', type=int, default=42, help='Training-validation split seed, independent of the model seed')
    args = parser.parse_args()
    train_model(args.model, args.dataset, args.output_dir, args.embeddings_dir, args.device, args.epochs, args.lr, args.batch_size, args.hidden_dim, args.dropout, args.early_stopping, args.seed, args.model_role, args.independent_model, args.split_type, args.validation_fraction, args.validation_seed)

if __name__ == '__main__':
    main()
