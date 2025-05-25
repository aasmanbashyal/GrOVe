import argparse
from grove.models.gnn import GATModel, GINModel, GraphSAGEModel
import torch 
import os 
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def train_model(model_name, dataset_name, output_dir, embeddings_dir=None, device='cuda', epochs=100, lr=0.001, 
                batch_size=32, hidden_dim=128, dropout=0.5, early_stopping=10, 
                seed=42, model_role='target', split_type='non-overlapped'):
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data with safe globals for PyTorch Geometric
    torch.serialization.add_safe_globals([
        'torch_geometric.data.data.Data',
        'torch_geometric.data.data.DataEdgeAttr',
        'torch_geometric.data.data.DataNodeAttr'
    ])
    
    # Load data
    data_dir = f"data/processed/{split_type}/{dataset_name}"
    print(f"Loading data from: {data_dir}")
    
    target_train = torch.load(os.path.join(data_dir, 'target_train.pt'), weights_only=False)
    surrogate_train = torch.load(os.path.join(data_dir, 'surrogate_train.pt'), weights_only=False)
    independent_train = torch.load(os.path.join(data_dir, 'target_train.pt'), weights_only=False)
    test = torch.load(os.path.join(data_dir, 'test.pt'), weights_only=False)
    verification = torch.load(os.path.join(data_dir, 'verification.pt'), weights_only=False)
    
    # Debug prints for data shapes
    print(f"\nData shapes:")
    print(f"target_train.x shape: {target_train.x.shape}")
    print(f"target_train.y shape: {target_train.y.shape if hasattr(target_train, 'y') else 'No y attribute'}")
    print(f"target_train.edge_index shape: {target_train.edge_index.shape}")
    
    # Ensure y is properly shaped
    if not hasattr(target_train, 'y'):
        raise ValueError("Target training data missing 'y' attribute")
    if len(target_train.y.shape) == 1:
        # If y is 1D, convert to 2D one-hot encoding
        num_classes = target_train.y.max().item() + 1
        target_train.y = torch.nn.functional.one_hot(target_train.y, num_classes).float()
        print(f"Converted y to one-hot encoding with shape: {target_train.y.shape}")
    
    # Initialize model
    input_dim = target_train.x.shape[1]
    output_dim = target_train.y.shape[1]
    print(f"\nModel configuration:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Output dimension: {output_dim}")
    
    if model_name == 'gat':
        model = GATModel(input_dim, hidden_dim, output_dim, dropout=dropout)
    elif model_name == 'gin':
        model = GINModel(input_dim, hidden_dim, output_dim, dropout=dropout)
    elif model_name == 'sage':
        model = GraphSAGEModel(input_dim, hidden_dim, output_dim, dropout=dropout)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    model = model.to(device)
    
    # Setup data based on model role
    if model_role == 'target':
        train_data = target_train
    elif model_role == 'independent':
        train_data = independent_train
    elif model_role == 'surrogate':
        train_data = surrogate_train
    else:
        raise ValueError(f"Invalid model role: {model_role}")
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([verification], batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    train_losses = []
    val_losses = []
    best_embeddings = None
    
    # Create progress bar
    pbar = tqdm(range(epochs), desc='Training Progress')
    
    for epoch in pbar:
        # Training
        model.train()
        total_train_loss = 0
        all_train_embeddings = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            embeddings, out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            all_train_embeddings.append(embeddings.detach())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_val_embeddings = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings, out = model(batch)
                loss = criterion(out, batch.y)
                total_val_loss += loss.item()
                all_val_embeddings.append(embeddings.detach())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Concatenate embeddings
        train_embeddings = torch.cat(all_train_embeddings, dim=0)
        val_embeddings = torch.cat(all_val_embeddings, dim=0)
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'patience': f'{patience_counter}/{early_stopping}'
        })
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            # Store best embeddings
            best_embeddings = {
                'train': train_embeddings.cpu(),
                'val': val_embeddings.cpu()
            }
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
                break
    
    # Get final embeddings
    model.eval()
    final_train_embeddings = []
    final_val_embeddings = []
    
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            embeddings, _ = model(batch)
            final_train_embeddings.append(embeddings.detach())
        
        for batch in val_loader:
            batch = batch.to(device)
            embeddings, _ = model(batch)
            final_val_embeddings.append(embeddings.detach())
    
    final_embeddings = {
        'train': torch.cat(final_train_embeddings, dim=0).cpu(),
        'val': torch.cat(final_val_embeddings, dim=0).cpu()
    }
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    # Save the best model and embeddings
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
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss
            }
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Save embeddings
        embeddings_save_path = embeddings_path / f"{model_name}_{dataset_name}_{model_role}_{split_type}_embeddings.pt"
        torch.save({
            'best_embeddings': best_embeddings,
            'final_embeddings': final_embeddings,
            'best_epoch': best_epoch
        }, embeddings_save_path)
        print(f"Embeddings saved to {embeddings_save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train a Grove GNN model')
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'gin', 'sage'], help='Model to train')
    parser.add_argument('--dataset', type=str, default='citeseer', choices=['citeseer', 'cora', 'pubmed'], help='Dataset to train on')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for model checkpoints')
    parser.add_argument('--embeddings-dir', type=str, default=None, help='Output directory for embeddings (defaults to output-dir if not specified)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to train on')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument("--early-stopping", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--model-role", type=str, default="target", choices=["target", "independent", "surrogate"], help="Model role (default: target)")
    parser.add_argument("--split-type", type=str, default="non-overlapped", choices=["non-overlapped", "overlapped"], help="Split type (default: non-overlapped)")
    args = parser.parse_args()
    train_model(args.model, args.dataset, args.output_dir, args.embeddings_dir, args.device, args.epochs, args.lr, args.batch_size, args.hidden_dim, args.dropout, args.early_stopping, args.seed, args.model_role, args.split_type)

if __name__ == '__main__':
    main()
