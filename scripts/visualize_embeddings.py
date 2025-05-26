import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch_geometric.data import Data


sns.set_theme(style="whitegrid")    

# Register safe globals for PyTorch Geometric
torch.serialization.add_safe_globals([
    'torch_geometric.data.data.Data',
    'torch_geometric.data.data.DataEdgeAttr',
    'torch_geometric.data.data.DataNodeAttr',
    'torch_geometric.data.storage.EdgeStorage',
    'torch_geometric.data.storage.NodeStorage'
])

MODEL_COLORS = {
    'target': '#1f77b4',      # deep blue
    'independent': '#2ca02c', # vivid green
    'surrogate': '#d62728'    # strong red
}

# Add marker styles for different models
MODEL_MARKERS = {
    'target': 'o',           # circle
    'independent': 's',      # square
    'surrogate': '^'         # triangle up
}

# Add jitter function
def add_jitter(data, scale=0.1):
    """Add small random jitter to data points."""
    return data + np.random.normal(0, scale, data.shape)

def load_embeddings(embeddings_path):
    """Load embeddings from file."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    # Set weights_only=False to properly load custom classes
    embeddings_data = torch.load(embeddings_path, weights_only=False)
    return embeddings_data



def plot_embeddings(embeddings_paths, output_dir, combined=True, perplexity=30):
    """Plot embeddings using t-SNE visualization with customizable perplexity."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each embeddings file
    model_data = {}
    for path in embeddings_paths:
        # Extract model type from filename
        file_name = os.path.basename(path)
        parts = file_name.split('_')
        
        # Handle different path formats
        if len(parts) >= 3:
            model_name = parts[0]
            dataset_name = parts[1]
            model_role = parts[2].split('.')[0]  # Remove any file extension
            if '_' in model_role:
                model_role = model_role.split('_')[0]  
        else:
            raise ValueError(f"Invalid file name format: {file_name}")
        
        # Load embeddings
        embeddings_data = load_embeddings(path)
        
        # Extract embeddings from PyG Data objects
        best_embeddings = embeddings_data['best_embeddings']
        
        # Handle PyTorch Geometric Data objects
        if isinstance(best_embeddings['train'], Data):
            train_embeddings = best_embeddings['train'].x.numpy()
            val_embeddings = best_embeddings['val'].x.numpy()
        else:
            # Fallback for old format
            train_embeddings = best_embeddings['train'].numpy()
            val_embeddings = best_embeddings['val'].numpy()
        
        # Combine train and val embeddings
        combined_embeddings = np.vstack([train_embeddings, val_embeddings])
        
        # Store original embeddings
        model_data[model_role] = {
            'original': combined_embeddings,
            'name': f"{model_name} {model_role.capitalize()}"
        }
    
    # Perform t-SNE dimensionality reduction for visualization
    model_types = list(model_data.keys())
    embeddings = [model_data[model_type]['original'] for model_type in model_types]
    
    # Find minimum size across all embeddings to ensure equal sizes
    min_size = min(emb.shape[0] for emb in embeddings)
    embeddings = [emb[:min_size] for emb in embeddings]
    
    # Combine all embeddings for t-SNE
    combined_embeddings = np.vstack(embeddings)
    
    # Create labels to distinguish different models
    labels = np.concatenate([np.full(emb.shape[0], i) for i, emb in enumerate(embeddings)])
    
    # Perform t-SNE with configurable perplexity
    print(f"Performing t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=2000)
    tsne_results = tsne.fit_transform(combined_embeddings)
    
    # Split t-SNE results by model
    tsne_by_model = []
    start_idx = 0
    for emb in embeddings:
        end_idx = start_idx + emb.shape[0]
        tsne_by_model.append(tsne_results[start_idx:end_idx])
        start_idx = end_idx
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 10))
    
    for i, model_type in enumerate(model_types):
        # Add small jitter to help distinguish overlapping points
        jittered_x = add_jitter(tsne_by_model[i][:, 0], scale=0.5)
        jittered_y = add_jitter(tsne_by_model[i][:, 1], scale=0.5)
        
        # Get marker and color for this model type
        marker = MODEL_MARKERS.get(model_type, 'o')
        color = MODEL_COLORS.get(model_type, f'C{i}')
        
        print(f"Plotting {model_type} with marker {marker} and color {color}")  # Debug print
        
        plt.scatter(
            jittered_x, 
            jittered_y,
            alpha=0.6, 
            s=40,       
            c=color,
            marker=marker,
            label=model_data[model_type]['name'],
            edgecolors='white',  
            linewidth=0.5        
        )
   
    # Add title and labels with enhanced styling
    plt.title(f'Embedding Visualization ({dataset_name} {model_name} {perplexity})', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    # Add legend with better positioning and styling
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9, 
              markerscale=1.5,  #
              edgecolor='gray')  
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high resolution
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_tsne_per_{perplexity}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    


def try_multiple_perplexity_values(embeddings_paths, output_dir, perplexity_values=None):
    """
    Try multiple perplexity values for t-SNE to find the best visualization.
    
    Args:
        embeddings_paths: Paths to embeddings files
        output_dir: Directory to save visualizations
        perplexity_values: List of perplexity values to try (default: [5, 10, 30, 50, 100])
    """
    if perplexity_values is None:
        perplexity_values = [5, 10, 30, 50, 100]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try each perplexity value
    for perplexity in perplexity_values:
        print(f"\nTrying perplexity value: {perplexity}")
        plot_embeddings(embeddings_paths, output_dir, combined=True, perplexity=perplexity)

def main():
    parser = argparse.ArgumentParser(description='Visualize model embeddings')
    parser.add_argument('--embeddings-path', type=str, nargs='+', required=True,
                      help='Path(s) to embedding file(s)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--combined', action='store_true',
                      help='Combine embeddings from multiple models')
    parser.add_argument('--perplexity', type=int, default=30,
                      help='Perplexity value for t-SNE (default: 30)')
    parser.add_argument('--try-multiple-perplexity', action='store_true',
                      help='Try multiple perplexity values for t-SNE')
    parser.add_argument('--perplexity-values', type=int, nargs='+',
                      default=[5, 10, 30, 50, 100],
                      help='List of perplexity values to try (default: 5 10 30 50 100)')
    args = parser.parse_args()
    
    if args.try_multiple_perplexity:
        try_multiple_perplexity_values(args.embeddings_path, args.output_dir, args.perplexity_values)
    else:
        plot_embeddings(args.embeddings_path, args.output_dir, args.combined, args.perplexity)

if __name__ == '__main__':
    main() 