import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import seaborn as sns
sns.set_theme(style="whitegrid")

def load_embeddings(embeddings_path):
    """Load embeddings from saved file."""
    data = torch.load(embeddings_path)
    return data['best_embeddings']

def plot_embeddings(embeddings, labels, title, save_path):
    """Plot embeddings with color-coded labels."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.title(title)
    plt.colorbar(scatter)
    plt.savefig(save_path)
    plt.close()

def visualize_embeddings(embeddings_path, output_dir, perplexity=30, viz_type='both'):
    """Visualize embeddings using PCA and/or t-SNE.
    
    Args:
        embeddings_path: Path to embeddings file
        output_dir: Directory to save visualizations
        perplexity: Perplexity parameter for t-SNE
        viz_type: Type of visualization ('pca', 'tsne', or 'both')
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    best_embeddings = load_embeddings(embeddings_path)
    
    # Get labels from embeddings file name
    file_name = Path(embeddings_path).stem
    model_name, dataset, role, split_type, _ = file_name.split('_')
    
    # Process embeddings
    for emb_type, embeddings in [('best', best_embeddings)]:
        # Convert embeddings to numpy arrays
        train_emb = embeddings['train'].numpy()
        val_emb = embeddings['val'].numpy()
        
        # Get labels (assuming they're the same for both train and val)
        train_labels = np.argmax(train_emb, axis=1) if train_emb.shape[1] > 1 else train_emb
        val_labels = np.argmax(val_emb, axis=1) if val_emb.shape[1] > 1 else val_emb
        
        # Combine train and val embeddings
        combined_emb = np.vstack([train_emb, val_emb])
        combined_labels = np.concatenate([train_labels, val_labels])
        
        # Generate visualizations based on viz_type
        if viz_type in ['pca', 'both']:
            pca = PCA(n_components=2)
            pca_emb = pca.fit_transform(combined_emb)
            plot_embeddings(
                pca_emb, 
                combined_labels,
                f'PCA Visualization of {emb_type.capitalize()} Embeddings\n{model_name.upper()} on {dataset} ({role}, {split_type})',
                output_path / f'{file_name}_{emb_type}_pca.png'
            )
        
        if viz_type in ['tsne', 'both']:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_emb = tsne.fit_transform(combined_emb)
            plot_embeddings(
                tsne_emb,
                combined_labels,
                f't-SNE Visualization of {emb_type.capitalize()} Embeddings\n{model_name.upper()} on {dataset} ({role}, {split_type})',
                output_path / f'{file_name}_{emb_type}_tsne.png'
            )

def main():
    parser = argparse.ArgumentParser(description='Visualize GNN embeddings using PCA and t-SNE')
    parser.add_argument('--embeddings-path', type=str, required=True, help='Path to the embeddings file')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE')
    parser.add_argument('--viz-type', type=str, default='both', choices=['pca', 'tsne', 'both'],
                      help='Type of visualization to generate (pca, tsne, or both)')
    args = parser.parse_args()
    
    visualize_embeddings(
        args.embeddings_path,
        args.output_dir,
        args.perplexity,
        args.viz_type
    )

if __name__ == '__main__':
    main() 