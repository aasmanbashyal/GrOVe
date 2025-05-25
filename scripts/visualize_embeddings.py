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

def visualize_embeddings(embeddings_path, output_dir, perplexity=30):
    """Visualize embeddings using PCA and t-SNE."""
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
        
        # 1. PCA
        pca = PCA(n_components=2)
        pca_emb = pca.fit_transform(combined_emb)
        plot_embeddings(
            pca_emb, 
            combined_labels,
            f'PCA Visualization of {emb_type.capitalize()} Embeddings\n{model_name.upper()} on {dataset} ({role}, {split_type})',
            output_path / f'{file_name}_{emb_type}_pca.png'
        )
        
        # 2. t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_emb = tsne.fit_transform(combined_emb)
        plot_embeddings(
            tsne_emb,
            combined_labels,
            f't-SNE Visualization of {emb_type.capitalize()} Embeddings\n{model_name.upper()} on {dataset} ({role}, {split_type})',
            output_path / f'{file_name}_{emb_type}_tsne.png'
        )
        
        # Create a combined plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Embedding Visualizations for {model_name.upper()} on {dataset} ({role}, {split_type})\n{emb_type.capitalize()} Embeddings', fontsize=16)
        
        # PCA
        scatter1 = axes[0].scatter(pca_emb[:, 0], pca_emb[:, 1], c=combined_labels, cmap='tab10', alpha=0.6)
        axes[0].set_title('PCA')
        plt.colorbar(scatter1, ax=axes[0])
        
        # t-SNE
        scatter2 = axes[1].scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=combined_labels, cmap='tab10', alpha=0.6)
        axes[1].set_title('t-SNE')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_path / f'{file_name}_{emb_type}_combined.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize GNN embeddings using PCA and t-SNE')
    parser.add_argument('--embeddings-path', type=str, required=True, help='Path to the embeddings file')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE')
    args = parser.parse_args()
    
    visualize_embeddings(
        args.embeddings_path,
        args.output_dir,
        args.perplexity
    )

if __name__ == '__main__':
    main() 