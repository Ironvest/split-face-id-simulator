import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def get_feature_map_visualization(feature_maps):
    """
    Visualize feature maps from the edge device.
    
    Args:
        feature_maps (torch.Tensor): Feature maps from edge processing
    
    Returns:
        PIL.Image: Visualization of feature maps
    """
    # Convert to numpy
    feature_maps = feature_maps.detach().cpu().numpy()
    
    # If feature maps are 4D (batch, channels, height, width), reshape to show all channels
    if len(feature_maps.shape) == 4:
        # Remove batch dimension and reshape to show all channels
        feature_maps = feature_maps[0]  # Take first batch
        n_maps = feature_maps.shape[0]  # Number of channels
    else:
        n_maps = feature_maps.shape[0]
    
    # Normalize each feature map
    feature_maps = np.array([(fm - fm.min()) / (fm.max() - fm.min() + 1e-8) for fm in feature_maps])
    
    # Get dimensions
    n_cols = 8  # Show 8 columns for better visibility
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2*n_rows))
    axes = axes.flatten()
    
    # Plot each feature map
    for i in range(n_maps):
        ax = axes[i]
        im = ax.imshow(feature_maps[i], cmap='viridis')
        ax.set_title(f'Channel {i+1}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig_to_image(fig)

def get_embedding_visualization(embedding):
    """
    Visualize face embedding as a horizontal bar chart.
    
    Args:
        embedding (torch.Tensor): Face embedding vector
    
    Returns:
        PIL.Image: Visualization of embedding
    """
    # Convert to numpy
    embedding = embedding.detach().cpu().numpy().flatten()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(range(len(embedding)), embedding)
    ax.set_title('Face Embedding')
    ax.set_xlabel('Value')
    ax.set_ylabel('Dimension')
    
    # Convert figure to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def get_embedding_comparison(split_embedding, server_embedding):
    """
    Compare two embeddings (from split processing and server-only processing).
    
    Args:
        split_embedding (torch.Tensor): Embedding from split processing
        server_embedding (torch.Tensor): Embedding from server-only processing
    
    Returns:
        PIL.Image: Visualization of comparison
    """
    # Convert to numpy
    split_emb = split_embedding.detach().cpu().numpy().flatten()
    server_emb = server_embedding.detach().cpu().numpy().flatten()
    
    # Calculate difference
    diff = split_emb - server_emb
    max_diff = np.max(np.abs(diff))
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot split embedding
    colors1 = plt.cm.RdBu_r((split_emb - split_emb.min()) / (split_emb.max() - split_emb.min() + 1e-8))
    ax1.barh(range(len(split_emb)), split_emb, color=colors1)
    ax1.set_title('Split Processing Embedding')
    
    # Plot server embedding
    colors2 = plt.cm.RdBu_r((server_emb - server_emb.min()) / (server_emb.max() - server_emb.min() + 1e-8))
    ax2.barh(range(len(server_emb)), server_emb, color=colors2)
    ax2.set_title('Server-Only Embedding')
    
    # Plot difference
    colors3 = plt.cm.RdBu_r((diff + max_diff) / (2 * max_diff + 1e-8))
    ax3.barh(range(len(diff)), diff, color=colors3)
    ax3.set_title(f'Difference (Max: {max_diff:.6f})')
    
    plt.tight_layout()
    return fig_to_image(fig) 