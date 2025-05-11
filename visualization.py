import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

def get_feature_map_visualization(feature_maps, num_channels=4, full_view=False):
    """
    Visualize feature maps from the edge device
    
    Args:
        feature_maps: Tensor of shape [1, C, H, W]
        num_channels: Number of channels to visualize by default
        full_view: If True, display all channels (max 64)
    
    Returns:
        A figure with feature map visualizations
    """
    # Take the first batch and select channels to visualize
    feature_maps = feature_maps[0].detach().cpu().numpy()
    
    # Determine how many channels to show
    if full_view:
        num_channels = min(64, feature_maps.shape[0])
    else:
        num_channels = min(num_channels, feature_maps.shape[0])
    
    # Create a grid for the feature maps (4 columns)
    n_cols = 4
    n_rows = (num_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_channels):
        # Normalize the feature map for better visualization
        feature_map = feature_maps[i]
        vmin, vmax = feature_map.min(), feature_map.max()
        normalized_map = (feature_map - vmin) / (vmax - vmin + 1e-8)
        
        axes[i].imshow(normalized_map, cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def get_embedding_visualization(embedding):
    """
    Visualize the face embedding as a horizontal bar chart
    
    Args:
        embedding: Tensor of shape [1, D] where D is the embedding dimension
    
    Returns:
        A figure with embedding visualization
    """
    embedding = embedding.detach().cpu().numpy().flatten()
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(embedding))
    
    # Use a colormap to color bars based on values
    colors = plt.cm.RdBu_r((embedding - embedding.min()) / (embedding.max() - embedding.min()))
    
    ax.barh(y_pos, embedding, color=colors)
    ax.set_yticks(y_pos[::16])  # Show every 16th tick to avoid crowding
    ax.set_yticklabels([f'{i}' for i in y_pos[::16]])
    ax.set_xlabel('Value')
    ax.set_ylabel('Dimension')
    ax.set_title('128-Dimensional Face Embedding')
    
    plt.tight_layout()
    return fig

def get_embedding_comparison(split_embedding, server_embedding):
    """
    Compare two embeddings and visualize their difference
    
    Args:
        split_embedding: Tensor from split processing
        server_embedding: Tensor from server-only processing
    
    Returns:
        A figure with comparison visualization
    """
    # Convert to numpy arrays
    split_emb = split_embedding.detach().cpu().numpy().flatten()
    server_emb = server_embedding.detach().cpu().numpy().flatten()
    
    # Calculate difference
    diff = split_emb - server_emb
    max_diff = np.max(np.abs(diff))
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot split embedding
    y_pos = np.arange(len(split_emb))
    colors1 = plt.cm.RdBu_r((split_emb - split_emb.min()) / (split_emb.max() - split_emb.min()))
    ax1.barh(y_pos, split_emb, color=colors1)
    ax1.set_yticks(y_pos[::16])
    ax1.set_yticklabels([f'{i}' for i in y_pos[::16]])
    ax1.set_title('Split Processing Embedding')
    
    # Plot server embedding
    colors2 = plt.cm.RdBu_r((server_emb - server_emb.min()) / (server_emb.max() - server_emb.min()))
    ax2.barh(y_pos, server_emb, color=colors2)
    ax2.set_yticks(y_pos[::16])
    ax2.set_yticklabels([f'{i}' for i in y_pos[::16]])
    ax2.set_title('Server-Only Processing Embedding')
    
    # Plot difference
    colors3 = plt.cm.RdBu_r((diff + max_diff) / (2 * max_diff))
    ax3.barh(y_pos, diff, color=colors3)
    ax3.set_yticks(y_pos[::16])
    ax3.set_yticklabels([f'{i}' for i in y_pos[::16]])
    ax3.set_title(f'Difference (Max: {max_diff:.6f})')
    
    plt.tight_layout()
    return fig

def fig_to_image(fig):
    """Convert a matplotlib figure to an image for Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf 