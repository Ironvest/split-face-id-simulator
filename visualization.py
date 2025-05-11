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

def fig_to_image(fig):
    """Convert a matplotlib figure to an image for Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf 