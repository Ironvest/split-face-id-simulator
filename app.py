import streamlit as st
import torch
from PIL import Image
import numpy as np
from model import FaceIDModel
import visualization as viz
import os
import time

def get_sample_images():
    """Get list of sample face images."""
    sample_dir = 'data/sample_faces'
    return [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

def load_image(image_file):
    """Load and preprocess an image file."""
    if image_file is not None:
        return Image.open(image_file)
    return None

def format_size(size_bytes):
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    st.title("Split Face ID Simulator")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = FaceIDModel()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Split point selection
    split_point = st.sidebar.selectbox(
        "Select Split Point",
        ['conv1', 'layer1', 'layer2', 'layer3', 'layer4'],
        index=0,
        help="Choose where to split the model between edge and server"
    )
    
    # Update model if split point changes
    if split_point != st.session_state.model.split_point:
        st.session_state.model = FaceIDModel(split_point=split_point)
    
    # Image input options
    use_sample = st.sidebar.checkbox("Use sample image")
    if use_sample:
        sample_images = get_sample_images()
        selected_image = st.sidebar.selectbox("Select sample image", sample_images)
        if selected_image:
            image_path = os.path.join('data/sample_faces', selected_image)
            image = Image.open(image_path)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'])
        image = load_image(uploaded_file)
    
    if image is not None:
        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Process image
        if st.sidebar.button("Run Face ID"):
            # Start timing
            start_time = time.time()
            
            # Preprocess image
            input_tensor = st.session_state.model.preprocess_image(image)
            
            # Get tensor shapes
            shapes = st.session_state.model.get_tensor_shapes(input_tensor)
            
            # Process through edge device
            edge_start = time.time()
            edge_output = st.session_state.model.forward_edge(input_tensor)
            edge_time = time.time() - edge_start
            
            # Process through server
            server_start = time.time()
            split_embedding = st.session_state.model.forward_server(edge_output)
            server_time = time.time() - server_start
            
            # Process through server-only path
            server_only_start = time.time()
            server_embedding = st.session_state.model.forward_server_only(input_tensor)
            server_only_time = time.time() - server_only_start
            
            total_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.edge_output = edge_output
            st.session_state.split_embedding = split_embedding
            st.session_state.server_embedding = server_embedding
            st.session_state.shapes = shapes
            st.session_state.edge_time = edge_time
            st.session_state.server_time = server_time
            st.session_state.total_time = total_time
            st.session_state.has_results = True
        
        # Display results if available
        if st.session_state.get('has_results', False):
            # Edge Device Output
            st.subheader("Edge Device Output")
            
            # Show feature maps
            edge_viz = viz.get_feature_map_visualization(st.session_state.edge_output)
            st.image(edge_viz, use_column_width=True)
            st.text(f"Tensor shape: {st.session_state.shapes['edge_output']}")
            
            # Comparison
            st.subheader("Comparison: Split vs Full Server Processing")
            comparison_viz = viz.get_embedding_comparison(
                st.session_state.split_embedding,
                st.session_state.server_embedding
            )
            st.image(comparison_viz, use_column_width=True)
            
            # Metrics
            st.subheader("Metrics")
            
            # Create a 2x3 grid for all metrics
            cols = st.columns(3)
            
            # Row 1: Similarity metrics
            with cols[0]:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    st.session_state.split_embedding.flatten(),
                    st.session_state.server_embedding.flatten(),
                    dim=0
                ).item()
                st.metric("Cosine Similarity", f"{cosine_sim:.6f}")
            
            with cols[1]:
                l2_dist = torch.norm(st.session_state.split_embedding - st.session_state.server_embedding).item()
                st.metric("L2 Distance", f"{l2_dist:.6f}")
            
            with cols[2]:
                st.metric("Edge Processing", f"{st.session_state.edge_time*1000:.2f} ms")
            
            # Row 2: Processing and transfer metrics
            cols = st.columns(3)
            with cols[0]:
                st.metric("Server Processing", f"{st.session_state.server_time*1000:.2f} ms")
            
            with cols[1]:
                st.metric("Total Time", f"{st.session_state.total_time*1000:.2f} ms")
            
            with cols[2]:
                data_size = np.prod(st.session_state.shapes['edge_output']) * 4  # 4 bytes per float32
                st.metric("Data Transfer Size", format_size(data_size))
            
            # Display split point information
            st.sidebar.markdown("---")
            st.sidebar.subheader("Split Point Information")
            st.sidebar.markdown(f"""
            - **Current Split**: {split_point}
            - **Edge Device Layers**: {len(st.session_state.model.edge_layers)} layers
            - **Server Layers**: {len(st.session_state.model.server_layers)} layers
            - **Data Transfer Size**: {format_size(data_size)}
            """)

if __name__ == "__main__":
    main()