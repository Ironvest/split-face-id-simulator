import streamlit as st
import torch
import torchvision
from PIL import Image
import os
import glob
import numpy as np
from model import FaceIDModel
from visualization import get_feature_map_visualization, get_embedding_visualization, fig_to_image

# Set page config
st.set_page_config(page_title="Split Face ID Simulator", layout="wide")

@st.cache_resource
def load_model():
    """Load the Face ID model with caching for performance"""
    model = FaceIDModel(embedding_dim=128)
    model.eval()
    return model

def create_data_directory():
    """Create data directory if it doesn't exist"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/sample_faces', exist_ok=True)

def load_image(image_file):
    """Load an image from file"""
    if image_file is not None:
        img = Image.open(image_file).convert('RGB')
        return img
    return None

def get_sample_images():
    """Get list of sample face images"""
    sample_images = glob.glob("data/sample_faces/*.jpg")
    return {os.path.basename(f).replace('.jpg', '').replace('_', ' ').title(): f for f in sample_images}

def run_face_id_pipeline(model, image):
    """Run the split Face ID pipeline on the input image"""
    # Preprocess the image
    input_tensor = model.preprocess_image(image)
    
    with torch.no_grad():
        # Edge device processing (first part)
        edge_output = model.backbone_layer1(input_tensor)
        
        # Print tensor shapes
        st.sidebar.write("#### Tensor Shapes:")
        st.sidebar.write(f"Input: {list(input_tensor.shape)}")
        st.sidebar.write(f"Edge Output: {list(edge_output.shape)}")
        
        # Server processing (second part)
        embedding = model.forward_from_intermediate(edge_output)
        st.sidebar.write(f"Final Embedding: {list(embedding.shape)}")
        
        return edge_output, embedding

def main():
    # Create data directory
    create_data_directory()
    
    # Load the model
    model = load_model()
    
    # App header
    st.title("Split Face ID Processing Simulator")
    st.write("""
    This app demonstrates a Face ID processing pipeline split between an edge device and a backend server.
    Upload a face image or select a sample to see the processing steps.
    """)
    
    # Sidebar inputs
    st.sidebar.title("Input")
    
    # Input selection tabs
    input_method = st.sidebar.radio("Select input method", ["Upload Image", "Use Sample Image"])
    
    # Initialize the image
    image = None
    
    if input_method == "Upload Image":
        # File uploader
        uploaded_file = st.sidebar.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
    else:
        # Sample image selection
        sample_images = get_sample_images()
        
        if sample_images:
            selected_sample = st.sidebar.selectbox(
                "Select a sample face", 
                list(sample_images.keys())
            )
            
            if selected_sample:
                image_path = sample_images[selected_sample]
                image = Image.open(image_path).convert('RGB')
                st.sidebar.info(f"Selected: {selected_sample}")
        else:
            st.sidebar.warning("No sample images found in data/sample_faces directory")
            
            # Create a default sample image if no samples exist
            sample_image = Image.new('RGB', (224, 224), color=(73, 109, 137))
            image = sample_image
            # Save for future use
            sample_image.save(os.path.join("data", "sample.jpg"))
    
    # Main content layout - three columns for the three processing stages
    col1, col2, col3 = st.columns(3)
    
    # Display the input image if available
    if image is not None:
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Input Face Image", use_column_width=True)
            
            # Process button below the image
            process_button = st.button("Run Face ID")
        
        if process_button:
            with st.spinner("Processing..."):
                # Run the pipeline
                edge_output, embedding = run_face_id_pipeline(model, image)
                
                # Display feature maps in the middle column
                with col2:
                    st.subheader("Edge Device Output")
                    
                    # Default view with 4 channels
                    fig = get_feature_map_visualization(edge_output, num_channels=4, full_view=False)
                    st.pyplot(fig)
                    
                    # Expandable section for all channels
                    with st.expander("Show all 64 feature maps"):
                        full_fig = get_feature_map_visualization(edge_output, full_view=True)
                        st.pyplot(full_fig)
                
                # Display embedding in the right column
                with col3:
                    st.subheader("Server Output (128D Embedding)")
                    fig = get_embedding_visualization(embedding)
                    st.pyplot(fig)
    else:
        st.info("Please upload an image or select a sample to start.")
        
    # Add app description in the sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This simulator demonstrates how Face ID processing can be split between:
        - **Edge Device**: Initial convolutional layers
        - **Server**: Remaining layers and embedding generation
        
        The model uses ResNet18 pretrained on ImageNet as a backbone.
        """)

if __name__ == "__main__":
    main() 