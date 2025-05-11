# Split Face ID Simulator

This web application simulates a Face ID processing pipeline that is split between an edge device and a backend server. The app visually demonstrates image transformations and feature maps at each processing stage.

## Features

- **Split Processing Pipeline**: Simulates running the initial convolution block on an edge device and the remaining blocks on a backend server
- **Visualizations**:
  - Original image display
  - Feature map visualization from edge device processing
  - Final 128D face embedding visualization
- **Tensor Shapes**: Shows tensor dimensions at each step of the pipeline

## Model Architecture

- **Backbone**: ResNet18 pretrained on ImageNet
- **Split Architecture**:
  - **Edge Device**: Conv1 → BatchNorm → ReLU → MaxPool
  - **Server**: Remaining ResNet blocks
  - **Embedding Projection**: Linear layer projecting to 128D embedding

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/split-face-id-simulator.git
   cd split-face-id-simulator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   Open your web browser and go to `http://localhost:8501`

## Usage

1. Upload a face image using the file uploader in the sidebar, or check the "Use sample image" option
2. Click the "Run Face ID" button to process the image
3. View the visualizations:
   - Original input image
   - Feature maps from edge device processing
   - 128D face embedding bar chart

## Project Structure

- `app.py`: Streamlit application code
- `model.py`: Face ID model with split processing functionality
- `visualization.py`: Functions for visualizing feature maps and embeddings
- `data/`: Directory for sample face images
- `requirements.txt`: Project dependencies

## Technology Stack

- Python 3.8+
- Streamlit
- PyTorch and torchvision
- Matplotlib for visualizations 