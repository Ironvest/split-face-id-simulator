# Split Face ID Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://split-face-id-simulator.streamlit.app/)

Try the live demo: [Split Face ID Simulator](https://split-face-id-simulator.streamlit.app/)

This project simulates a split Face ID system that processes face recognition tasks between edge devices and a server. It demonstrates how different split points in a neural network affect processing time, data transfer, and accuracy.

## Features

- Simulates processing face images through a ResNet18 model split between edge and server
- Configurable split points: conv1, layer1, layer2, layer3, layer4
- Visualizes feature maps from edge processing
- Compares results between split processing and full server processing
- Measures processing time and data transfer metrics
- Includes sample face images for testing

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Streamlit
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ironvest/split-face-id-simulator.git
cd split-face-id-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Select a split point from the sidebar (where to divide processing between edge and server)
2. Upload a face image or use one of the provided sample images
3. Click "Run Face ID" to process the image
4. View the results:
   - Feature maps from edge processing
   - Comparison between split and server-only processing
   - Processing time metrics
   - Data transfer metrics

## Project Structure

- `app.py`: Main Streamlit application
- `model.py`: Face ID model implementation with split processing
- `visualization.py`: Visualization utilities for feature maps and embeddings
- `data/sample_faces/`: Sample face images for testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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

## Discussion

This project demonstrates a novel approach to Face ID systems by splitting the neural network computation between an edge device and a server. This architecture offers several advantages:

1. **Privacy Enhancement**: By performing initial feature extraction on the edge device, raw face images never leave the user's device. Only abstract feature maps are transmitted to the server.

2. **Reduced Bandwidth**: The feature maps sent to the server are typically smaller than the original image, reducing data transmission requirements.

3. **Computational Balance**: The split architecture allows for efficient use of computational resources, with the edge device handling initial processing and the server managing more complex operations.

4. **Security Through Obscurity**: Since the complete model isn't stored on either the edge device or the server alone, it becomes more difficult for attackers to extract or reverse engineer the model.

## Next Steps

The demo could be enhanced with the following features:

1. **Network Analysis**:
   - Add bandwidth usage metrics to compare data transmission between split and full-server approaches
   - Implement simulated network latency to demonstrate real-world performance
   - Visualize data compression ratios of feature maps vs. original images

2. **Security Demonstrations**:
   - Add visualization of privacy preservation by attempting to reconstruct original images from feature maps
   - Implement attack simulations to demonstrate the security benefits of split architecture
   - Add encryption layer for feature map transmission

3. **Performance Optimizations**:
   - Implement quantization of feature maps for reduced bandwidth
   - Add support for batched processing
   - Explore different split points in the network architecture

4. **User Experience**:
   - Add real-time webcam support for live demonstrations
   - Implement face detection pre-processing
   - Add support for comparing multiple face embeddings
   - Include face verification/matching demonstrations

5. **Educational Features**:
   - Add interactive explanations of each processing stage
   - Visualize attention maps and layer activations
   - Include performance metrics and system resource usage statistics 