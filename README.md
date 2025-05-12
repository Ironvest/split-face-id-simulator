# Split Face ID Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://split-face-id-simulator.streamlit.app/)

Try the live demo: [Split Face ID Simulator](https://split-face-id-simulator.streamlit.app/)

A cutting-edge demonstration of split inference for face recognition systems, showing how computation can be optimally distributed between edge devices and servers.

## 🎯 Goal

Design and implement a Face ID pipeline split between an edge device and a server, such that:

- The edge performs the first part of the model inference
- The edge outputs a compact representation (ideally a single image or small vector)
- The server continues inference using this output, producing the final face embedding
- The split maintains high accuracy and allows auditing and visualization of edge outputs
- The design is efficient enough to run on modern mobile hardware

## 📐 Design Constraints

- Accuracy must not degrade vs full end-to-end inference
- Edge output must be lightweight (low bandwidth), preferably:
  - A single image
  - Or a fixed-length vector
- The representation must be:
  - Deterministic and preferably reversible
  - Optionally visualizable for human inspection or debugging
- Must be practical to run on modern mobile phones (e.g., iPhones, Androids)

## ✅ Model Architecture

- **Backbone**: ResNet-18, pretrained on ImageNet
- **Inference flow**:
  - Input: 3 × 224 × 224 face image
  - Edge: conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool
  - Server: fc (512 → 128) → normalize → embedding

## 🔀 Explored Cut Points for the Split

This app demonstrates different possible split points in the neural network:

1. **After conv1** (first convolutional layer)
   - Output shape: [64, 112, 112]
   - Size: ~800 KB (float32)
   - Too large, not feasible for transmission or visualization
   
2. **After bn1** (batch normalization)
   - Output shape: [64, 112, 112]
   - Size: ~800 KB (float32)
   
3. **After relu** (ReLU activation)
   - Output shape: [64, 112, 112]
   - Size: ~800 KB (float32)
   
4. **After maxpool** (max pooling layer)
   - Output shape: [64, 56, 56]
   - Size: ~200 KB (float32)

5. **After layer1** (first residual block)
   - Output shape: [64, 56, 56] 
   - Size: ~200 KB (float32)

6. **After layer2** (second residual block)
   - Output shape: [128, 28, 28]
   - Size: ~100 KB (float32)

7. **After layer3** (third residual block)
   - Output shape: [256, 14, 14]
   - Size: ~50 KB (float32)

8. **After layer4** (fourth residual block)
   - Output shape: [512, 7, 7]
   - Size: ~25 KB (float32)
   - Still heavy for low-latency transmission

9. ✅ **After avgpool** (global average pooling)
   - Output shape: [512]
   - Size: 2 KB (float32)
   - Easily transmittable and storable
   - Server can perform final projection (fc)
   - Chosen as the best split point

## 📱 Edge Feasibility

- Model size (float32): ~44 MB
- Quantized size (int8): ~11 MB
- Inference cost: ~1.8 GFLOPs
- Run time:
  - iPhone 12+: ~15 ms
  - Pixel 6+: ~20–25 ms
  - Mid-range Android: ~30–50 ms
- **Conclusion**: feasible for on-demand inference

## 🔁 Visualization Options

The app demonstrates several visualization options:

- Feature map visualization with channel grids
- Embedding vectors as bar charts
- Side-by-side comparison of split vs. server-only processing
- Difference highlighting between processing paths

## 📊 Features

- Simulates processing face images through a ResNet18 model split between edge and server
- Configurable split points: conv1, layer1, layer2, layer3, layer4
- Visualizes feature maps from edge processing
- Compares results between split processing and full server processing
- Measures processing time and data transfer metrics
- Includes sample face images for testing

## 🚀 Getting Started

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

## 💻 Usage

1. Select a split point from the sidebar (where to divide processing between edge and server)
2. Upload a face image or use one of the provided sample images
3. Click "Run Face ID" to process the image
4. View the results:
   - Feature maps from edge processing
   - Comparison between split and server-only processing
   - Processing time metrics
   - Data transfer metrics

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── model.py                # Face ID model implementation with split processing
├── visualization.py        # Visualization utilities for feature maps and embeddings
├── data/sample_faces/      # Sample face images for testing
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🧪 Testing Strategy

1. **Unit and Integration Tests**
   - Validate that edge and server produce identical embeddings when chained directly
   - Test serialization-deserialization fidelity of the 512D vector
   - Validate model consistency across backends (PyTorch, TFLite, CoreML)

2. **Accuracy Testing**
   - Compare embeddings from end-to-end full model vs split model
   - Cosine similarity should be > 0.99
   - Run face verification benchmarks on LFW, CFP-FP, IJB-C (if available)

3. **Latency and Throughput Tests**
   - Measure edge inference time (image to vector)
   - Measure transmission + decode + server inference time

4. **Visualization Debugging**
   - Implement image/heatmap visualizers for feature tensors and vectors
   - Display both edge-generated and server-reconstructed representations

## 🔧 Discussion

This project demonstrates a novel approach to Face ID systems by splitting the neural network computation between an edge device and a server. This architecture offers several advantages:

1. **Privacy Enhancement**: By performing initial feature extraction on the edge device, raw face images never leave the user's device. Only abstract feature maps are transmitted to the server.

2. **Reduced Bandwidth**: The feature maps sent to the server are typically smaller than the original image, reducing data transmission requirements.

3. **Computational Balance**: The split architecture allows for efficient use of computational resources, with the edge device handling initial processing and the server managing more complex operations.

4. **Security Through Obscurity**: Since the complete model isn't stored on either the edge device or the server alone, it becomes more difficult for attackers to extract or reverse engineer the model.

## 🚀 Next Steps

1. **Optimization**
   - Implement quantization of feature maps for reduced bandwidth
   - Explore different encoding/compression methods for the intermediate representations
   - Add support for batched processing

2. **Security & Privacy**
   - Add visualization of privacy preservation by attempting to reconstruct original images from feature maps
   - Implement encryption layer for feature map transmission

3. **User Experience**
   - Add real-time webcam support for live demonstrations
   - Implement face detection pre-processing
   - Add support for comparing multiple face embeddings

4. **Educational Features**
   - Add interactive explanations of each processing stage
   - Include performance metrics and system resource usage statistics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 