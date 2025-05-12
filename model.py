import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import transforms as torch_transforms
from PIL import Image

class FaceIDModel:
    def __init__(self, split_point='conv1'):
        """
        Initialize the Face ID model with configurable split point.
        
        Args:
            split_point (str): Where to split the model between edge and server.
                Options: 'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        """
        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Define split points
        self.split_points = {
            'conv1': 0,
            'bn1': 1,
            'relu': 2,
            'maxpool': 3,
            'layer1': 4,
            'layer2': 5,
            'layer3': 6,
            'layer4': 7,
            'avgpool': 8  # Added avgpool as a split point
        }
        
        self.split_point = split_point
        self.split_idx = self.split_points[split_point]
        
        # Split the model into edge and server parts
        self.edge_layers = list(self.model.children())[:self.split_idx + 1]
        self.server_layers = list(self.model.children())[self.split_idx + 1:]
        
        # Add projection layer for face embedding
        self.projection = nn.Linear(512, 128)
        
        # Create edge and server models
        self.edge_model = nn.Sequential(*self.edge_layers)
        self.server_model = nn.Sequential(
            *self.server_layers,
            nn.Flatten(),
            self.projection
        )
        
        # Create full model for server-only processing
        self.full_model = nn.Sequential(
            self.model,
            nn.Flatten(),
            self.projection
        )
        
        # Set to evaluation mode
        self.edge_model.eval()
        self.server_model.eval()
        self.full_model.eval()
        
        # Define image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward_edge(self, x):
        """Run the edge device portion of the model."""
        return self.edge_model(x)
    
    def forward_server(self, x):
        """Run the server portion of the model."""
        return self.server_model(x)
    
    def forward_server_only(self, x):
        """Run the entire model on the server."""
        return self.full_model(x)
    
    def get_tensor_shapes(self, x):
        """Get tensor shapes at each stage of processing."""
        shapes = {
            'input': x.shape,
            'edge_output': self.forward_edge(x).shape,
            'server_output': self.forward_server(self.forward_edge(x)).shape,
            'server_only_output': self.forward_server_only(x).shape
        }
        return shapes
    
    def preprocess_image(self, image):
        """Preprocess an image for the model"""
        return self.preprocess(image).unsqueeze(0)  # Add batch dimension 