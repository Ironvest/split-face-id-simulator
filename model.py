import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class FaceIDModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceIDModel, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Split the model: first part runs on edge device
        self.backbone_layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # Rest of the backbone runs on the server
        self.backbone_rest = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        # Final embedding projection
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        """Full forward pass through the model"""
        x = self.backbone_layer1(x)
        x = self.backbone_rest(x)
        x = self.embedding(x)
        return x

    def forward_split(self, x):
        """Split forward pass that returns both intermediate output and final embedding"""
        edge_output = self.backbone_layer1(x)
        server_output = self.backbone_rest(edge_output)
        final_embedding = self.embedding(server_output)
        return edge_output, final_embedding

    def forward_from_intermediate(self, x):
        """Resume processing from intermediate tensor (output from edge device)"""
        x = self.backbone_rest(x)
        x = self.embedding(x)
        return x

    def preprocess_image(self, image):
        """Preprocess an image for the model"""
        return self.preprocess(image).unsqueeze(0)  # Add batch dimension 