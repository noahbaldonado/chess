import torch
import torch.nn as nn
import torchvision.models as models

class BoardDetector(nn.Module):
    """Predicts 4 board corners = 8 numbers."""

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # keep spatial map

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),   # 4 corners × (x, y)
            nn.Sigmoid()        # normalized
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.fc(x)

class BoardSegmenter(nn.Module):
    """RegNet-based segmentation model for chess boards."""
    
    def __init__(self):
        super().__init__()
        # Use pretrained RegNet as backbone
        regnet = models.regnet_y_400mf(pretrained=True)
        self.backbone = nn.Sequential(*list(regnet.children())[:-2])  # Remove avgpool and fc
        
        # Decoder - adjusted for RegNet output channels (440)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(440, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.decoder(features)