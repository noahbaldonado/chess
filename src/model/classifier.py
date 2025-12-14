"""CNN classifier for chess piece recognition."""

import torch
import torch.nn as nn
import torchvision.models as models

# Constants - Full 13-class system
PIECE_CLASSES = ["empty", "wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
NUM_CLASSES = len(PIECE_CLASSES)

class ChessPieceClassifier(nn.Module):
    """CNN classifier using MobileNetV3."""
    
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # 13 classes
        )
    
    def forward(self, x):
        return self.backbone(x)