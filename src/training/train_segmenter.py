"""Training script for board segmentation model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from ..model.board_detector import BoardSegmenter
from ..data.segmentation_dataset import SegmentationDataset

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def train_segmenter(annotations_file, model_save_path, epochs=100, batch_size=4, lr=0.001):
    """Train board segmentation model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load dataset
    dataset = SegmentationDataset(annotations_file)
    print(f"Dataset size: {len(dataset)} images")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = BoardSegmenter().to(device)
    
    # Loss function - combination of BCE and Dice loss
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Resize outputs to match mask size if needed
            if outputs.shape != masks.shape:
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # Combined loss
            bce = bce_loss(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce + dice
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                if outputs.shape != masks.shape:
                    outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                bce = bce_loss(outputs, masks)
                dice = dice_loss(outputs, masks)
                loss = bce + dice
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
        scheduler.step()
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    annotations_file = "data/annotations.json"
    model_save_path = "models/board_segmenter.pth"
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    train_segmenter(annotations_file, model_save_path)