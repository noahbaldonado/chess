"""Train the grid detection model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.grid_detector import GridDetector

def train_model():
    # Load data
    data = torch.load("data/grid_training/training_data.pt")
    images = data['images']
    labels = data['labels']
    
    # Split train/val
    split = int(0.8 * len(images))
    train_images, val_images = images[:split], images[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    
    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model and optimizer
    model = GridDetector()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(50), desc="Training"):
        # Train
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for images, labels in train_pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False)
            for images, labels in val_pbar:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        tqdm.write(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/grid_detector.pth")
            tqdm.write(f"Saved best model with val loss: {val_loss:.6f}")

if __name__ == "__main__":
    train_model()