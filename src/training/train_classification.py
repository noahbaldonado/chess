"""Training script for chess piece classification."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from ..model.classifier import ChessPieceClassifier
from ..data.classification_dataset import ClassificationDataset

def train_classification(annotations_file, model_save_path, epochs=50, batch_size=32, lr=0.001):
    """Train chess piece classification model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load and validate dataset
    dataset = ClassificationDataset(annotations_file)
    print(f"Dataset: {len(dataset)} squares from {len(dataset.annotations)} boards")
    
    if len(dataset) == 0:
        print("Error: No training data found! Use batch_annotate.py to add FEN annotations.")
        return
    
    # Setup training
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ChessPieceClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_correct = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        
        # Metrics and saving
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved: {best_val_acc:.4f}")
        
        scheduler.step()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    annotations_file = "data/annotations.json"
    model_save_path = "models/chess_classifier.pth"
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    train_classification(annotations_file, model_save_path)