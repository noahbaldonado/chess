"""Generate training data for grid detection model."""

import cv2
import numpy as np
import torch
import json
from pathlib import Path

def generate_synthetic_grid(image_size=256):
    """Generate synthetic chessboard with random noise lines."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Random grid position and size (allow smaller grids)
    margin = image_size // 10
    grid_size = np.random.randint(image_size//3, image_size - 2*margin)
    start_x = np.random.randint(margin, image_size - grid_size - margin)
    start_y = np.random.randint(margin, image_size - grid_size - margin)
    
    # Draw 9x9 grid lines with some missing/broken portions
    spacing = grid_size / 8
    h_lines, v_lines = [], []
    
    for i in range(9):
        # Horizontal lines (sometimes broken)
        if np.random.random() > 0.1:  # 90% chance to draw
            y = int(start_y + i * spacing)
            x1, x2 = start_x, start_x + grid_size
            
            # Sometimes break the line into segments
            if np.random.random() > 0.7:  # 30% chance to break
                break_start = np.random.randint(x1 + grid_size//4, x1 + 3*grid_size//4)
                break_end = break_start + np.random.randint(grid_size//8, grid_size//4)
                # Draw two segments
                cv2.line(img, (x1, y), (break_start, y), 255, 1)
                cv2.line(img, (break_end, y), (x2, y), 255, 1)
                h_lines.extend([(x1, y, break_start, y), (break_end, y, x2, y)])
            else:
                cv2.line(img, (x1, y), (x2, y), 255, 1)
                h_lines.append((x1, y, x2, y))
        
        # Vertical lines (sometimes broken)
        if np.random.random() > 0.1:  # 90% chance to draw
            x = int(start_x + i * spacing)
            y1, y2 = start_y, start_y + grid_size
            
            # Sometimes break the line into segments
            if np.random.random() > 0.7:  # 30% chance to break
                break_start = np.random.randint(y1 + grid_size//4, y1 + 3*grid_size//4)
                break_end = break_start + np.random.randint(grid_size//8, grid_size//4)
                # Draw two segments
                cv2.line(img, (x, y1), (x, break_start), 255, 1)
                cv2.line(img, (x, break_end), (x, y2), 255, 1)
                v_lines.extend([(x, y1, x, break_start), (x, break_end, x, y2)])
            else:
                cv2.line(img, (x, y1), (x, y2), 255, 1)
                v_lines.append((x, y1, x, y2))
    
    # Add horizontal and vertical noise lines only
    num_noise = np.random.randint(5, 30)
    for _ in range(num_noise):
        if np.random.random() > 0.5:  # Horizontal noise line
            y = np.random.randint(0, image_size)
            x1 = np.random.randint(0, image_size//2)
            x2 = np.random.randint(image_size//2, image_size)
            cv2.line(img, (x1, y), (x2, y), 255, 1)
        else:  # Vertical noise line
            x = np.random.randint(0, image_size)
            y1 = np.random.randint(0, image_size//2)
            y2 = np.random.randint(image_size//2, image_size)
            cv2.line(img, (x, y1), (x, y2), 255, 1)
    
    # Ground truth corners
    corners = np.array([
        [start_x, start_y],                    # top-left
        [start_x + grid_size, start_y],        # top-right
        [start_x + grid_size, start_y + grid_size],  # bottom-right
        [start_x, start_y + grid_size]         # bottom-left
    ]) / image_size  # Normalize to [0,1]
    
    return img, corners.flatten(), h_lines, v_lines

def generate_dataset(num_samples=1000, output_dir="data/grid_training"):
    """Generate training dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    images = []
    labels = []
    
    for i in range(num_samples):
        img, corners, h_lines, v_lines = generate_synthetic_grid()
        
        images.append(img)
        labels.append(corners)
        
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} samples")
    
    # Save as tensors
    torch.save({
        'images': torch.FloatTensor(np.array(images)).unsqueeze(1) / 255.0,
        'labels': torch.FloatTensor(np.array(labels))
    }, f"{output_dir}/training_data.pt")
    
    print(f"Saved {num_samples} samples to {output_dir}/training_data.pt")

if __name__ == "__main__":
    generate_dataset()