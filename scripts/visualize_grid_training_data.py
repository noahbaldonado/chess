"""Visualize synthetic grid training data."""

import cv2
import numpy as np
import torch
from generate_grid_training_data import generate_synthetic_grid

def visualize_training_samples(num_samples=10):
    """Generate and display training samples."""
    for i in range(num_samples):
        img, corners, h_lines, v_lines = generate_synthetic_grid()
        
        # Convert to color for visualization
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw ground truth corners
        corners_2d = corners.reshape(4, 2) * 256  # Denormalize
        for j, (x, y) in enumerate(corners_2d):
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(j), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw corner connections
        pts = corners_2d.astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 0, 0), 2)
        
        cv2.imshow(f"Training Sample {i+1}", vis)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            break

def visualize_saved_data(data_path="data/grid_training/training_data.pt", num_samples=10):
    """Visualize saved training data."""
    data = torch.load(data_path)
    images = data['images']
    labels = data['labels']
    
    for i in range(min(num_samples, len(images))):
        img = (images[i].squeeze().numpy() * 255).astype(np.uint8)
        corners = labels[i].numpy().reshape(4, 2) * 256  # Denormalize
        
        # Convert to color
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw corners
        for j, (x, y) in enumerate(corners):
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(j), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw corner connections
        pts = corners.astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 0, 0), 2)
        
        cv2.imshow(f"Saved Sample {i+1}", vis)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            break

if __name__ == "__main__":
    print("Generating new samples...")
    visualize_training_samples(5)
    
    print("Loading saved data...")
    try:
        visualize_saved_data(num_samples=5)
    except FileNotFoundError:
        print("No saved data found. Run generate_grid_training_data.py first.")