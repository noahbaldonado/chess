"""Visualize training data samples with augmentations."""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.classification_dataset import ClassificationDataset
from src.model.classifier import PIECE_CLASSES

def denormalize_tensor(tensor):
    """Convert normalized tensor back to displayable image."""
    # Denormalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert tensor to numpy and denormalize
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def visualize_samples(annotations_file="data/annotations.json", num_samples=20):
    """Visualize random samples from the dataset."""
    
    # Create dataset with augmentations
    dataset = ClassificationDataset(annotations_file, augment=True)
    print(f"Dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("No training data found!")
        return
    
    # Show samples
    for i in range(min(num_samples, len(dataset))):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        tensor, label_idx = dataset[idx]
        piece_name = PIECE_CLASSES[label_idx]
        
        # Convert tensor back to displayable image
        img = denormalize_tensor(tensor)
        
        # Resize for better visibility
        img_large = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Add label text
        cv2.putText(img_large, piece_name, (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_large, piece_name, (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Show image
        cv2.imshow(f'Training Sample {i+1}/{num_samples}', img_large)
        
        print(f"Sample {i+1}: {piece_name}")
        print("Press SPACE for next, ESC to exit")
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            cv2.destroyAllWindows()
            continue
    
    cv2.destroyAllWindows()

def compare_augmentations(annotations_file="data/annotations.json"):
    """Compare original vs augmented versions of the same sample."""
    
    # Create datasets with and without augmentations
    dataset_aug = ClassificationDataset(annotations_file, augment=True)
    dataset_orig = ClassificationDataset(annotations_file, augment=False)
    
    if len(dataset_aug) == 0:
        print("No training data found!")
        return
    
    print(f"Comparing original vs augmented samples...")
    
    for i in range(min(10, len(dataset_aug))):
        idx = np.random.randint(0, len(dataset_aug))
        
        # Get same sample with and without augmentation
        tensor_aug, label_idx = dataset_aug[idx]
        tensor_orig, _ = dataset_orig[idx]
        piece_name = PIECE_CLASSES[label_idx]
        
        # Convert tensors to images
        img_aug = denormalize_tensor(tensor_aug)
        img_orig = denormalize_tensor(tensor_orig)
        
        # Resize for better visibility
        img_aug = cv2.resize(img_aug, (128, 128), interpolation=cv2.INTER_NEAREST)
        img_orig = cv2.resize(img_orig, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Create side-by-side comparison
        comparison = np.hstack([img_orig, img_aug])
        
        # Add labels
        cv2.putText(comparison, f"Original - {piece_name}", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(comparison, f"Original - {piece_name}", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(comparison, "Augmented", (135, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(comparison, "Augmented", (135, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow(f'Comparison {i+1}/10', comparison)
        
        print(f"Comparison {i+1}: {piece_name}")
        print("Press SPACE for next, ESC to exit")
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            cv2.destroyAllWindows()
            continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize training data")
    parser.add_argument("--mode", choices=["samples", "compare"], default="samples",
                       help="samples: show random samples, compare: compare original vs augmented")
    parser.add_argument("--annotations", default="data/annotations.json")
    parser.add_argument("--num-samples", type=int, default=20)
    
    args = parser.parse_args()
    
    if args.mode == "samples":
        visualize_samples(args.annotations, args.num_samples)
    else:
        compare_augmentations(args.annotations)