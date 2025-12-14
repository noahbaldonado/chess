"""Dataset for board segmentation training."""

import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from ..utils.segmentation_utils import create_board_mask

class SegmentationDataset(Dataset):
    """Dataset for training board segmentation model."""
    
    def __init__(self, annotations_file, transform=None, target_size=(224, 224)):
        with open(annotations_file, 'r') as f:
            all_annotations = json.load(f)
        
        # Filter to only include images with board corners
        self.annotations = [ann for ann in all_annotations if 'corners' in ann and ann['corners']]
        print(f"Filtered {len(all_annotations)} -> {len(self.annotations)} images with board corners")
        
        self.transform = transform or self.get_default_transform()
        self.target_size = target_size
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        image = cv2.imread(annotation['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask from corners
        corners = np.array(annotation['corners'])
        mask = create_board_mask(image, corners)
        
        # Resize image and mask
        image_resized = cv2.resize(image, self.target_size)
        mask_resized = cv2.resize(mask, self.target_size)
        
        # Convert to tensors
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        
        mask_tensor = torch.from_numpy(mask_resized).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
        
        return image_tensor, mask_tensor
    
    @staticmethod
    def get_default_transform():
        """Default transforms for training."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])