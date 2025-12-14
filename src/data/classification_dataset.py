"""Dataset for piece classification training from annotated boards."""

import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from ..inference.detect_board import apply_perspective_transform
from ..inference.extract_squares import extract_squares
from ..utils.fen import fen_to_board_array
from ..model.classifier import PIECE_CLASSES



class ClassificationDataset(Dataset):
    """Dataset for training piece classification from board annotations."""
    
    def __init__(self, annotations_file, transform=None, augment=True):
        with open(annotations_file, 'r') as f:
            all_annotations = json.load(f)
        
        # Filter to only include images with both corners AND FEN
        self.annotations = [ann for ann in all_annotations 
                          if ann.get('corners') and ann.get('fen')]
        print(f"Filtered {len(all_annotations)} -> {len(self.annotations)} images with FEN")
        
        self.transform = transform or self._get_default_transform()
        self.augment = augment
        
        # Build samples with class balancing
        self.samples = self._build_balanced_samples()
        print(f"Built {len(self.samples)} balanced samples")
    
    def _build_balanced_samples(self):
        """Build balanced samples to prevent bias toward common pieces."""
        # First collect all samples by class
        class_samples = {piece_class: [] for piece_class in PIECE_CLASSES}
        
        for board_idx, annotation in enumerate(self.annotations):
            try:
                # Get piece positions from FEN
                pieces = fen_to_board_array(annotation['fen'])
                if annotation.get('orientation', 'W') == 'B':
                    pieces = np.flip(pieces)
                
                # Collect all squares from this board
                for square_idx, piece_label in enumerate(pieces.flatten()):
                    if piece_label in PIECE_CLASSES:
                        class_samples[piece_label].append((board_idx, square_idx, piece_label))
                        
            except Exception:
                continue
        
        # Find the smallest class size
        class_sizes = {piece_class: len(samples) for piece_class, samples in class_samples.items()}
        min_class_size = min(class_sizes.values())
        max_samples_per_class = min_class_size * 2  # 2x the smallest class
        
        print(f"Class sizes: {class_sizes}")
        print(f"Limiting each class to {max_samples_per_class} samples (2x smallest class: {min_class_size})")
        
        # Balance classes
        balanced_samples = []
        import random
        
        for piece_class, samples in class_samples.items():
            if len(samples) > max_samples_per_class:
                samples = random.sample(samples, max_samples_per_class)
            balanced_samples.extend(samples)
            print(f"{piece_class}: {len(samples)} samples (was {class_sizes[piece_class]})")
        
        # Shuffle the final dataset
        random.shuffle(balanced_samples)
        return balanced_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        board_idx, square_idx, piece_label = self.samples[idx]
        annotation = self.annotations[board_idx]
        
        try:
            # Load and process image
            image = cv2.imread(annotation['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract board and squares
            corners = np.array(annotation['corners'])
            board_image = apply_perspective_transform(image, corners)
            squares = extract_squares(board_image)
            square_image = squares[square_idx]
            
            # Apply augmentations and transforms
            if self.augment:
                square_image = self._augment_square(square_image)
            
            return self.transform(square_image), PIECE_CLASSES.index(piece_label)
            
        except Exception:
            # Return empty square as fallback
            empty_square = np.zeros((64, 64, 3), dtype=np.uint8)
            return self.transform(empty_square), 0
    
    def _augment_square(self, square):
        """Apply conservative augmentations to preserve piece colors."""
        # Light brightness/contrast to preserve colors
        if np.random.random() < 0.5:
            alpha = np.random.uniform(0.85, 1.15)  # Light contrast
            beta = np.random.uniform(-8, 8)        # Light brightness
            square = cv2.convertScaleAbs(square, alpha=alpha, beta=beta)
        
        # Small rotation
        if np.random.random() < 0.3:
            angle = np.random.uniform(-3, 3)
            center = (square.shape[1]//2, square.shape[0]//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            square = cv2.warpAffine(square, matrix, square.shape[:2][::-1])
        
        # Light crop
        if np.random.random() < 0.4:
            h, w = square.shape[:2]
            crop_size = int(min(h, w) * np.random.uniform(0.85, 0.95))
            start_y = np.random.randint(0, h - crop_size + 1)
            start_x = np.random.randint(0, w - crop_size + 1)
            square = square[start_y:start_y+crop_size, start_x:start_x+crop_size]
            square = cv2.resize(square, (64, 64))
        
        return square
    
    @staticmethod
    def _get_default_transform():
        """Default transforms for piece training."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])