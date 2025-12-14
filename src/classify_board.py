"""Board classification using trained CNN model."""

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from .model.classifier import ChessPieceClassifier, PIECE_CLASSES
from .utils.fen import board_array_to_fen
from .detect_board import detect_chessboard
from .extract_squares import extract_squares

# Constants
COLOR_SIMILARITY_THRESHOLD = 0.9
BOOST_FACTOR = 1.2  # Light boosting
COLOR_QUANTIZATION = 16

class ChessBoardClassifier:
    """Complete pipeline for classifying chess positions from screenshots."""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint."""
        model = ChessPieceClassifier()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def classify_screenshot(self, image_path, debug=False):
        """Classify chess position from screenshot.
        
        Returns:
            FEN string or None if detection fails
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # Detect board and get orientation
        board_image, orientation = detect_chessboard(image, debug=debug)
        if board_image is None:
            return None, None
        
        # Extract and classify squares
        squares = extract_squares(board_image)
        predictions = self._classify_squares(squares)
        board_array = np.array(predictions).reshape(8, 8)
        
        # Flip board if from black's perspective
        if orientation == 'B':
            board_array = np.flip(board_array)
        
        cv2.destroyAllWindows()
        return board_array_to_fen(board_array), orientation
    
    def _classify_squares(self, squares):
        """Classify list of square images with color-aware correction."""
        # Get raw predictions and probabilities
        raw_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for square in squares:
                square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                tensor = self.transform(Image.fromarray(square_rgb)).unsqueeze(0).to(self.device)
                
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                pred_idx = np.argmax(probs)
                
                raw_predictions.append(PIECE_CLASSES[pred_idx])
                all_probs.append(probs)
        
        # Apply color-aware correction
        return self._apply_color_correction(squares, raw_predictions, all_probs)
    
    def _apply_color_correction(self, squares, predictions, all_probs):
        """Apply light color-based probability boosting."""
        # Extract colors and calculate type representatives
        center_colors = [self._extract_center_color(square) for square in squares]
        type_colors = self._calculate_type_colors(center_colors, predictions)
        
        corrected_predictions = []
        
        for i, pred in enumerate(predictions):
            probs = all_probs[i].copy()
            
            # Light boost for matching color types
            for type_name, type_color in type_colors.items():
                if type_color is not None and self._colors_match(center_colors[i], type_color):
                    self._boost_probabilities(probs, type_name)
            
            corrected_predictions.append(PIECE_CLASSES[np.argmax(probs)])
        
        return corrected_predictions
    
    def _extract_center_color(self, square):
        """Extract mode color from center rectangle."""
        h, w = square.shape[:2]
        x1, x2 = w // 3, 2 * w // 3
        y1, y2 = h // 5, 4 * h // 5
        
        rect_pixels = square[y1:y2, x1:x2]
        return self._calculate_color_mode(rect_pixels.reshape(-1, 3)) if rect_pixels.size > 0 else np.array([128, 128, 128])
    
    def _calculate_color_mode(self, pixels):
        """Calculate the mode (most common) color."""
        if len(pixels) == 0:
            return np.array([128, 128, 128])
        
        pixels = np.array(pixels)
        quantized = (pixels // COLOR_QUANTIZATION) * COLOR_QUANTIZATION
        unique_colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        return unique_colors[np.argmax(counts)].astype(np.float64)
    
    def _calculate_type_colors(self, center_colors, predictions):
        """Calculate representative colors for each type."""
        color_groups = {'black_piece': [], 'white_piece': [], 'dark_square': [], 'light_square': []}
        
        for i, pred in enumerate(predictions):
            color = center_colors[i]
            
            if pred.startswith('b'):
                color_groups['black_piece'].append(color)
            elif pred.startswith('w'):
                color_groups['white_piece'].append(color)
            elif pred == 'empty':
                row, col = i // 8, i % 8
                square_type = 'dark_square' if (row + col) % 2 == 0 else 'light_square'
                color_groups[square_type].append(color)
        
        return {name: self._calculate_color_mode(colors) if colors else None 
                for name, colors in color_groups.items()}
    
    def _boost_probabilities(self, probs, type_name):
        """Light boost for matching piece types."""
        if type_name == 'black_piece':
            for j, piece_class in enumerate(PIECE_CLASSES):
                if piece_class.startswith('b'):
                    probs[j] *= BOOST_FACTOR
        elif type_name == 'white_piece':
            for j, piece_class in enumerate(PIECE_CLASSES):
                if piece_class.startswith('w'):
                    probs[j] *= BOOST_FACTOR
        elif type_name in ['dark_square', 'light_square']:
            for j, piece_class in enumerate(PIECE_CLASSES):
                if piece_class == 'empty':
                    probs[j] *= BOOST_FACTOR
    
    def _colors_match(self, color1, color2):
        """Check if two colors match within threshold."""
        brightness_diff = abs(np.mean(color1) - np.mean(color2))
        similarity = 1.0 - (brightness_diff / 255.0)
        return similarity > COLOR_SIMILARITY_THRESHOLD
    

    
