"""Extract individual squares from normalized chessboard."""

import cv2
import numpy as np

def extract_squares(board_image):
    """Extract 64 individual squares from normalized board image.
    
    Returns:
        List of 64 square images (8x8 grid, row-major order)
    """
    height, width = board_image.shape[:2]
    square_size = height // 8
    squares = []
    
    for row in range(8):
        for col in range(8):
            y1, y2 = row * square_size, (row + 1) * square_size
            x1, x2 = col * square_size, (col + 1) * square_size
            
            square = board_image[y1:y2, x1:x2]
            squares.append(cv2.resize(square, (64, 64)))
    
    return squares

