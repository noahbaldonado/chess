"""Simple Hough-based chessboard grid detection."""

import cv2
import numpy as np
from pathlib import Path

try:
    from .model.grid_detector import detect_grid_with_model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

def detect_chessboard_grid(image, debug=False):
    """Detect 8x8 chessboard grid using simplified Hough line detection."""
    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    min_size = min(image.shape[:2])
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=max(20, min_size//15),
                           minLineLength=max(15, min_size//12),
                           maxLineGap=max(30, min_size//10))
    
    if lines is None or len(lines) < 10:
        return None
    
    # Separate horizontal and vertical lines
    h_lines, v_lines = _separate_lines(lines)
    
    if debug:
        print(f"Found {len(h_lines)} horizontal, {len(v_lines)} vertical lines")
        _show_lines(image, h_lines, v_lines, "Detected Lines")
    
    # Find best grid
    corners = detect_grid_with_model(h_lines, v_lines, image, debug=debug)
    return corners

def _separate_lines(lines):
    """Separate lines into horizontal and vertical."""
    h_lines, v_lines = [], []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2-y1, x2-x1)
        if angle < 0:
            angle += np.pi
        
        # Horizontal: near 0 or π, Vertical: near π/2
        if angle < np.pi/30 or angle > np.pi - np.pi/30:
            h_lines.append((x1, y1, x2, y2))
        elif np.pi/2 - np.pi/30 <= angle <= np.pi/2 + np.pi/30:
            v_lines.append((x1, y1, x2, y2))
    
    return h_lines, v_lines

def _show_lines(image, h_lines, v_lines, title):
    """Visualize lines for debugging."""
    vis = image.copy()
    
    for line in h_lines:
        cv2.line(vis, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 0, 255), 2)
    for line in v_lines:
        cv2.line(vis, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 0, 0), 2)
    
    cv2.putText(vis, f"H: {len(h_lines)} (red), V: {len(v_lines)} (blue)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyWindow(title)