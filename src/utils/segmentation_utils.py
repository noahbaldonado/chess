"""Utilities for board segmentation and corner extraction."""

import cv2
import numpy as np

def get_bounds(mask):
    """
    mask: numpy array (H, W) with values in {0,1} or floats
    returns (x1, y1, x2, y2)
    """
    # threshold if needed
    mask_bin = (mask > 0.5).astype(np.uint8)

    # Clean up mask with morphological operations to remove trails
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    # Find contours instead of connected components for better shape analysis
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find best board-like contour (large, roughly square)
    best_contour = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Too small
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Prefer square-ish shapes (chessboards are square)
        aspect_ratio = min(w, h) / max(w, h)
        
        # Score based on area and squareness
        score = area * aspect_ratio
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    if best_contour is None:
        return None
    
    # Get tight bounding box
    x, y, w, h = cv2.boundingRect(best_contour)
    return x, y, x + w, y + h

def create_board_mask(image, corners):
    """Create binary mask from corner annotations."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Create filled polygon from corners
    corners_array = np.array(corners, dtype=np.int32)
    cv2.fillPoly(mask, [corners_array], 255)
    
    return mask

def extract_corners_from_mask(mask):
    """Extract 4 corners from segmentation mask."""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour (should be the board)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we get 4 points, great! Otherwise, find convex hull
    if len(approx) == 4:
        corners = approx.reshape(4, 2)
    else:
        # Use convex hull and find 4 extreme points
        hull = cv2.convexHull(largest_contour)
        corners = find_extreme_points(hull)
    
    return order_corners(corners)


def find_extreme_points(contour):
    """Find 4 extreme points from contour."""
    contour = contour.reshape(-1, 2)
    
    # Find extreme points
    top_left = contour[np.argmin(contour[:, 0] + contour[:, 1])]
    top_right = contour[np.argmax(contour[:, 0] - contour[:, 1])]
    bottom_right = contour[np.argmax(contour[:, 0] + contour[:, 1])]
    bottom_left = contour[np.argmin(contour[:, 0] - contour[:, 1])]
    
    return np.array([top_left, top_right, bottom_right, bottom_left])

def order_corners(corners):
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    # Sort by y-coordinate
    corners = corners[corners[:, 1].argsort()]
    
    # Top two points
    top_points = corners[:2]
    top_points = top_points[top_points[:, 0].argsort()]
    
    # Bottom two points
    bottom_points = corners[2:]
    bottom_points = bottom_points[bottom_points[:, 0].argsort()]
    
    return np.array([
        top_points[0],      # top-left
        top_points[1],      # top-right
        bottom_points[1],   # bottom-right
        bottom_points[0]    # bottom-left
    ])