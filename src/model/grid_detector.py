"""Neural network for detecting 8x8 grids from line data."""

import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm

class GridDetector(nn.Module):
    """CNN that takes line representation and outputs grid corners."""
    
    def __init__(self, image_size=256):
        super().__init__()
        self.image_size = image_size
        
        # Simple CNN for line pattern recognition
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )
        
        # Output 8 coordinates (4 corners x 2)
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # x1,y1,x2,y2,x3,y3,x4,y4
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def lines_to_image(h_lines, v_lines, image_size=256):
    """Convert line data to binary image for CNN input."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    all_lines = h_lines + v_lines
    
    if not all_lines:
        return torch.FloatTensor(img).unsqueeze(0) / 255.0
    
    # Calculate bounds and scaling
    all_coords = [coord for line in all_lines for coord in line]
    min_coord, max_coord = min(all_coords), max(all_coords)
    original_size = max_coord - min_coord
    
    if original_size <= 0:
        return torch.FloatTensor(img).unsqueeze(0) / 255.0
    
    scale = image_size / original_size
    
    # Draw scaled lines
    for x1, y1, x2, y2 in all_lines:
        x1 = int((x1 - min_coord) * scale)
        y1 = int((y1 - min_coord) * scale)
        x2 = int((x2 - min_coord) * scale)
        y2 = int((y2 - min_coord) * scale)
        
        x1, x2 = np.clip([x1, x2], 0, image_size-1)
        y1, y2 = np.clip([y1, y2], 0, image_size-1)
        cv2.line(img, (x1, y1), (x2, y2), 255, 1)
    
    return torch.FloatTensor(img).unsqueeze(0) / 255.0

def _remove_duplicate_lines(lines, threshold=20):
    """Remove lines that are too close to each other."""
    if not lines:
        return lines
    
    def get_line_pos(line):
        x1, y1, x2, y2 = line
        return (y1 + y2) / 2 if abs(y2 - y1) < abs(x2 - x1) else (x1 + x2) / 2
    
    filtered = []
    for line in lines:
        line_pos = get_line_pos(line)
        if not any(abs(line_pos - get_line_pos(existing)) < threshold for existing in filtered):
            filtered.append(line)
    
    return filtered

def detect_grid_with_model(h_lines, v_lines, original_image=None, model_path="models/grid_detector.pth", debug=False, num_candidates=5):
    """Use trained model to detect grid, then refine with line scoring."""
    # Load model and get initial prediction
    model = GridDetector()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    line_img = lines_to_image(h_lines, v_lines)
    with torch.no_grad():
        corners = model(line_img.unsqueeze(0)).squeeze().numpy().reshape(4, 2)
    
    # Scale to original coordinates
    all_coords = [coord for line in h_lines + v_lines for coord in line]
    if all_coords:
        min_coord, max_coord = min(all_coords), max(all_coords)
        corners = corners * (max_coord - min_coord) + min_coord
    
    if debug:
        _show_corners_on_lines(line_img, corners, all_coords, "AI Predicted Corners")
    
    # Two-stage refinement
    coarse_candidates = _find_best_corners(corners, _remove_duplicate_lines(h_lines), _remove_duplicate_lines(v_lines), 0.5, original_image, num_candidates)
    
    best_score, best_corners = float('-inf'), corners
    for candidate in coarse_candidates:
        # Fine search with strict duplicate removal (threshold=1)
        fine_h = _remove_duplicate_lines(h_lines, threshold=3)
        fine_v = _remove_duplicate_lines(v_lines, threshold=3)
        refined = _find_best_corners(candidate, fine_h, fine_v, 0.1, original_image, 1)[0]
        score = _get_total_score(refined, fine_h, fine_v, original_image)
        
        if debug:
            print(f"Candidate score: {score:.2f}")
        #     _show_corners_on_lines(line_img, refined, all_coords, "Refined Candidate Corners")
        
        if score > best_score:
            best_score, best_corners = score, refined
    
    if debug:
        _show_corners_on_lines(line_img, best_corners, all_coords, "Final Corners")
    
    return best_corners

def _show_corners_on_lines(line_img, corners, all_coords, title):
    """Show corners overlaid on line image."""
    # Convert to display format
    line_img_display = (line_img.squeeze().numpy() * 255).astype(np.uint8)
    line_img_color = cv2.cvtColor(line_img_display, cv2.COLOR_GRAY2BGR)
    
    # Scale corners to image coordinates
    if all_coords:
        min_coord, max_coord = min(all_coords), max(all_coords)
        original_size = max_coord - min_coord
        corners_scaled = (corners - min_coord) / original_size * 256
    else:
        corners_scaled = corners
    
    # Draw corners
    for i, (x, y) in enumerate(corners_scaled):
        x_img, y_img = int(np.clip(x, 0, 255)), int(np.clip(y, 0, 255))
        cv2.circle(line_img_color, (x_img, y_img), 5, (0, 0, 255), -1)
        cv2.putText(line_img_color, str(i), (x_img+10, y_img), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow(title, line_img_color)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def _find_best_corners(predicted_corners, h_lines, v_lines, tolerance_factor, original_image, max_candidates):
    """Find best corner candidates with given tolerance."""
    tl, tr, br, bl = predicted_corners
    tolerance = min(abs(tr[0] - tl[0]), abs(bl[1] - tl[1])) * tolerance_factor
    
    line_candidates = {
        'top': _get_nearby_lines(h_lines, (tl[1] + tr[1]) / 2, [tl, tr], tolerance, 'h'),
        'bottom': _get_nearby_lines(h_lines, (bl[1] + br[1]) / 2, [bl, br], tolerance, 'h'),
        'left': _get_nearby_lines(v_lines, (tl[0] + bl[0]) / 2, [tl, bl], tolerance, 'v'),
        'right': _get_nearby_lines(v_lines, (tr[0] + br[0]) / 2, [tr, br], tolerance, 'v')
    }
    
    scored_candidates = []
    total = len(line_candidates['top']) * len(line_candidates['bottom']) * len(line_candidates['left']) * len(line_candidates['right'])
    
    desc = "Coarse search" if max_candidates > 1 else "Fine search"
    show_progress = max_candidates > 1
    
    iterator = tqdm(total=total, desc=desc) if show_progress else range(total)
    
    for top in line_candidates['top']:
        for bottom in line_candidates['bottom']:
            for left in line_candidates['left']:
                for right in line_candidates['right']:
                    corners = _corners_from_lines(top, bottom, left, right)
                    if corners is not None:
                        score = _get_total_score(corners, h_lines, v_lines, original_image)
                        scored_candidates.append((score, corners))
                    if show_progress:
                        iterator.update(1)
    
    scored_candidates.sort(reverse=True)
    return [corners for _, corners in scored_candidates[:max_candidates]] if scored_candidates else [predicted_corners]

def _get_total_score(corners, h_lines, v_lines, original_image):
    """Get total score for corner configuration."""
    line_score, pattern_score, _ = _score_grid(corners, h_lines, v_lines, original_image=original_image)
    return line_score + pattern_score

def _get_nearby_lines(lines, target_pos, corners, tolerance, line_type):
    """Get lines near target position within tolerance."""
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line
        line_pos = (y1 + y2) / 2 if line_type == 'h' else (x1 + x2) / 2
        pos_dist = abs(line_pos - target_pos)
        
        if pos_dist <= tolerance:
            pts = [(x1, y1), (x2, y2)] if (line_type == 'h' and x1 <= x2) or (line_type == 'v' and y1 <= y2) else [(x2, y2), (x1, y1)]
            endpoint_dist = sum(np.linalg.norm(np.array(pt) - corner) for pt, corner in zip(pts, corners))
            
            if pos_dist + 0.3 * endpoint_dist < tolerance * 2:
                candidates.append(line)
    
    return candidates

def _corners_from_lines(top_line, bottom_line, left_line, right_line):
    """Calculate corners from boundary line positions."""
    positions = {
        'top_y': (top_line[1] + top_line[3]) / 2,
        'bottom_y': (bottom_line[1] + bottom_line[3]) / 2,
        'left_x': (left_line[0] + left_line[2]) / 2,
        'right_x': (right_line[0] + right_line[2]) / 2
    }
    
    width = abs(positions['right_x'] - positions['left_x'])
    height = abs(positions['bottom_y'] - positions['top_y'])
    
    if width <= 0 or height <= 0 or abs(width - height) / max(width, height) > 0.15:
        return None
    
    return np.array([
        [positions['left_x'], positions['top_y']],
        [positions['right_x'], positions['top_y']],
        [positions['right_x'], positions['bottom_y']],
        [positions['left_x'], positions['bottom_y']]
    ])

def _score_grid(corners, h_lines, v_lines, original_image=None, line_score_weight=3.0, debug=False):
    """Score how well the configuration matches an 8x8 grid."""
    unit_spacing = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[3] - corners[0])) / 16
    if unit_spacing <= 0:
        return 0
    
    # Get boundaries and internal lines
    top_y, bottom_y = (corners[0][1] + corners[1][1]) / 2, (corners[2][1] + corners[3][1]) / 2
    left_x, right_x = (corners[0][0] + corners[3][0]) / 2, (corners[1][0] + corners[2][0]) / 2
    
    h_internal = [line for line in h_lines if top_y < (line[1]+line[3])/2 < bottom_y]
    v_internal = [line for line in v_lines if left_x < (line[0]+line[2])/2 < right_x]
    
    # Score line alignment
    line_score = 0
    for i in range(1, 8):
        # Horizontal lines
        expected_y = top_y + i * unit_spacing
        if h_internal:
            closest_dist = min(abs((line[1]+line[3])/2 - expected_y) for line in h_internal)
            if closest_dist < 0.3 * unit_spacing:
                line_score += 1.0 - (closest_dist / (0.3 * unit_spacing))
        
        # Vertical lines
        expected_x = left_x + i * unit_spacing
        if v_internal:
            closest_dist = min(abs((line[0]+line[2])/2 - expected_x) for line in v_internal)
            if closest_dist < 0.3 * unit_spacing:
                line_score += 1.0 - (closest_dist / (0.3 * unit_spacing))
    
    line_score *= line_score_weight
    
    # Add chessboard pattern score
    pattern_score = 1.0
    if original_image is not None:
        pattern_score, square_values = _score_chessboard_pattern(corners, original_image, debug)
    return line_score, pattern_score, square_values if original_image is not None else []

def _score_chessboard_pattern(corners, image, debug=False):
    """Score based on alternating square pattern."""
    try:
        # Extract 8x8 grid from image
        h, w = image.shape[:2]
        corners_clipped = np.clip(corners, [0, 0], [w-1, h-1]).astype(int)
        
        # Simple perspective transform to 64x64
        target = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(corners_clipped.astype(np.float32), target)
        warped = cv2.warpPerspective(image, matrix, (64, 64))
        
        # Convert to grayscale and sample 8x8 squares efficiently
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        
        # Vectorized edge sampling - sample border pixels only
        square_values = np.zeros(64)
        for i in range(64):
            row, col = i // 8, i % 8
            y1, y2 = row * 8, (row + 1) * 8
            x1, x2 = col * 8, (col + 1) * 8
            
            # Sample just the border (top, bottom, left, right edges)
            square_region = gray[y1:y2, x1:x2]
            border_pixels = np.concatenate([
                square_region[0, :],      # Top edge
                square_region[-1, :],     # Bottom edge  
                square_region[1:-1, 0],   # Left edge (excluding corners)
                square_region[1:-1, -1]   # Right edge (excluding corners)
            ])
            
            square_values[i] = np.mean(border_pixels)
        
        # Vectorized odd/even calculation
        indices = np.arange(64)
        rows, cols = indices // 8, indices % 8
        is_odd = (rows + cols) % 2 == 1
        
        odd_avg = np.mean(square_values[is_odd])
        even_avg = np.mean(square_values[~is_odd])
        
        # Vectorized variance calculation
        expected_values = np.where(is_odd, odd_avg, even_avg)
        total_variance = np.sum((square_values - expected_values) ** 2)
        
        # Use negative variance scaled down (lower variance = higher score)
        final_score = -total_variance / 10000
        return final_score, square_values
    
    except:
        return 1.0, []  # Fallback if pattern scoring fails

def _show_checkerboard_overlay(image, corners, square_values):
    """Show checkerboard pattern overlay on original image."""
    if len(square_values) != 64:
        return
    
    overlay = image.copy()
    h, w = image.shape[:2]
    corners_clipped = np.clip(corners, [0, 0], [w-1, h-1]).astype(int)
    
    # Draw grid and color squares
    for row in range(8):
        for col in range(8):
            # Calculate square corners in original image
            tl = corners_clipped[0] + (corners_clipped[1] - corners_clipped[0]) * col/8 + (corners_clipped[3] - corners_clipped[0]) * row/8
            tr = corners_clipped[0] + (corners_clipped[1] - corners_clipped[0]) * (col+1)/8 + (corners_clipped[3] - corners_clipped[0]) * row/8
            bl = corners_clipped[0] + (corners_clipped[1] - corners_clipped[0]) * col/8 + (corners_clipped[3] - corners_clipped[0]) * (row+1)/8
            br = corners_clipped[0] + (corners_clipped[1] - corners_clipped[0]) * (col+1)/8 + (corners_clipped[3] - corners_clipped[0]) * (row+1)/8
            
            # Color based on calculated square value
            value = int(square_values[row * 8 + col])
            color = (value, value, value)
            
            # Draw semi-transparent square
            pts = np.array([tl, tr, br, bl], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
    
    # Show solid overlay without blending
    cv2.imshow("Checkerboard Pattern", overlay)
    cv2.waitKey(0)
    cv2.destroyWindow("Checkerboard Pattern")