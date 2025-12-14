"""Board detection and perspective correction using OpenCV."""

import cv2
import numpy as np






def detect_chessboard(image, debug=False):
    """Detect chessboard corners and return perspective-corrected board and orientation."""
    if image is None or image.size == 0:
        return None, None
        
    auto_corners = automatic_corner_detection(image, debug=debug)
    
    if auto_corners is not None:
        return manual_corner_selection(image, auto_corners)
    else:
        return manual_corner_selection(image)

def manual_corner_selection(image, initial_corners=None):
    """Manual corner selection with drag support."""
    corners = initial_corners.tolist() if initial_corners is not None else []
    dragging = False
    drag_index = -1
    
    def redraw():
        img_copy = image.copy()
        for i, corner in enumerate(corners):
            cv2.circle(img_copy, tuple(map(int, corner)), 8, (0, 255, 0), -1)
            cv2.putText(img_copy, str(i+1), (int(corner[0])+15, int(corner[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if len(corners) == 4:
            pts = np.array(corners, np.int32)
            cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
        cv2.imshow('Select Board Corners', img_copy)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, drag_index
        
        if not dragging and event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 4:
                corners.append([x, y])
            else:
                for i, corner in enumerate(corners):
                    if abs(x - corner[0]) < 15 and abs(y - corner[1]) < 15:
                        dragging = True
                        drag_index = i
                        break
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            corners[drag_index] = [x, y]
            redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
    
    cv2.imshow('Select Board Corners', image)
    cv2.setMouseCallback('Select Board Corners', mouse_callback)
    redraw()
    
    print("W/B=save with orientation, SPACE=save (ask orientation), ESC=cancel")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if (key == ord(' ') or key == ord('w') or key == ord('b')) and len(corners) == 4:
            cv2.destroyAllWindows()
            
            # Determine orientation based on key pressed
            if key == ord('w'):
                orientation = 'W'
            elif key == ord('b'):
                orientation = 'B'
            else:  # space key
                orientation = input("Board orientation - W (white) or B (black): ").strip().upper()
                if orientation not in ['W', 'B']:
                    orientation = 'W'
            
            return apply_perspective_transform(image, np.array(corners)), orientation
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None, None

def _load_segmentation_model():
    """Load the board segmentation model."""
    try:
        import torch
        import os
        from .model.board_detector import BoardSegmenter
        
        model_path = "models/board_segmenter.pth"
        if not os.path.exists(model_path):
            return None
        
        model = BoardSegmenter()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception:
        return None

def _predict_mask(model, image, transform):
    """Predict segmentation mask for image."""
    import torch
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, 512))
    input_tensor = transform(image_resized).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask_pred = output.squeeze().numpy()
    
    # Convert to binary mask and resize back to original size
    mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255
    height, width = image.shape[:2]
    return cv2.resize(mask_binary, (width, height))

def _show_mask_overlay(image, mask, title):
    """Show mask overlay on image."""
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]
    combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.imshow(title, combined)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def automatic_corner_detection(image, debug=False):
    """Detect board corners using Hough grid detection, with optional segmentation cropping."""
    # Try segmentation-assisted Hough detection first
    try:
        from .data.segmentation_dataset import SegmentationDataset
        from .utils.segmentation_utils import get_bounds
        
        model = _load_segmentation_model()
        if model is not None:
            transform = SegmentationDataset.get_default_transform()
            mask = _predict_mask(model, image, transform)
            
            # Crop to expanded board region and apply Hough detection
            bounds = get_bounds(mask)
            if bounds is None:
                if debug:
                    print("No valid bounds found from segmentation")
                return None
                
            x1, y1, x2, y2 = bounds
            
            if debug:
                _show_mask_overlay(image, mask, 'Board Prediction Overlay')
            
            x1, y1, x2, y2 = _expand_bounds(x1, y1, x2, y2, image.shape, 1.2)
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                if debug:
                    print("Cropped image is empty")
                return None
            
            if debug:
                cv2.imshow("Cropped Image", cropped_image)
                cv2.waitKey(0)
                cv2.destroyWindow("Cropped Image")
            
            from .hough_grid import detect_chessboard_grid
            corners = detect_chessboard_grid(cropped_image, debug=debug)
            
            if corners is not None:
                # Adjust corners to original image coordinates
                corners[:, 0] += x1
                corners[:, 1] += y1
                return corners.astype(int)
            
    except Exception as e:
        if debug:
            print(f"Automatic detection failed: {e}")
    
    return None

def apply_perspective_transform(image, corners):
    """Apply perspective transform to get normalized board view."""
    corners = order_corners(corners)
    
    target_size = 512
    target_corners = np.array([
        [0, 0], [target_size, 0], [target_size, target_size], [0, target_size]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
    return cv2.warpPerspective(image, matrix, (target_size, target_size))

def order_corners(corners):
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    corners = corners[corners[:, 1].argsort()]
    top_points = corners[:2][corners[:2, 0].argsort()]
    bottom_points = corners[2:][corners[2:, 0].argsort()]
    
    return np.array([
        top_points[0], top_points[1], bottom_points[1], bottom_points[0]
    ])

def _expand_bounds(x1, y1, x2, y2, image_shape, scale_factor):
    """Expand bounding box by scale_factor while keeping within image bounds."""
    height, width = image_shape[:2]
    
    # Calculate current dimensions and centers
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Calculate new dimensions
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Calculate new bounds centered on original center
    new_x1 = max(0, cx - new_w // 2)
    new_y1 = max(0, cy - new_h // 2)
    new_x2 = min(width, cx + new_w // 2)
    new_y2 = min(height, cy + new_h // 2)
    
    return new_x1, new_y1, new_x2, new_y2