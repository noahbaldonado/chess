"""Batch annotation tool for unannotated screenshots."""

import os
import cv2
import numpy as np
import json
import argparse

def get_annotations(annotations_file):
    """Get annotation status for all images."""
    annotations_map = {}
    if not os.path.exists(annotations_file):
        return {}
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    for ann in annotations:
        annotations_map[ann['image_path']] = ann
    return annotations_map

def annotate_image(image_path, annotations_map, fen_mode, use_fen_editor=True, classifier=None):
    """Annotate corners for a single image using two-corner method."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"image in annotations_map: {image_path in annotations_map}")
    if image_path in annotations_map and 'corners' in annotations_map[image_path]:
        print(f"corners: {annotations_map[image_path]['corners']}")
        print(f"\nCompleting annotation for: {os.path.basename(image_path)}")
        # Convert existing 4 corners to TL and BR
        existing_corners = annotations_map[image_path]['corners']
        x_coords = [c[0] for c in existing_corners]
        y_coords = [c[1] for c in existing_corners]
        corners = [[min(x_coords), min(y_coords)], [max(x_coords), max(y_coords)]]
    else:
        print(f"\nAnnotating: {os.path.basename(image_path)}")
        corners = []
    
    dragging = False
    drag_index = -1
    
    def redraw():
        img_copy = image.copy()
        for i, corner in enumerate(corners):
            cv2.circle(img_copy, tuple(map(int, corner)), 8, (0, 255, 0), -1)
            label = "TL" if i == 0 else "BR"
            cv2.putText(img_copy, label, (int(corner[0])+15, int(corner[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show rectangle if both corners selected
        if len(corners) == 2:
            x1, y1 = corners[0]
            x2, y2 = corners[1]
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        cv2.imshow('Annotate Board', img_copy)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, drag_index
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 2:
                corners.append([x, y])
                label = "top-left" if len(corners) == 1 else "bottom-right"
                print(f"Selected {label} corner")
            else:
                # Check if clicking on existing corner for dragging
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
            drag_index = -1
    
    cv2.imshow('Annotate Board', image)
    cv2.setMouseCallback('Annotate Board', mouse_callback)
    redraw()
    
    print("Click top-left then bottom-right corner. W/B=save with orientation, SPACE=save (ask orientation), R=reset, ESC=skip")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if (key == ord(' ') or key == ord('w') or key == ord('b')) and len(corners) == 2:
            cv2.destroyAllWindows()
            
            # Convert 2 corners to 4 corners (rectangle)
            x1, y1 = corners[0]
            x2, y2 = corners[1]
            four_corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            # Get FEN and orientation
            if not fen_mode:
                annotations_map[image_path] = {
                    'image_path': image_path,
                    'corners': four_corners,
                }
                return
            
            # Determine orientation based on key pressed
            if key == ord('w'):
                orientation = 'W'
            elif key == ord('b'):
                orientation = 'B'
            else:  # space key
                orientation = input("Board orientation - W (white) or B (black): ").strip().upper()
                if orientation not in ['W', 'B']:
                    orientation = 'W'
            
            if use_fen_editor:
                # Use classifier to predict FEN and open board editor
                fen = predict_and_edit_fen(image_path, four_corners, orientation, classifier)
            else:
                # Manual FEN entry
                fen = input("Enter FEN position (or press Enter to skip): ").strip()
            if not fen:
                print("Skipping image (no FEN provided)")
                annotations_map[image_path] = {
                    'image_path': image_path,
                    'corners': four_corners,
                }
                return
            
            annotations_map[image_path] = {
                'image_path': image_path,
                'corners': four_corners,
                'fen': fen,
                'orientation': orientation
            }
            return
        elif key == ord('r'):
            corners.clear()
            redraw()
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return

def predict_and_edit_fen(image_path, corners, orientation, classifier=None):
    """Use classifier to predict FEN from annotated corners."""
    import sys
    import cv2
    import numpy as np
    import subprocess
    import os
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.inference.classify_board import ChessBoardClassifier
        from src.inference.detect_board import apply_perspective_transform
        from src.inference.extract_squares import extract_squares
        
        # Use pre-loaded classifier or load if not provided
        if classifier is None:
            model_path = "models/chess_classifier.pth"
            if not os.path.exists(model_path):
                print("Classifier model not found. Using manual FEN entry.")
                fen = input("Enter FEN position (or press Enter to skip): ").strip()
                return fen
            classifier = ChessBoardClassifier(model_path)
        
        # Load image and apply perspective transform using annotated corners
        print("Predicting FEN using classifier...")
        image = cv2.imread(image_path)
        board_image = apply_perspective_transform(image, np.array(corners))
        
        # Extract squares
        squares = extract_squares(board_image)
        predictions = classifier._classify_squares(squares)
        
        # Convert to FEN
        board_array = np.array(predictions).reshape(8, 8)
        
        # Flip board if from black's perspective
        if orientation == 'B':
            board_array = np.flip(board_array)
        
        from src.utils.fen import board_array_to_fen
        predicted_fen = board_array_to_fen(board_array)
        
        if predicted_fen:
            print(f"Predicted FEN: {predicted_fen}")
            
            # Show the original image side-by-side with board editor
            # Position OpenCV window on the left
            cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Original Image', 50, 50)
            cv2.resizeWindow('Original Image', 600, 400)
            
            # Load and display original image
            display_image = cv2.imread(image_path)
            # Draw the detected board area
            pts = np.array(corners, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_image, [pts], True, (0, 255, 0), 3)
            cv2.imshow('Original Image', display_image)
            
            # Open board editor with predicted FEN for correction
            print("Opening board editor for correction...")
            print("Close the board editor when done to continue.")
            
            # Set environment variable to position pygame window on the right
            env = os.environ.copy()
            env['SDL_VIDEO_WINDOW_POS'] = '700,50'  # Position on the right
            
            result = subprocess.run([
                "python3", "board_editor/chess_editor.py", 
                "--fen", predicted_fen,
                "--orientation", orientation,
                "--return-fen"
            ], capture_output=True, text=True, env=env)
            
            # Close the image window
            cv2.destroyWindow('Original Image')
            
            if result.returncode == 0 and result.stdout.strip():
                # Extract only the last line (the FEN) to avoid pygame output
                lines = result.stdout.strip().split('\n')
                corrected_fen = lines[-1].strip()
                print(f"FEN: {corrected_fen}")
                return corrected_fen
            else:
                print(f"Board editor process returned code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                print(f"Here's the predicted fen:\n{predicted_fen}")
                fen = input("Enter FEN position (or press Enter to skip): ").strip()
                return predicted_fen
        else:
            print("FEN prediction failed. Using manual entry.")
            fen = input("Enter FEN position (or press Enter to skip): ").strip()
            return fen
            
    except Exception as e:
        print(f"Error in FEN prediction: {e}")
        fen = input("Enter FEN position (or press Enter to skip): ").strip()
        return fen

def main():
    import sys
    
    parser = argparse.ArgumentParser(description="Batch annotate screenshots")
    parser.add_argument("--screenshots_dir", default="data/screenshots")
    parser.add_argument("--annotations", default="data/annotations.json")
    parser.add_argument("--mode", choices=["no fen", "fen"], default="fen")
    parser.add_argument("--fen-editor", action="store_true", default=True)
    
    args = parser.parse_args()
    fen_mode = (args.mode == "fen")
    
    # Load existing annotations
    annotations_map = get_annotations(args.annotations)
    
    # Find images to process
    to_process = []
    for filename in sorted(os.listdir(args.screenshots_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.screenshots_dir, filename)
            annotation = annotations_map.get(image_path, {})
            
            needs_processing = (
                image_path not in annotations_map or
                (fen_mode and "fen" not in annotation) or
                (not fen_mode and "corners" not in annotation)
            )
            
            if needs_processing:
                to_process.append(image_path)
    
    if not to_process:
        print("No images need annotation!")
        return
    
    print(f"Found {len(to_process)} images to process")
    
    # Pre-load classifier once if using FEN editor
    classifier = None
    if args.fen_editor and fen_mode:
        model_path = "models/chess_classifier.pth"
        if os.path.exists(model_path):
            print("Loading classifier model...")
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.inference.classify_board import ChessBoardClassifier
            classifier = ChessBoardClassifier(model_path)
            print("Classifier loaded!")
    
    # Process each image
    for image_path in to_process:
        annotate_image(image_path, annotations_map, fen_mode, args.fen_editor, classifier)
        
        # Save after each annotation
        os.makedirs(os.path.dirname(args.annotations), exist_ok=True)
        with open(args.annotations, 'w') as f:
            json.dump(list(annotations_map.values()), f, indent=2)
        
        print(f"Saved: {os.path.basename(image_path)}")
    
    print(f"Complete! Total annotations: {len(annotations_map)}")

if __name__ == "__main__":
    main()