"""Main script to run chess position detection on screenshots."""

import argparse
import os
from src.classify_board import ChessBoardClassifier

def main():
    import sys
    parser = argparse.ArgumentParser(description="Chess position detection from screenshots")
    parser.add_argument("image_path", help="Path to screenshot image")
    parser.add_argument("--model_path", default="models/chess_classifier.pth", 
                       help="Path to trained model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to run inference on")
    parser.add_argument("--absolute", action="store_true", 
                       help="Use absolute path (default: relative to ~/Desktop)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with visualizations")
    
    args = parser.parse_args()
    
    # Resolve image path
    if args.absolute:
        image_path = args.image_path
    else:
        desktop_path = os.path.expanduser("~/Desktop")
        image_path = os.path.join(desktop_path, args.image_path)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first using src/training/train_classifier.py")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Initialize classifier
    classifier = ChessBoardClassifier(args.model_path, args.device)
    
    # Classify position
    print(f"Processing image: {image_path}")
    fen, orientation = classifier.classify_screenshot(image_path, debug=args.debug)
    
    if fen:
        print(f"Detected FEN: {fen}")
        
        # Generate Lichess analysis link
        lichess_url = f"https://lichess.org/analysis/{fen.replace(' ', '_')}"
        print(f"Lichess Analysis: {lichess_url}")
        
        import subprocess
        subprocess.run(["python3", "board_editor/chess_editor.py", "--fen", fen, "--orientation", orientation])
    else:
        print("Failed to detect chessboard in image")

if __name__ == "__main__":
    main()