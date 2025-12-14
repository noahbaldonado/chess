# Chess Board Editor

Interactive chess board editor with drag-and-drop functionality.

## Features
- Drag and drop pieces
- Piece palette for adding/removing pieces
- Real-time FEN display and copying
- Board flipping
- Icon support for chess pieces

## Icon Setup

Place chess piece icons in the `icons/` directory with these filenames:

**White pieces:**
- `white_pawn.png`
- `white_knight.png`
- `white_bishop.png`
- `white_rook.png`
- `white_queen.png`
- `white_king.png`

**Black pieces:**
- `black_pawn.png`
- `black_knight.png`
- `black_bishop.png`
- `black_rook.png`
- `black_queen.png`
- `black_king.png`

Icons should be PNG format, ideally 70x70 pixels or similar square dimensions.

If icons are not found, the editor will fall back to Unicode chess symbols.

## Usage

```bash
# Run standalone
python3 chess_editor.py

# Run with initial position
python3 chess_editor.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
```

## Controls
- **Drag pieces**: Click and drag any piece to move it
- **Add pieces**: Click piece in palette, then click board square
- **Remove pieces**: Click "empty" in palette, then click piece to delete
- **Copy FEN**: Click "Copy FEN" button (requires pyperclip: `pip install pyperclip`)
- **Flip Board**: Switch between white/black perspective