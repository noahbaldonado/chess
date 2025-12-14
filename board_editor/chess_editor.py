"""Interactive chess board editor using pygame with piece icons."""

import pygame
import sys
import os
# Add parent directory to path to access src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.fen import board_array_to_fen

class ChessEditor:
    def __init__(self, initial_fen=None):
        pygame.init()
        
        # Constants
        self.SQUARE_SIZE = 80
        self.BOARD_SIZE = self.SQUARE_SIZE * 8
        self.PANEL_WIDTH = 300
        self.WINDOW_WIDTH = self.BOARD_SIZE + self.PANEL_WIDTH + 40
        self.WINDOW_HEIGHT = max(self.BOARD_SIZE + 40, 700)  # Ensure minimum height
        
        # Colors
        self.LIGHT_SQUARE = (240, 217, 181)
        self.DARK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT = (255, 255, 0, 128)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        
        # Setup display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Chess Board Editor")
        
        # Fonts
        self.piece_font = pygame.font.Font(None, 60)
        self.ui_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Load piece icons
        self.piece_images = self.load_piece_images()
        
        # Board state - default to starting position
        default_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.board = self.fen_to_board(initial_fen or default_fen)
        self.flipped = False
        self.active_color = 'w'  # 'w' for white, 'b' for black
        
        # Parse turn from FEN
        self.parse_fen_metadata(initial_fen or default_fen)
        
        # Piece symbols (fallback if no icons)
        self.pieces = {
            'wp': '♙', 'wn': '♘', 'wb': '♗', 'wr': '♖', 'wq': '♕', 'wk': '♔',
            'bp': '♟', 'bn': '♞', 'bb': '♝', 'br': '♜', 'bq': '♛', 'bk': '♚',
            'empty': ''
        }
        
        # Drag state
        self.dragging = None
        self.drag_offset = (0, 0)
        self.selected_piece = None
        
        # UI elements
        self.setup_ui()
    
    def load_piece_images(self):
        """Load piece icon images from icons directory."""
        images = {}
        icon_dir = os.path.join(os.path.dirname(__file__), 'icons')
        
        piece_files = {
            'wp': 'white_pawn.png', 'wn': 'white_knight.png', 'wb': 'white_bishop.png',
            'wr': 'white_rook.png', 'wq': 'white_queen.png', 'wk': 'white_king.png',
            'bp': 'black_pawn.png', 'bn': 'black_knight.png', 'bb': 'black_bishop.png',
            'br': 'black_rook.png', 'bq': 'black_queen.png', 'bk': 'black_king.png'
        }
        
        for piece, filename in piece_files.items():
            filepath = os.path.join(icon_dir, filename)
            if os.path.exists(filepath):
                try:
                    image = pygame.image.load(filepath)
                    # Scale to fit square with some padding
                    image = pygame.transform.scale(image, (self.SQUARE_SIZE - 10, self.SQUARE_SIZE - 10))
                    images[piece] = image
                except pygame.error:
                    print(f"Could not load {filename}, using text fallback")
        
        return images
    
    def setup_ui(self):
        """Setup UI elements."""
        self.buttons = []
        panel_x = self.BOARD_SIZE + 30
        
        # Control buttons
        button_y = 20
        button_height = 30
        button_spacing = 35
        
        buttons_data = [
            ("Copy FEN", self.copy_fen),
            ("Load FEN", self.load_fen),
            ("Lichess Analysis", self.open_lichess),
            ("Toggle Turn", self.toggle_turn),
            ("Flip Board", self.flip_board),
            ("Clear Board", self.clear_board),
            ("Reset Board", self.reset_board)
        ]
        
        for text, callback in buttons_data:
            rect = pygame.Rect(panel_x, button_y, 120, button_height)
            self.buttons.append((rect, text, callback))
            button_y += button_spacing
        
        # Piece palette (moved down to avoid overlap)
        self.palette = []
        palette_pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk', 'empty']
        
        palette_y = 320  # Moved down from 280
        for i, piece in enumerate(palette_pieces):
            row, col = i // 2, i % 2
            x = panel_x + col * 60
            y = palette_y + row * 40
            rect = pygame.Rect(x, y, 50, 35)
            self.palette.append((rect, piece))
    
    def empty_board(self):
        """Create empty 8x8 board."""
        return [['empty' for _ in range(8)] for _ in range(8)]
    
    def fen_to_board(self, fen):
        """Convert FEN to board array."""
        piece_placement = fen.split()[0]
        board = []
        
        fen_mapping = {
            'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
            'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
        }
        
        for row in piece_placement.split('/'):
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend(['empty'] * int(char))
                else:
                    board_row.append(fen_mapping.get(char, 'empty'))
            board.append(board_row)
        
        return board
    
    def parse_fen_metadata(self, fen):
        """Parse FEN metadata (active color, etc.)."""
        parts = fen.split()
        if len(parts) >= 2:
            self.active_color = parts[1]  # 'w' or 'b'
    
    def get_square_from_pos(self, pos):
        """Get board square from mouse position."""
        x, y = pos
        if x < 20 or y < 20 or x > self.BOARD_SIZE + 20 or y > self.BOARD_SIZE + 20:
            return None
        
        col = (x - 20) // self.SQUARE_SIZE
        row = (y - 20) // self.SQUARE_SIZE
        
        if self.flipped:
            row, col = 7 - row, 7 - col
        
        return (row, col) if 0 <= row < 8 and 0 <= col < 8 else None
    
    def draw_piece(self, piece, x, y):
        """Draw a piece at the given position."""
        if piece == 'empty':
            return
        
        # Try to use icon first, fallback to text
        if piece in self.piece_images:
            image = self.piece_images[piece]
            image_rect = image.get_rect(center=(x + self.SQUARE_SIZE//2, y + self.SQUARE_SIZE//2))
            self.screen.blit(image, image_rect)
        else:
            # Fallback to text
            piece_text = self.piece_font.render(self.pieces[piece], True, self.BLACK)
            piece_rect = piece_text.get_rect(center=(x + self.SQUARE_SIZE//2, y + self.SQUARE_SIZE//2))
            self.screen.blit(piece_text, piece_rect)
    
    def draw_board(self):
        """Draw the chess board."""
        for row in range(8):
            for col in range(8):
                # Square color
                is_light = (row + col) % 2 == 0
                color = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
                
                # Square position
                x = 20 + col * self.SQUARE_SIZE
                y = 20 + row * self.SQUARE_SIZE
                rect = pygame.Rect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                # Get piece for this square
                board_row = 7 - row if self.flipped else row
                board_col = 7 - col if self.flipped else col
                piece = self.board[board_row][board_col]
                
                # Draw piece (if not being dragged)
                if piece != 'empty' and self.dragging != (board_row, board_col):
                    self.draw_piece(piece, x, y)
    
    def draw_ui(self):
        """Draw UI elements."""
        panel_x = self.BOARD_SIZE + 30
        
        # FEN display (moved down to avoid overlap)
        fen = self.get_full_fen()
        fen_text = self.small_font.render("Current FEN:", True, self.BLACK)
        self.screen.blit(fen_text, (panel_x, 620))  # Moved down
        
        # Split FEN into multiple lines if too long
        fen_lines = [fen[i:i+25] for i in range(0, len(fen), 25)]
        for i, line in enumerate(fen_lines):
            line_text = self.small_font.render(line, True, self.BLACK)
            self.screen.blit(line_text, (panel_x, 640 + i * 20))  # Moved down
        
        # Buttons with turn indicator next to Toggle Turn
        for i, (rect, text, _) in enumerate(self.buttons):
            pygame.draw.rect(self.screen, self.GRAY, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 2)
            text_surface = self.ui_font.render(text, True, self.BLACK)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
            
            # Show turn indicator next to Toggle Turn button
            if text == "Toggle Turn":
                turn_text = "White" if self.active_color == 'w' else "Black"
                turn_display = self.small_font.render(f"({turn_text})", True, self.BLACK)
                self.screen.blit(turn_display, (rect.right + 10, rect.centery - 8))
        
        # Piece palette
        palette_text = self.ui_font.render("Piece Palette:", True, self.BLACK)
        self.screen.blit(palette_text, (panel_x, 295))  # Moved down from 255
        
        for rect, piece in self.palette:
            color = self.WHITE if self.selected_piece == piece else self.GRAY
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 2)
            
            # Draw piece icon or text in palette
            if piece in self.piece_images:
                image = pygame.transform.scale(self.piece_images[piece], (30, 30))
                image_rect = image.get_rect(center=rect.center)
                self.screen.blit(image, image_rect)
            else:
                piece_text = self.ui_font.render(self.pieces[piece], True, self.BLACK)
                text_rect = piece_text.get_rect(center=rect.center)
                self.screen.blit(piece_text, text_rect)
        
        # Selected piece indicator
        if self.selected_piece:
            selected_text = self.small_font.render(f"Selected: {self.selected_piece}", True, self.BLACK)
            self.screen.blit(selected_text, (panel_x, 580))  # Moved down
    
    def draw_dragging_piece(self, mouse_pos):
        """Draw piece being dragged."""
        if self.dragging:
            piece = self.board[self.dragging[0]][self.dragging[1]]
            if piece != 'empty':
                if piece in self.piece_images:
                    image = self.piece_images[piece]
                    image_rect = image.get_rect(center=mouse_pos)
                    self.screen.blit(image, image_rect)
                else:
                    piece_text = self.piece_font.render(self.pieces[piece], True, self.BLACK)
                    piece_rect = piece_text.get_rect(center=mouse_pos)
                    self.screen.blit(piece_text, piece_rect)
    
    def handle_click(self, pos):
        """Handle mouse click."""
        # Check UI buttons
        for rect, text, callback in self.buttons:
            if rect.collidepoint(pos):
                callback()
                return
        
        # Check piece palette
        for rect, piece in self.palette:
            if rect.collidepoint(pos):
                self.selected_piece = piece
                return
        
        # Check board
        square = self.get_square_from_pos(pos)
        if square:
            row, col = square
            if self.selected_piece:
                # Place selected piece
                self.board[row][col] = self.selected_piece
                self.selected_piece = None
            else:
                # Start dragging
                piece = self.board[row][col]
                if piece != 'empty':
                    self.dragging = (row, col)
    
    def handle_drag_end(self, pos):
        """Handle end of drag."""
        if self.dragging:
            square = self.get_square_from_pos(pos)
            if square and square != self.dragging:
                # Move piece
                piece = self.board[self.dragging[0]][self.dragging[1]]
                self.board[self.dragging[0]][self.dragging[1]] = 'empty'
                self.board[square[0]][square[1]] = piece
            
            self.dragging = None
    
    def copy_fen(self):
        """Copy FEN to clipboard."""
        fen = self.get_full_fen()
        # Copy to clipboard (platform specific)
        try:
            import pyperclip
            pyperclip.copy(fen)
        except ImportError:
            pass  # Silently fail if pyperclip not available
    
    def load_fen(self):
        """Load FEN with input dialog."""
        self.show_fen_input_dialog()
    
    def show_fen_input_dialog(self):
        """Show FEN input dialog."""
        # Simple input dialog using console for now
        try:
            fen_input = input("Enter FEN (or press Enter for starting position): ").strip()
            if not fen_input:
                fen_input = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
            
            # Try to parse the FEN
            test_board = self.fen_to_board(fen_input)
            self.board = test_board
            
        except Exception:
            # Load starting position on error
            self.board = self.fen_to_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
            fen_input = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        
        # Parse turn from loaded FEN
        self.parse_fen_metadata(fen_input + " w KQkq - 0 1")
    
    def get_full_fen(self):
        """Get complete FEN string with metadata."""
        # board_array_to_fen already includes metadata, so extract just the position
        full_fen = board_array_to_fen(self.board)
        position = full_fen.split()[0]  # Get just the piece placement part
        # Add our own metadata
        return f"{position} {self.active_color} KQkq - 0 1"
    
    def toggle_turn(self):
        """Toggle whose turn it is."""
        self.active_color = 'b' if self.active_color == 'w' else 'w'
    
    def open_lichess(self):
        """Open current position in Lichess analysis."""
        fen = self.get_full_fen()
        # Replace spaces with underscores for URL
        fen_url = fen.replace(' ', '_')
        lichess_url = f"https://lichess.org/analysis/{fen_url}"
        
        # Open URL in default browser
        import webbrowser
        webbrowser.open(lichess_url)
    
    def flip_board(self):
        """Flip board orientation."""
        self.flipped = not self.flipped
    
    def clear_board(self):
        """Clear all pieces."""
        self.board = self.empty_board()
    
    def reset_board(self):
        """Reset to starting position."""
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.board = self.fen_to_board(start_fen)
        self.parse_fen_metadata(start_fen)
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click release
                        self.handle_drag_end(event.pos)
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board()
            self.draw_ui()
            self.draw_dragging_piece(mouse_pos)
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess Board Editor")
    parser.add_argument("--fen", help="Initial FEN position")
    parser.add_argument("--orientation", choices=["W", "B"], default="W", help="Board orientation (W=white, B=black)")
    parser.add_argument("--return-fen", action="store_true", help="Print final FEN to stdout on exit")
    args = parser.parse_args()
    
    editor = ChessEditor(args.fen)
    
    # Set board orientation
    if args.orientation == "B":
        editor.flipped = True
    
    editor.run()
    
    # Print final FEN if requested
    if args.return_fen:
        print(editor.get_full_fen())

if __name__ == "__main__":
    main()