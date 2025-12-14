"""FEN string utilities."""

import numpy as np

# FEN piece mappings
FEN_TO_PIECE = {
    'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
}
PIECE_TO_FEN = {v: k for k, v in FEN_TO_PIECE.items()}

def fen_to_board_array(fen_string):
    """Convert FEN string to 8x8 piece array."""
    piece_placement = fen_string.split()[0]
    board = []
    
    for row in piece_placement.split('/'):
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['empty'] * int(char))
            else:
                board_row.append(FEN_TO_PIECE.get(char, 'empty'))
        
        # Ensure exactly 8 squares per row
        board_row = (board_row + ['empty'] * 8)[:8]
        board.append(board_row)
    
    # Ensure exactly 8 rows
    while len(board) < 8:
        board.append(['empty'] * 8)
    
    return np.array(board[:8])

def board_array_to_fen(board_array):
    """Convert 8x8 piece array to FEN string."""
    fen_rows = []
    
    for row in board_array:
        fen_row = ""
        empty_count = 0
        
        for piece in row:
            if piece == 'empty':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += PIECE_TO_FEN.get(piece, '')
        
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    return '/'.join(fen_rows) + ' w - - 0 1'