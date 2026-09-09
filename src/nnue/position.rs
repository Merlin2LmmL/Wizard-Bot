// Minimal position representation for driving NNUE evaluation.
//
// Square and Piece encodings are copied verbatim (as integer conventions,
// not literal tables) from Stockfish <sf_16, commit 68e1e9b3811e16cad014b590d7443b9063b3eb52>,
// see Stockfish/src/types.h lines 136-246:
//   enum Color { WHITE, BLACK, COLOR_NB = 2 };
//   enum PieceType { NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, ... PIECE_TYPE_NB = 8 };
//   enum Piece { NO_PIECE, W_PAWN = PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
//                B_PAWN = PAWN + 8, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, PIECE_NB = 16 };
//   enum Square : int { SQ_A1, SQ_B1, ..., SQ_H8, SQUARE_NB = 64 };
// i.e. Square = file + 8*rank (file a..h = 0..7, rank 1..8 = 0..7),
// Piece = NO_PIECE=0, W_PAWN=1..W_KING=6, B_PAWN=9..B_KING=14.

pub type Square = u32; // 0..=63, SQ_A1 == 0, SQ_H8 == 63
pub type Piece = u32; // 0..=15 per the Piece enum above
pub type Color = u32; // 0 = WHITE, 1 = BLACK

pub const WHITE: Color = 0;
pub const BLACK: Color = 1;

pub const NO_PIECE: Piece = 0;
pub const W_PAWN: Piece = 1;
pub const W_KNIGHT: Piece = 2;
pub const W_BISHOP: Piece = 3;
pub const W_ROOK: Piece = 4;
pub const W_QUEEN: Piece = 5;
pub const W_KING: Piece = 6;
pub const B_PAWN: Piece = 9;
pub const B_KNIGHT: Piece = 10;
pub const B_BISHOP: Piece = 11;
pub const B_ROOK: Piece = 12;
pub const B_QUEEN: Piece = 13;
pub const B_KING: Piece = 14;

pub const SQUARE_NB: usize = 64;

#[inline]
pub fn make_square(file: u32, rank: u32) -> Square {
    file + 8 * rank
}

/// A minimal board representation: piece-on-square array, side to move,
/// and cached king squares (both colors) -- exactly what the NNUE feature
/// transformer needs (Position::square<KING>(c), Position::pieces(),
/// Position::piece_on(s), Position::side_to_move()).
#[derive(Clone, Debug)]
pub struct Position {
    pub board: [Piece; SQUARE_NB],
    pub side_to_move: Color,
    pub king_square: [Square; 2], // indexed by Color
}

impl Position {
    pub fn piece_on(&self, s: Square) -> Piece {
        self.board[s as usize]
    }

    pub fn king_square(&self, c: Color) -> Square {
        self.king_square[c as usize]
    }

    /// Iterator over all occupied squares (any piece), analogous to
    /// Stockfish's `pos.pieces()` bitboard iterated with pop_lsb(). Order
    /// does not affect the result (accumulator sum is order independent),
    /// but we iterate in increasing square order for determinism.
    pub fn occupied_squares(&self) -> impl Iterator<Item = Square> + '_ {
        (0..SQUARE_NB as Square).filter(move |&s| self.board[s as usize] != NO_PIECE)
    }

    pub fn count_all_pieces(&self) -> u32 {
        self.board.iter().filter(|&&p| p != NO_PIECE).count() as u32
    }

    /// Parse a FEN string (board + side-to-move fields; castling/ep/clocks
    /// are accepted but not needed for NNUE evaluation).
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(format!("FEN needs at least board + side-to-move: {fen}"));
        }
        let board_field = fields[0];
        let stm_field = fields[1];

        let mut board = [NO_PIECE; SQUARE_NB];
        let ranks: Vec<&str> = board_field.split('/').collect();
        if ranks.len() != 8 {
            return Err(format!("FEN board must have 8 ranks: {fen}"));
        }

        // FEN ranks are listed from rank 8 down to rank 1.
        for (rank_from_top, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - rank_from_top as u32; // 0-indexed rank (0 = rank1)
            let mut file: u32 = 0;
            for ch in rank_str.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    file += empty;
                } else {
                    if file >= 8 {
                        return Err(format!("FEN rank overflow: {fen}"));
                    }
                    let piece = piece_from_char(ch)
                        .ok_or_else(|| format!("bad piece char '{ch}' in FEN: {fen}"))?;
                    let sq = make_square(file, rank);
                    board[sq as usize] = piece;
                    file += 1;
                }
            }
            if file != 8 {
                return Err(format!("FEN rank does not sum to 8 files: {fen}"));
            }
        }

        let side_to_move = match stm_field {
            "w" => WHITE,
            "b" => BLACK,
            other => return Err(format!("bad side-to-move field '{other}' in FEN: {fen}")),
        };

        let mut king_square = [Square::MAX; 2];
        for s in 0..SQUARE_NB as Square {
            match board[s as usize] {
                W_KING => king_square[WHITE as usize] = s,
                B_KING => king_square[BLACK as usize] = s,
                _ => {}
            }
        }
        if king_square[0] == Square::MAX || king_square[1] == Square::MAX {
            return Err(format!("FEN missing a king: {fen}"));
        }

        Ok(Position {
            board,
            side_to_move,
            king_square,
        })
    }
}

fn piece_from_char(ch: char) -> Option<Piece> {
    Some(match ch {
        'P' => W_PAWN,
        'N' => W_KNIGHT,
        'B' => W_BISHOP,
        'R' => W_ROOK,
        'Q' => W_QUEEN,
        'K' => W_KING,
        'p' => B_PAWN,
        'n' => B_KNIGHT,
        'b' => B_BISHOP,
        'r' => B_ROOK,
        'q' => B_QUEEN,
        'k' => B_KING,
        _ => return None,
    })
}
