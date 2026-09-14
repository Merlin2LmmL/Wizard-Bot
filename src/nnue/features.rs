// Ported from Stockfish sf_16 (68e1e9b3811e16cad014b590d7443b9063b3eb52),
// see Stockfish/src/nnue/features/half_ka_v2_hm.h and half_ka_v2_hm.cpp

use crate::nnue::position::{Color, Piece, Position, Square, BLACK, NO_PIECE, SQUARE_NB, WHITE};

/// Feature name (Stockfish/src/nnue/features/half_ka_v2_hm.h:71)
pub const NAME: &str = "HalfKAv2_hm(Friend)";

/// Hash value embedded in the evaluation file.
/// Stockfish/src/nnue/features/half_ka_v2_hm.h:74
pub const HASH_VALUE: u32 = 0x7f234cb8;

// unique number for each piece type on each square
// Stockfish/src/nnue/features/half_ka_v2_hm.h:40-54
const PS_NONE: u32 = 0;
const PS_W_PAWN: u32 = 0;
const PS_B_PAWN: u32 = 1 * SQUARE_NB as u32;
const PS_W_KNIGHT: u32 = 2 * SQUARE_NB as u32;
const PS_B_KNIGHT: u32 = 3 * SQUARE_NB as u32;
const PS_W_BISHOP: u32 = 4 * SQUARE_NB as u32;
const PS_B_BISHOP: u32 = 5 * SQUARE_NB as u32;
const PS_W_ROOK: u32 = 6 * SQUARE_NB as u32;
const PS_B_ROOK: u32 = 7 * SQUARE_NB as u32;
const PS_W_QUEEN: u32 = 8 * SQUARE_NB as u32;
const PS_B_QUEEN: u32 = 9 * SQUARE_NB as u32;
const PS_KING: u32 = 10 * SQUARE_NB as u32;
const PS_NB: u32 = 11 * SQUARE_NB as u32;

/// Stockfish/src/nnue/features/half_ka_v2_hm.h:56-63
/// PieceSquareIndex[color][piece], piece indexed 0..=15 per the Piece enum.
/// convention: W - us, B - them; viewed from other side, W and B are reversed.
const PIECE_SQUARE_INDEX: [[u32; 16]; 2] = [
    [
        PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
        PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
    ],
    [
        PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
        PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
    ],
];

// Stockfish/src/nnue/features/half_ka_v2_hm.h:80-99
// #define B(v) (v * PS_NB)
// KingBuckets[COLOR_NB][SQUARE_NB], copied verbatim (values already multiplied by PS_NB below).
const fn b(v: u32) -> i32 {
    (v * PS_NB) as i32
}

#[rustfmt::skip]
const KING_BUCKETS: [[i32; 64]; 2] = [
    [
        b(28), b(29), b(30), b(31), b(31), b(30), b(29), b(28),
        b(24), b(25), b(26), b(27), b(27), b(26), b(25), b(24),
        b(20), b(21), b(22), b(23), b(23), b(22), b(21), b(20),
        b(16), b(17), b(18), b(19), b(19), b(18), b(17), b(16),
        b(12), b(13), b(14), b(15), b(15), b(14), b(13), b(12),
        b( 8), b( 9), b(10), b(11), b(11), b(10), b( 9), b( 8),
        b( 4), b( 5), b( 6), b( 7), b( 7), b( 6), b( 5), b( 4),
        b( 0), b( 1), b( 2), b( 3), b( 3), b( 2), b( 1), b( 0),
    ],
    [
        b( 0), b( 1), b( 2), b( 3), b( 3), b( 2), b( 1), b( 0),
        b( 4), b( 5), b( 6), b( 7), b( 7), b( 6), b( 5), b( 4),
        b( 8), b( 9), b(10), b(11), b(11), b(10), b( 9), b( 8),
        b(12), b(13), b(14), b(15), b(15), b(14), b(13), b(12),
        b(16), b(17), b(18), b(19), b(19), b(18), b(17), b(16),
        b(20), b(21), b(22), b(23), b(23), b(22), b(21), b(20),
        b(24), b(25), b(26), b(27), b(27), b(26), b(25), b(24),
        b(28), b(29), b(30), b(31), b(31), b(30), b(29), b(28),
    ],
];

// Orient a square according to perspective (rotates by 180 for black).
// Stockfish/src/nnue/features/half_ka_v2_hm.h:102-119
// SQ_A1=0, SQ_H1=7, SQ_A8=56, SQ_H8=63 (see Stockfish/src/types.h:233-246).
const SQ_H1: i32 = 7;
const SQ_A1: i32 = 0;
const SQ_H8: i32 = 63;
const SQ_A8: i32 = 56;

#[rustfmt::skip]
const ORIENT_TBL: [[i32; 64]; 2] = [
    [
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
        SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    ],
    [
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
        SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_A8, SQ_A8, SQ_A8, SQ_A8,
    ],
];

/// Number of feature dimensions.
/// Stockfish/src/nnue/features/half_ka_v2_hm.h:77-78:
///   Dimensions = SQUARE_NB * PS_NB / 2
pub const DIMENSIONS: u32 = SQUARE_NB as u32 * PS_NB / 2;

/// Maximum number of simultaneously active features.
/// Stockfish/src/nnue/features/half_ka_v2_hm.h:122
pub const MAX_ACTIVE_DIMENSIONS: usize = 32;

/// Fixed-capacity, stack-allocated replacement for the `Vec<u32>` this used
/// to collect active feature indices into.
///
/// PERF FIX: `append_active_indices` runs twice per `evaluate()` call (once
/// per perspective) and `evaluate()` itself is called from nearly every
/// search node (every static eval in negamax.rs's reverse-futility/futility
/// margins, every quiescence stand-pat). With `MAX_ACTIVE_DIMENSIONS` fixed
/// at compile time, there is no reason this needs a heap allocation at
/// all -- the old `Vec::with_capacity(32)` in accum.rs still pays for one
/// per call regardless of the pre-reserved capacity (allocation, not
/// reallocation, was always the cost here). Under `parallel-search`,
/// multiple root-move threads were also all hitting the global allocator
/// concurrently for this same tiny, short-lived buffer on every node. A
/// plain `[u32; 32]` + length counter removes this entirely; this is purely
/// a representation change, the values pushed and their order are
/// identical to before.
pub struct ActiveIndices {
    data: [u32; MAX_ACTIVE_DIMENSIONS],
    len: usize,
}

impl ActiveIndices {
    #[inline]
    pub fn new() -> Self {
        ActiveIndices { data: [0; MAX_ACTIVE_DIMENSIONS], len: 0 }
    }

    #[inline]
    fn push(&mut self, v: u32) {
        debug_assert!(
            self.len < MAX_ACTIVE_DIMENSIONS,
            "more than MAX_ACTIVE_DIMENSIONS active features"
        );
        self.data[self.len] = v;
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.data[..self.len]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Index of a feature for a given king position and another piece on some square.
/// Stockfish/src/nnue/features/half_ka_v2_hm.cpp:28-31
///
/// `pub(crate)`, not private: this used to only ever be called from
/// `append_active_indices` below (looping over every occupied square for a
/// full refresh). `Position::toggle_piece_feature` (in `crate::position`)
/// now also calls it directly, once per piece event during
/// make_move/unmake_move, to compute the single feature index that needs
/// an incremental += or -= instead of rebuilding the whole active set.
#[inline]
pub(crate) fn make_index(perspective: Color, s: Square, pc: Piece, ksq: Square) -> u32 {
    let oriented = (s as i32) ^ ORIENT_TBL[perspective as usize][ksq as usize];
    let piece_sq_index = PIECE_SQUARE_INDEX[perspective as usize][pc as usize] as i32;
    let king_bucket = KING_BUCKETS[perspective as usize][ksq as usize];
    (oriented + piece_sq_index + king_bucket) as u32
}

/// Get a list of indices for active features.
/// Stockfish/src/nnue/features/half_ka_v2_hm.cpp:34-46
pub fn append_active_indices(pos: &Position, perspective: Color, active: &mut ActiveIndices) {
    let ksq = pos.king_square(perspective);
    for s in 0..64 {
        if pos.piece_on(s as u32) != NO_PIECE {
            let pc = pos.piece_on(s as u32);
            active.push(make_index(perspective, s as u32, pc, ksq));
        }
    }
    debug_assert!(active.len() <= MAX_ACTIVE_DIMENSIONS);
}

// Silence "unused" warnings for the exported color constants re-exported for callers.
#[allow(dead_code)]
const _USE_COLORS: (Color, Color) = (WHITE, BLACK);
