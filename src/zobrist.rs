use crate::position::{Color, PieceType, Position};

// Fixed-seed splitmix64 PRNG, evaluated at compile/const-eval time so the
// whole table is a static with zero runtime initialization cost.
const fn splitmix64_next(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = state;

    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;

    (state, z)
}

struct ZobristTable {
    // [color][piece_type][square]
    pieces: [[[u64; 64]; 6]; 2],
    side: u64,
    castling: [u64; 16],
    ep_file: [u64; 8],
}

const fn build_table() -> ZobristTable {
    let mut state: u64 = 0x5EED_C0DE_F00D_1234;

    let mut pieces = [[[0u64; 64]; 6]; 2];

    let mut c = 0;
    while c < 2 {
        let mut pt = 0;

        while pt < 6 {
            let mut s = 0;

            while s < 64 {
                let (next_state, value) = splitmix64_next(state);
                state = next_state;
                pieces[c][pt][s] = value;
                s += 1;
            }

            pt += 1;
        }

        c += 1;
    }

    let (next_state, side) = splitmix64_next(state);
    state = next_state;

    let mut castling = [0u64; 16];
    let mut i = 0;

    while i < 16 {
        let (next_state, value) = splitmix64_next(state);
        state = next_state;
        castling[i] = value;
        i += 1;
    }

    let mut ep_file = [0u64; 8];
    let mut file = 0;

    while file < 8 {
        let (next_state, value) = splitmix64_next(state);
        state = next_state;
        ep_file[file] = value;
        file += 1;
    }

    ZobristTable {
        pieces,
        side,
        castling,
        ep_file,
    }
}

static TABLE: ZobristTable = build_table();

#[inline(always)]
pub fn piece_key(color: Color, piece_type: PieceType, square: u8) -> u64 {
    TABLE.pieces[color.idx()][piece_type as usize][square as usize]
}

#[inline(always)]
pub fn side_key() -> u64 {
    TABLE.side
}

#[inline(always)]
pub fn castling_key(rights: u8) -> u64 {
    TABLE.castling[(rights & 0x0F) as usize]
}

#[inline(always)]
pub fn ep_file_key(file: u8) -> u64 {
    TABLE.ep_file[(file & 0x07) as usize]
}

/// Full from-scratch recomputation, used at position construction time and
/// for consistency assertions in tests.
pub fn compute_full(pos: &Position) -> u64 {
    let mut key = 0u64;

    for color_index in 0..2 {
        let color = if color_index == 0 {
            Color::White
        } else {
            Color::Black
        };

        for piece_index in 0..6 {
            let piece_type: PieceType =
                unsafe { std::mem::transmute(piece_index as u8) };

            let mut bitboard = pos.pieces[color_index][piece_index];

            while bitboard != 0 {
                let square = bitboard.trailing_zeros() as u8;
                bitboard &= bitboard - 1;

                key ^= piece_key(color, piece_type, square);
            }
        }
    }

    if pos.side_to_move == Color::Black {
        key ^= side_key();
    }

    key ^= castling_key(pos.castling_rights);

    if let Some(ep_square) = pos.ep_square {
        let file = crate::bitboard::file_of(ep_square);
        key ^= ep_file_key(file);
    }

    key
}
