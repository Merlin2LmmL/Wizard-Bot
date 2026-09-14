use crate::bitboard::*;
use crate::zobrist;
use crate::nnue;
use crate::nnue::accum::Accumulator;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub fn opponent(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    #[inline(always)]
    pub fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
    None = 6,
}

pub const PIECE_TYPES: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

#[inline(always)]
pub fn piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King => 0,
        PieceType::None => 0,
    }
}

// Castling rights bits
pub const CR_WK: u8 = 1 << 0;
pub const CR_WQ: u8 = 1 << 1;
pub const CR_BK: u8 = 1 << 2;
pub const CR_BQ: u8 = 1 << 3;

// ---- Move encoding ----
// bits [0..6)=from, [6..12)=to, [12..16)=flag, [16..19)=piece moved, [19..22)=captured piece
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Move(pub u32);

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MoveFlag {
    Quiet = 0,
    DoublePawnPush = 1,
    KingCastle = 2,
    QueenCastle = 3,
    Capture = 4,
    EpCapture = 5,
    PromoN = 6,
    PromoB = 7,
    PromoR = 8,
    PromoQ = 9,
    PromoCaptureN = 10,
    PromoCaptureB = 11,
    PromoCaptureR = 12,
    PromoCaptureQ = 13,
}

impl MoveFlag {
    #[inline(always)]
    pub fn from_u8(v: u8) -> MoveFlag {
        unsafe { std::mem::transmute(v) }
    }
    #[inline(always)]
    pub fn is_capture(self) -> bool {
        matches!(
            self,
            MoveFlag::Capture
                | MoveFlag::EpCapture
                | MoveFlag::PromoCaptureN
                | MoveFlag::PromoCaptureB
                | MoveFlag::PromoCaptureR
                | MoveFlag::PromoCaptureQ
        )
    }
    #[inline(always)]
    pub fn is_promotion(self) -> bool {
        matches!(
            self,
            MoveFlag::PromoN
                | MoveFlag::PromoB
                | MoveFlag::PromoR
                | MoveFlag::PromoQ
                | MoveFlag::PromoCaptureN
                | MoveFlag::PromoCaptureB
                | MoveFlag::PromoCaptureR
                | MoveFlag::PromoCaptureQ
        )
    }
    #[inline(always)]
    pub fn promo_piece(self) -> PieceType {
        match self {
            MoveFlag::PromoN | MoveFlag::PromoCaptureN => PieceType::Knight,
            MoveFlag::PromoB | MoveFlag::PromoCaptureB => PieceType::Bishop,
            MoveFlag::PromoR | MoveFlag::PromoCaptureR => PieceType::Rook,
            MoveFlag::PromoQ | MoveFlag::PromoCaptureQ => PieceType::Queen,
            _ => PieceType::None,
        }
    }
}

impl Move {
    pub const NULL: Move = Move(0xFFFF_FFFF);

    #[inline(always)]
    pub fn new(from: u8, to: u8, flag: MoveFlag, piece: PieceType, captured: PieceType) -> Move {
        Move(
            (from as u32)
                | ((to as u32) << 6)
                | ((flag as u32) << 12)
                | ((piece as u32) << 16)
                | ((captured as u32) << 19),
        )
    }
    #[inline(always)]
    pub fn from_sq(self) -> u8 {
        (self.0 & 0x3F) as u8
    }
    #[inline(always)]
    pub fn to_sq(self) -> u8 {
        ((self.0 >> 6) & 0x3F) as u8
    }
    #[inline(always)]
    pub fn flag(self) -> MoveFlag {
        MoveFlag::from_u8(((self.0 >> 12) & 0xF) as u8)
    }
    #[inline(always)]
    pub fn piece(self) -> PieceType {
        unsafe { std::mem::transmute(((self.0 >> 16) & 0x7) as u8) }
    }
    #[inline(always)]
    pub fn captured(self) -> PieceType {
        unsafe { std::mem::transmute(((self.0 >> 19) & 0x7) as u8) }
    }
    #[inline(always)]
    pub fn is_null(self) -> bool {
        self.0 == 0xFFFF_FFFF
    }

    pub fn to_uci(self) -> String {
        let f = self.from_sq();
        let t = self.to_sq();
        let mut s = String::with_capacity(5);
        s.push((b'a' + file_of(f)) as char);
        s.push((b'1' + rank_of(f)) as char);
        s.push((b'a' + file_of(t)) as char);
        s.push((b'1' + rank_of(t)) as char);
        if self.flag().is_promotion() {
            s.push(match self.flag().promo_piece() {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => 'q',
            });
        }
        s
    }
}

pub const MAX_MOVES: usize = 218;

pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub count: usize,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList {
            moves: [Move::NULL; MAX_MOVES],
            count: 0,
        }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        self.moves[self.count] = m;
        self.count += 1;
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.count]
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        &mut self.moves[..self.count]
    }
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct Undo {
    pub captured: PieceType,
    pub captured_color: Color,
    pub prev_castling: u8,
    pub prev_ep: Option<u8>,
    pub prev_halfmove: u16,
    pub prev_zobrist: u64,
}

#[derive(Clone)]
pub struct Position {
    /// [color][piece_type] bitboards
    pub pieces: [[Bitboard; 6]; 2],
    pub occ: [Bitboard; 2],
    pub occ_all: Bitboard,
    pub mailbox: [Option<(Color, PieceType)>; 64],
    pub side_to_move: Color,
    pub castling_rights: u8,
    pub ep_square: Option<u8>,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
    pub king_square: [u8; 2],
    pub zobrist_key: u64,
    pub undo_stack: Vec<Undo>,
    /// Persisted NNUE feature-transformer accumulators, one per absolute
    /// perspective: `nnue_accum[0]` = WHITE's perspective, `nnue_accum[1]`
    /// = BLACK's (matching `crate::nnue::position::{WHITE, BLACK}` /
    /// `Color::idx()` numbering). Kept incrementally up to date across
    /// make_move/unmake_move by `toggle_piece_feature`; only ever fully
    /// recomputed when that perspective's own king moves (see
    /// `nnue_dirty` below) or at construction.
    ///
    /// Meaningless / stale whenever NNUE isn't loaded (`crate::eval::nnue()`
    /// returns `None`) -- nothing reads these fields in that case, since
    /// `crate::eval::evaluate()` falls back to `classical_evaluate()`.
    pub nnue_accum: [Accumulator; 2],
    /// Per-perspective "needs a full refresh before next read" flag.
    /// HalfKAv2_hm features are king-relative, so when a perspective's own
    /// king moves, every one of that perspective's active features changes
    /// orientation/king-bucket at once -- not just the king's own feature.
    /// Rather than trying to incrementally re-derive every other piece's
    /// feature index under the new king square, that perspective is simply
    /// marked dirty and `refresh_dirty_nnue_perspectives()` recomputes it
    /// from scratch via `compute_accumulator()` once the move is fully
    /// applied. The *other* perspective (whose own king didn't move) still
    /// updates incrementally as normal for the same move.
    pub nnue_dirty: [bool; 2],
}

impl Position {
    pub fn startpos() -> Position {
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err("FEN needs at least 4 fields".to_string());
        }
        let board_str = parts[0];
        let turn_str = parts[1];
        let castle_str = parts[2];
        let ep_str = parts[3];
        let halfmove = parts.get(4).and_then(|s| s.parse::<u16>().ok()).unwrap_or(0);
        let fullmove = parts.get(5).and_then(|s| s.parse::<u16>().ok()).unwrap_or(1);

        let mut pieces = [[0u64; 6]; 2];
        let mut mailbox: [Option<(Color, PieceType)>; 64] = [None; 64];

        let ranks: Vec<&str> = board_str.split('/').collect();
        if ranks.len() != 8 {
            return Err("FEN board must have 8 ranks".to_string());
        }
        // FEN ranks go from rank8 down to rank1; our rank index 0 = rank1
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file = 0u8;
            for c in rank_str.chars() {
                if let Some(d) = c.to_digit(10) {
                    file += d as u8;
                } else {
                    let color = if c.is_uppercase() { Color::White } else { Color::Black };
                    let pt = match c.to_ascii_lowercase() {
                        'p' => PieceType::Pawn,
                        'n' => PieceType::Knight,
                        'b' => PieceType::Bishop,
                        'r' => PieceType::Rook,
                        'q' => PieceType::Queen,
                        'k' => PieceType::King,
                        _ => return Err(format!("bad piece char {}", c)),
                    };
                    let s = sq(rank, file);
                    pieces[color.idx()][pt as usize] |= bit(s);
                    mailbox[s as usize] = Some((color, pt));
                    file += 1;
                }
            }
        }

        let side_to_move = if turn_str == "w" { Color::White } else { Color::Black };

        let mut castling_rights = 0u8;
        if castle_str.contains('K') {
            castling_rights |= CR_WK;
        }
        if castle_str.contains('Q') {
            castling_rights |= CR_WQ;
        }
        if castle_str.contains('k') {
            castling_rights |= CR_BK;
        }
        if castle_str.contains('q') {
            castling_rights |= CR_BQ;
        }

        let ep_square = if ep_str == "-" {
            None
        } else {
            let bytes = ep_str.as_bytes();
            if bytes.len() != 2 {
                None
            } else {
                let file = bytes[0] - b'a';
                let rank = bytes[1] - b'1';
                Some(sq(rank, file))
            }
        };

        let mut occ = [0u64; 2];
        for c in 0..2 {
            for pt in 0..6 {
                occ[c] |= pieces[c][pt];
            }
        }
        let occ_all = occ[0] | occ[1];

        let king_square = [
            (pieces[0][PieceType::King as usize]).trailing_zeros() as u8,
            (pieces[1][PieceType::King as usize]).trailing_zeros() as u8,
        ];

        let mut pos = Position {
            pieces,
            occ,
            occ_all,
            mailbox,
            side_to_move,
            castling_rights,
            ep_square,
            halfmove_clock: halfmove,
            fullmove_number: fullmove,
            king_square,
            zobrist_key: 0,
            undo_stack: Vec::with_capacity(256),
            nnue_accum: [Accumulator::new_zeroed(), Accumulator::new_zeroed()],
            nnue_dirty: [false, false],
        };
        pos.zobrist_key = zobrist::compute_full(&pos);
        pos.refresh_all_nnue_accumulators();
        Ok(pos)
    }

    /// First four fields of the FEN, used as the opening-book key.
    pub fn book_key(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty_run = 0;
            for file in 0..8 {
                let sqi = sq(rank, file);
                match self.mailbox[sqi as usize] {
                    None => empty_run += 1,
                    Some((color, pt)) => {
                        if empty_run > 0 {
                            s.push((b'0' + empty_run) as char);
                            empty_run = 0;
                        }
                        let c = match pt {
                            PieceType::Pawn => 'p',
                            PieceType::Knight => 'n',
                            PieceType::Bishop => 'b',
                            PieceType::Rook => 'r',
                            PieceType::Queen => 'q',
                            PieceType::King => 'k',
                            PieceType::None => '?',
                        };
                        s.push(if color == Color::White {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        });
                    }
                }
            }
            if empty_run > 0 {
                s.push((b'0' + empty_run) as char);
            }
            if rank != 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.side_to_move == Color::White { 'w' } else { 'b' });
        s.push(' ');
        let mut any = false;
        if self.castling_rights & CR_WK != 0 {
            s.push('K');
            any = true;
        }
        if self.castling_rights & CR_WQ != 0 {
            s.push('Q');
            any = true;
        }
        if self.castling_rights & CR_BK != 0 {
            s.push('k');
            any = true;
        }
        if self.castling_rights & CR_BQ != 0 {
            s.push('q');
            any = true;
        }
        if !any {
            s.push('-');
        }
        s.push(' ');
        match self.ep_square {
            None => s.push('-'),
            Some(epsq) => {
                s.push((b'a' + file_of(epsq)) as char);
                s.push((b'1' + rank_of(epsq)) as char);
            }
        }
        s
    }

    #[inline(always)]
    pub fn ply(&self) -> u32 {
        2 * (self.fullmove_number as u32 - 1) + if self.side_to_move == Color::Black { 1 } else { 0 }
    }

    #[inline(always)]
    fn put_piece(&mut self, color: Color, pt: PieceType, square: u8) {
        debug_assert!(pt != PieceType::None, "put_piece called with PieceType::None on {square}");
        self.pieces[color.idx()][pt as usize] |= bit(square);
        self.occ[color.idx()] |= bit(square);
        self.occ_all |= bit(square);
        self.mailbox[square as usize] = Some((color, pt));
    }

    #[inline(always)]
    fn remove_piece(&mut self, color: Color, pt: PieceType, square: u8) {
        debug_assert!(pt != PieceType::None, "remove_piece called with PieceType::None on {square}");
        self.pieces[color.idx()][pt as usize] &= !bit(square);
        self.occ[color.idx()] &= !bit(square);
        self.occ_all &= !bit(square);
        self.mailbox[square as usize] = None;
    }

    #[inline(always)]
    fn move_piece(&mut self, color: Color, pt: PieceType, from: u8, to: u8) {
        debug_assert!(pt != PieceType::None, "move_piece called with PieceType::None (from {from} to {to})");
        let mask = bit(from) | bit(to);
        self.pieces[color.idx()][pt as usize] ^= mask;
        self.occ[color.idx()] ^= mask;
        self.occ_all ^= mask;
        self.mailbox[from as usize] = None;
        self.mailbox[to as usize] = Some((color, pt));
    }

    /// crate::position::(Color, PieceType) -> crate::nnue::position::Piece.
    /// Same convention as eval.rs's `to_nnue` match table (W_x = 1..6,
    /// B_x = W_x + 8, per the doc comment on nnue/position.rs), just
    /// expressed as arithmetic instead of a 12-arm match since it's on a
    /// hot path now (called up to ~4 times per make_move/unmake_move,
    /// rather than once per board square per eval() call as before).
    #[inline(always)]
    fn nnue_piece(color: Color, pt: PieceType) -> nnue::position::Piece {
        use nnue::position as np;
        let base = match pt {
            PieceType::Pawn => np::W_PAWN,
            PieceType::Knight => np::W_KNIGHT,
            PieceType::Bishop => np::W_BISHOP,
            PieceType::Rook => np::W_ROOK,
            PieceType::Queen => np::W_QUEEN,
            PieceType::King => np::W_KING,
            PieceType::None => return np::NO_PIECE,
        };
        match color {
            Color::White => base,
            Color::Black => base + 8,
        }
    }

    /// Incrementally applies (or removes) one piece's contribution to both
    /// perspectives' NNUE accumulators, for a piece of color `color` and
    /// type `pt` arriving at (add = true) or leaving (add = false) `square`.
    ///
    /// Called once per piece event during make_move/unmake_move: capture
    /// removal, mover removal, mover placement, and (for castling) the
    /// rook's removal + placement. Four events, each touching both
    /// perspectives, covers every way a HalfKAv2_hm feature can toggle.
    ///
    /// For the perspective matching `color` when `pt == PieceType::King`,
    /// this doesn't apply an incremental delta at all -- it marks that
    /// perspective dirty (see the `nnue_dirty` field doc) so
    /// `refresh_dirty_nnue_perspectives()` does a full recompute once the
    /// move is fully applied. Both perspectives always read their *own*
    /// `king_square`, so it does not matter whether this is called before
    /// or after `self.king_square[..]` is updated for the current move --
    /// the dirty/skip path is taken before that value would ever be read
    /// for the king's own perspective, and the other perspective's king
    /// square is untouched by this move regardless.
    #[inline]
    fn toggle_piece_feature(&mut self, color: Color, pt: PieceType, square: u8, add: bool) {
        if pt == PieceType::None {
            return;
        }
        let Some(evaluator) = crate::eval::nnue() else {
            return; // NNUE not loaded -- classical_evaluate() is in use, nothing reads nnue_accum.
        };
        let ft = &evaluator.feature_transformer;
        let pc = Self::nnue_piece(color, pt);

        for perspective in [nnue::position::WHITE, nnue::position::BLACK] {
            let p_idx = perspective as usize;
            if self.nnue_dirty[p_idx] {
                continue; // already flagged for a full refresh this move; incremental work here would be wasted
            }
            if color.idx() == p_idx && pt == PieceType::King {
                self.nnue_dirty[p_idx] = true;
                continue;
            }
            let ksq = self.king_square[p_idx] as u32;
            let index = nnue::features::make_index(perspective, square as u32, pc, ksq);
            nnue::accum::apply_feature(&mut self.nnue_accum[p_idx], ft, index, add);
        }
    }

    /// Recomputes from scratch any perspective currently marked dirty
    /// (i.e. whose own king moved this ply). Must be called only after the
    /// board (mailbox, king_square, pieces/occ bitboards) is fully in its
    /// final, consistent post-move (or post-unmake) state --
    /// `compute_accumulator` reads the live board directly via
    /// `crate::eval::to_nnue`.
    ///
    /// No-op (aside from clearing the flags) if NNUE isn't loaded.
    fn refresh_dirty_nnue_perspectives(&mut self) {
        if !self.nnue_dirty[0] && !self.nnue_dirty[1] {
            return;
        }
        let Some(evaluator) = crate::eval::nnue() else {
            self.nnue_dirty = [false, false];
            return;
        };
        let ft = &evaluator.feature_transformer;
        let nnue_pos = crate::eval::to_nnue(self);
        for perspective in [nnue::position::WHITE, nnue::position::BLACK] {
            let p_idx = perspective as usize;
            if self.nnue_dirty[p_idx] {
                self.nnue_accum[p_idx] = nnue::accum::compute_accumulator(ft, &nnue_pos, perspective);
                self.nnue_dirty[p_idx] = false;
            }
        }
    }

    /// Unconditional full refresh of both perspectives, regardless of the
    /// current dirty flags. Used only at construction (`from_fen`), where
    /// there is no prior incremental state to preserve.
    fn refresh_all_nnue_accumulators(&mut self) {
        self.nnue_dirty = [true, true];
        self.refresh_dirty_nnue_perspectives();
    }

    pub fn make_move(&mut self, m: Move) {
        let us = self.side_to_move;
        let them = us.opponent();
        let from = m.from_sq();
        let to = m.to_sq();
        let flag = m.flag();
        let piece = m.piece();

        let undo = Undo {
            captured: m.captured(),
            captured_color: them,
            prev_castling: self.castling_rights,
            prev_ep: self.ep_square,
            prev_halfmove: self.halfmove_clock,
            prev_zobrist: self.zobrist_key,
        };

        // clear old ep file hash
        if let Some(epsq) = self.ep_square {
            self.zobrist_key ^= zobrist::ep_file_key(file_of(epsq));
        }
        self.ep_square = None;

        self.halfmove_clock += 1;
        if piece == PieceType::Pawn || flag.is_capture() {
            self.halfmove_clock = 0;
        }

        // Remove captured piece (handle en passant specially: captured pawn
        // is not on `to`)
        if flag == MoveFlag::EpCapture {
            let cap_sq = sq(rank_of(from), file_of(to));
            self.toggle_piece_feature(them, PieceType::Pawn, cap_sq, false);
            self.remove_piece(them, PieceType::Pawn, cap_sq);
            self.zobrist_key ^= zobrist::piece_key(them, PieceType::Pawn, cap_sq);
        } else if flag.is_capture() {
            let cap_pt = m.captured();
            debug_assert!(
                cap_pt != PieceType::None,
                "make_move: capture-flagged move with no captured piece: {m:?}"
            );
            self.toggle_piece_feature(them, cap_pt, to, false);
            self.remove_piece(them, cap_pt, to);
            self.zobrist_key ^= zobrist::piece_key(them, cap_pt, to);
        }

        // Move the piece itself (remove from `from`, then place - possibly
        // promoted - piece on `to`)
        self.toggle_piece_feature(us, piece, from, false);
        self.remove_piece(us, piece, from);
        self.zobrist_key ^= zobrist::piece_key(us, piece, from);

        let final_piece = if flag.is_promotion() { flag.promo_piece() } else { piece };
        self.toggle_piece_feature(us, final_piece, to, true);
        self.put_piece(us, final_piece, to);
        self.zobrist_key ^= zobrist::piece_key(us, final_piece, to);

        if piece == PieceType::King {
            self.king_square[us.idx()] = to;
        }

        // Castling: move the rook too
        match flag {
            MoveFlag::KingCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (sq(0, 7), sq(0, 5)),
                    Color::Black => (sq(7, 7), sq(7, 5)),
                };
                self.zobrist_key ^= zobrist::piece_key(us, PieceType::Rook, rook_from);
                self.toggle_piece_feature(us, PieceType::Rook, rook_from, false);
                self.toggle_piece_feature(us, PieceType::Rook, rook_to, true);
                self.move_piece(us, PieceType::Rook, rook_from, rook_to);
                self.zobrist_key ^= zobrist::piece_key(us, PieceType::Rook, rook_to);
            }
            MoveFlag::QueenCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (sq(0, 0), sq(0, 3)),
                    Color::Black => (sq(7, 0), sq(7, 3)),
                };
                self.zobrist_key ^= zobrist::piece_key(us, PieceType::Rook, rook_from);
                self.toggle_piece_feature(us, PieceType::Rook, rook_from, false);
                self.toggle_piece_feature(us, PieceType::Rook, rook_to, true);
                self.move_piece(us, PieceType::Rook, rook_from, rook_to);
                self.zobrist_key ^= zobrist::piece_key(us, PieceType::Rook, rook_to);
            }
            MoveFlag::DoublePawnPush => {
                let epsq = sq((rank_of(from) + rank_of(to)) / 2, file_of(from));
                self.ep_square = Some(epsq);
                self.zobrist_key ^= zobrist::ep_file_key(file_of(epsq));
            }
            _ => {}
        }

        // Update castling rights (king/rook moves or rook captures)
        let old_cr = self.castling_rights;
        self.update_castling_rights(from, to, piece, us);
        if old_cr != self.castling_rights {
            self.zobrist_key ^= zobrist::castling_key(old_cr);
            self.zobrist_key ^= zobrist::castling_key(self.castling_rights);
        }

        // Side to move flip
        self.zobrist_key ^= zobrist::side_key();
        self.side_to_move = them;

        if us == Color::Black {
            self.fullmove_number += 1;
        }

        // Board (mailbox/king_square/occupancy) is now fully in its
        // post-move state; safe to recompute any perspective flagged
        // dirty by a king move above.
        self.refresh_dirty_nnue_perspectives();

        self.undo_stack.push(undo);
    }

    fn update_castling_rights(&mut self, from: u8, to: u8, piece: PieceType, us: Color) {
        if piece == PieceType::King {
            match us {
                Color::White => self.castling_rights &= !(CR_WK | CR_WQ),
                Color::Black => self.castling_rights &= !(CR_BK | CR_BQ),
            }
        }
        // Rook moved from its home square, or rook captured on its home square
        if from == sq(0, 0) || to == sq(0, 0) {
            self.castling_rights &= !CR_WQ;
        }
        if from == sq(0, 7) || to == sq(0, 7) {
            self.castling_rights &= !CR_WK;
        }
        if from == sq(7, 0) || to == sq(7, 0) {
            self.castling_rights &= !CR_BQ;
        }
        if from == sq(7, 7) || to == sq(7, 7) {
            self.castling_rights &= !CR_BK;
        }
    }

    pub fn unmake_move(&mut self, m: Move) {
        let them = self.side_to_move;
        let us = them.opponent();
        self.side_to_move = us;

        if us == Color::Black {
            self.fullmove_number -= 1;
        }

        let from = m.from_sq();
        let to = m.to_sq();
        let flag = m.flag();
        let piece = m.piece();

        let final_piece = if flag.is_promotion() { flag.promo_piece() } else { piece };
        self.toggle_piece_feature(us, final_piece, to, false);
        self.remove_piece(us, final_piece, to);
        self.toggle_piece_feature(us, piece, from, true);
        self.put_piece(us, piece, from);

        if piece == PieceType::King {
            self.king_square[us.idx()] = from;
        }

        match flag {
            MoveFlag::KingCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (sq(0, 7), sq(0, 5)),
                    Color::Black => (sq(7, 7), sq(7, 5)),
                };
                self.toggle_piece_feature(us, PieceType::Rook, rook_to, false);
                self.toggle_piece_feature(us, PieceType::Rook, rook_from, true);
                self.move_piece(us, PieceType::Rook, rook_to, rook_from);
            }
            MoveFlag::QueenCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (sq(0, 0), sq(0, 3)),
                    Color::Black => (sq(7, 0), sq(7, 3)),
                };
                self.toggle_piece_feature(us, PieceType::Rook, rook_to, false);
                self.toggle_piece_feature(us, PieceType::Rook, rook_from, true);
                self.move_piece(us, PieceType::Rook, rook_to, rook_from);
            }
            MoveFlag::EpCapture => {
                let cap_sq = sq(rank_of(from), file_of(to));
                self.toggle_piece_feature(them, PieceType::Pawn, cap_sq, true);
                self.put_piece(them, PieceType::Pawn, cap_sq);
            }
            _ => {
                if flag.is_capture() {
                    debug_assert!(
                        m.captured() != PieceType::None,
                        "unmake_move: capture-flagged move with no captured piece: {m:?}"
                    );
                    self.toggle_piece_feature(them, m.captured(), to, true);
                    self.put_piece(them, m.captured(), to);
                }
            }
        }

        let undo = self.undo_stack.pop().expect("unmake_move: empty undo stack");
        self.castling_rights = undo.prev_castling;
        self.ep_square = undo.prev_ep;
        self.halfmove_clock = undo.prev_halfmove;
        self.zobrist_key = undo.prev_zobrist;

        // Same reasoning as make_move: board is fully back in its pre-move
        // state now, safe to recompute any perspective a king move flagged
        // dirty. Recomputing here (rather than storing/restoring a saved
        // accumulator in Undo) works because compute_accumulator is a pure
        // function of board state -- any full refresh of a given position
        // yields the same bit-for-bit accumulator regardless of how it was
        // reached (see the comment on that invariant in nnue/accum.rs), so
        // this necessarily reproduces the exact pre-move accumulator for
        // that perspective.
        self.refresh_dirty_nnue_perspectives();
    }

    pub fn make_null_move(&mut self) -> Option<u8> {
        let prev_ep = self.ep_square;
        if let Some(epsq) = prev_ep {
            self.zobrist_key ^= zobrist::ep_file_key(file_of(epsq));
        }
        self.ep_square = None;
        self.zobrist_key ^= zobrist::side_key();
        self.side_to_move = self.side_to_move.opponent();
        // No piece moves on a null move, so the board -- and therefore
        // both NNUE accumulators -- are untouched. Only side_to_move
        // flips, which evaluate() already accounts for when picking
        // perspective ordering.
        prev_ep
    }

    pub fn unmake_null_move(&mut self, prev_ep: Option<u8>) {
        self.side_to_move = self.side_to_move.opponent();
        self.zobrist_key ^= zobrist::side_key();
        if let Some(epsq) = prev_ep {
            self.zobrist_key ^= zobrist::ep_file_key(file_of(epsq));
        }
        self.ep_square = prev_ep;
    }

    #[inline]
    pub fn piece_at(&self, square: u8) -> Option<(Color, PieceType)> {
        self.mailbox[square as usize]
    }

    pub fn non_pawn_king_material(&self, color: Color) -> bool {
        let c = color.idx();
        (self.pieces[c][PieceType::Knight as usize]
            | self.pieces[c][PieceType::Bishop as usize]
            | self.pieces[c][PieceType::Rook as usize]
            | self.pieces[c][PieceType::Queen as usize])
            != 0
    }
}

#[cfg(test)]
mod nnue_accum_tests {
    use super::*;
    use crate::movegen;

    /// The round-trip test the incremental-accumulator design flagged as
    /// the thing most likely to hide a subtle sign/ordering bug: for a
    /// range of positions (including castling, en passant, promotion, and
    /// king moves specifically, since those take the full-refresh path
    /// instead of the incremental one), make_move followed by unmake_move
    /// must reproduce the pre-move accumulator bit-for-bit -- not just an
    /// eval() score that happens to match.
    ///
    /// Skips (rather than failing) if the NNUE network isn't loaded in
    /// this environment, since nnue_accum is meaningless without it.
    fn assert_round_trip(fen: &str) {
        if crate::eval::nnue().is_none() {
            return;
        }
        let mut pos = Position::from_fen(fen).expect("valid FEN");
        let mut list = MoveList::new();
        movegen::generate_legal_moves(&pos, &mut list);

        for &m in list.as_slice() {
            let before_white = pos.nnue_accum[0].accumulation;
            let before_white_psqt = pos.nnue_accum[0].psqt_accumulation;
            let before_black = pos.nnue_accum[1].accumulation;
            let before_black_psqt = pos.nnue_accum[1].psqt_accumulation;

            pos.make_move(m);
            pos.unmake_move(m);

            assert_eq!(
                pos.nnue_accum[0].accumulation, before_white,
                "WHITE accumulation mismatch after make/unmake {} on {}",
                m.to_uci(),
                fen
            );
            assert_eq!(
                pos.nnue_accum[0].psqt_accumulation, before_white_psqt,
                "WHITE psqt mismatch after make/unmake {} on {}",
                m.to_uci(),
                fen
            );
            assert_eq!(
                pos.nnue_accum[1].accumulation, before_black,
                "BLACK accumulation mismatch after make/unmake {} on {}",
                m.to_uci(),
                fen
            );
            assert_eq!(
                pos.nnue_accum[1].psqt_accumulation, before_black_psqt,
                "BLACK psqt mismatch after make/unmake {} on {}",
                m.to_uci(),
                fen
            );
        }
    }

    #[test]
    fn round_trip_startpos() {
        assert_round_trip("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    #[test]
    fn round_trip_castling_rights_available() {
        // Both sides can castle either way; also covers ordinary captures
        // and quiet moves among developed pieces.
        assert_round_trip("r3k2r/pppq1ppp/2n1bn2/2bpp3/2BPP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 4 8");
    }

    #[test]
    fn round_trip_en_passant_available() {
        assert_round_trip("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 4");
    }

    #[test]
    fn round_trip_promotion_available() {
        assert_round_trip("8/1P6/2k5/8/8/2K5/6p1/8 w - - 0 1");
    }

    #[test]
    fn round_trip_king_in_center() {
        // King moves (the full-refresh path) dominate the legal move list
        // here, for both perspectives across successive plies.
        assert_round_trip("8/8/4k3/8/4K3/8/8/8 w - - 0 1");
    }
}
