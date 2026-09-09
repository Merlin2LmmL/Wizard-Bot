use crate::bitboard::*;
use crate::zobrist;

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
        };
        pos.zobrist_key = zobrist::compute_full(&pos);
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
            self.remove_piece(them, PieceType::Pawn, cap_sq);
            self.zobrist_key ^= zobrist::piece_key(them, PieceType::Pawn, cap_sq);
        } else if flag.is_capture() {
            let cap_pt = m.captured();
            debug_assert!(
                cap_pt != PieceType::None,
                "make_move: capture-flagged move with no captured piece: {m:?}"
            );
            self.remove_piece(them, cap_pt, to);
            self.zobrist_key ^= zobrist::piece_key(them, cap_pt, to);
        }

        // Move the piece itself (remove from `from`, then place - possibly
        // promoted - piece on `to`)
        self.remove_piece(us, piece, from);
        self.zobrist_key ^= zobrist::piece_key(us, piece, from);

        let final_piece = if flag.is_promotion() { flag.promo_piece() } else { piece };
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
                self.move_piece(us, PieceType::Rook, rook_from, rook_to);
                self.zobrist_key ^= zobrist::piece_key(us, PieceType::Rook, rook_to);
            }
            MoveFlag::QueenCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (sq(0, 0), sq(0, 3)),
                    Color::Black => (sq(7, 0), sq(7, 3)),
                };
                self.zobrist_key ^= zobrist::piece_key(us, PieceType::Rook, rook_from);
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
        self.remove_piece(us, final_piece, to);
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
                self.move_piece(us, PieceType::Rook, rook_to, rook_from);
            }
            MoveFlag::QueenCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (sq(0, 0), sq(0, 3)),
                    Color::Black => (sq(7, 0), sq(7, 3)),
                };
                self.move_piece(us, PieceType::Rook, rook_to, rook_from);
            }
            MoveFlag::EpCapture => {
                let cap_sq = sq(rank_of(from), file_of(to));
                self.put_piece(them, PieceType::Pawn, cap_sq);
            }
            _ => {
                if flag.is_capture() {
                    debug_assert!(
                        m.captured() != PieceType::None,
                        "unmake_move: capture-flagged move with no captured piece: {m:?}"
                    );
                    self.put_piece(them, m.captured(), to);
                }
            }
        }

        let undo = self.undo_stack.pop().expect("unmake_move: empty undo stack");
        self.castling_rights = undo.prev_castling;
        self.ep_square = undo.prev_ep;
        self.halfmove_clock = undo.prev_halfmove;
        self.zobrist_key = undo.prev_zobrist;
    }

    pub fn make_null_move(&mut self) -> Option<u8> {
        let prev_ep = self.ep_square;
        if let Some(epsq) = prev_ep {
            self.zobrist_key ^= zobrist::ep_file_key(file_of(epsq));
        }
        self.ep_square = None;
        self.zobrist_key ^= zobrist::side_key();
        self.side_to_move = self.side_to_move.opponent();
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
