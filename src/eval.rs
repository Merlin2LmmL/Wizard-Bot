use crate::bitboard::*;
use crate::position::*;
use crate::nnue::inference::{NnueEvaluator};

pub const MATE_SCORE: i32 = 100_000;

type Pst = [[i32; 8]; 8];

const PAWN_MG: Pst = [
    [0,0,0,0,0,0,0,0],
    [5,10,10,-20,-20,10,10,5],
    [5,-5,-10,0,0,-10,-5,5],
    [0,0,0,20,20,0,0,0],
    [5,5,10,25,25,10,5,5],
    [10,10,20,30,30,20,10,10],
    [50,50,50,50,50,50,50,50],
    [0,0,0,0,0,0,0,0],
];
const PAWN_EG: Pst = [
    [0,0,0,0,0,0,0,0],
    [10,10,10,10,10,10,10,10],
    [15,15,20,20,20,20,15,15],
    [25,25,30,30,30,30,25,25],
    [40,40,45,45,45,45,40,40],
    [60,60,65,65,65,65,60,60],
    [80,80,85,85,85,85,80,80],
    [0,0,0,0,0,0,0,0],
];
const KNIGHT_MG: Pst = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,0,0,0,0,-20,-40],
    [-30,0,10,15,15,10,0,-30],
    [-30,5,15,20,20,15,5,-30],
    [-30,0,15,20,20,15,0,-30],
    [-30,5,10,15,15,10,5,-30],
    [-40,-20,0,5,5,0,-20,-40],
    [-50,-35,-30,-30,-30,-30,-35,-50],
];
const KNIGHT_EG: Pst = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,0,5,5,0,-20,-40],
    [-30,0,15,20,20,15,0,-30],
    [-30,5,20,25,25,20,5,-30],
    [-30,0,20,25,25,20,0,-30],
    [-30,5,15,20,20,15,5,-30],
    [-40,-20,0,5,5,0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50],
];
const BISHOP_MG: Pst = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,5,0,0,0,0,5,-10],
    [-10,10,10,10,10,10,10,-10],
    [-10,0,10,10,10,10,0,-10],
    [-10,5,5,10,10,5,5,-10],
    [-10,0,5,10,10,5,0,-10],
    [-10,0,0,0,0,0,0,-10],
    [-20,-10,-15,-10,-10,-15,-10,-20],
];
const BISHOP_EG: Pst = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,0,0,0,0,0,0,-10],
    [-10,0,10,10,10,10,0,-10],
    [-10,5,10,10,10,10,5,-10],
    [-10,0,10,10,10,10,0,-10],
    [-10,0,10,10,10,10,0,-10],
    [-10,0,0,0,0,0,0,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20],
];
const ROOK_MG: Pst = [
    [0,0,0,5,5,0,0,0],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [5,10,10,10,10,10,10,5],
    [0,0,0,5,5,0,0,0],
];
const ROOK_EG: Pst = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [5,5,5,5,5,5,5,5],
    [0,0,0,0,0,0,0,0],
];
const QUEEN_MG: Pst = [
    [-20,-10,-10,-5,-5,-10,-10,-20],
    [-10,0,5,0,0,0,0,-10],
    [-10,5,5,5,5,5,0,-10],
    [0,0,5,5,5,5,0,-5],
    [-5,0,5,5,5,5,0,-5],
    [-10,0,5,5,5,5,0,-10],
    [-10,0,0,0,0,0,0,-10],
    [-20,-10,-10,-5,-5,-10,-10,-20],
];
const QUEEN_EG: Pst = [
    [-20,-10,-10,-5,-5,-10,-10,-20],
    [-10,0,0,0,0,0,0,-10],
    [-10,0,5,5,5,5,0,-10],
    [-5,0,5,5,5,5,0,-5],
    [-5,0,5,5,5,5,0,-5],
    [-10,0,5,5,5,5,0,-10],
    [-10,0,0,0,0,0,0,-10],
    [-20,-10,-10,-5,-5,-10,-10,-20],
];
const KING_MG: Pst = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20,20,0,0,0,0,20,20],
    [20,30,10,0,0,10,30,20],
];
const KING_EG: Pst = [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,0,0,-10,-20,-30],
    [-30,-10,20,30,30,20,-10,-30],
    [-30,-10,30,40,40,30,-10,-30],
    [-30,-10,30,40,40,30,-10,-30],
    [-30,-10,20,30,30,20,-10,-30],
    [-30,-30,0,0,0,0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50],
];

#[inline]
fn pst_lookup(table: &Pst, color: Color, rank: u8, file: u8) -> i32 {
    let idx = if color == Color::White { 7 - rank } else { rank };
    table[idx as usize][file as usize]
}

const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;


// Adapter: crate::position::Position -> nnue::position::Position
//
// pub(crate): still used here by transform()/evaluate() indirectly via
// NnueEvaluator::evaluate(), but also now called from
// crate::position::Position::refresh_dirty_nnue_perspectives() /
// refresh_all_nnue_accumulators(), which need to build a fresh
// nnue::position::Position whenever a king move forces a full accumulator
// refresh for that perspective.
pub(crate) fn to_nnue(pos: &crate::position::Position) -> crate::nnue::position::Position {
    let mut board = [crate::nnue::position::NO_PIECE; 64];
    let mut king_square = [0u32; 2];
    for s in 0..64u8 {
        if let Some((c, pt)) = pos.piece_at(s) {
            let pc = match (c.idx(), pt as usize) {
                (0, 0) => crate::nnue::position::W_PAWN,
                (0, 1) => crate::nnue::position::W_KNIGHT,
                (0, 2) => crate::nnue::position::W_BISHOP,
                (0, 3) => crate::nnue::position::W_ROOK,
                (0, 4) => crate::nnue::position::W_QUEEN,
                (0, 5) => crate::nnue::position::W_KING,
                (1, 0) => crate::nnue::position::B_PAWN,
                (1, 1) => crate::nnue::position::B_KNIGHT,
                (1, 2) => crate::nnue::position::B_BISHOP,
                (1, 3) => crate::nnue::position::B_ROOK,
                (1, 4) => crate::nnue::position::B_QUEEN,
                (1, 5) => crate::nnue::position::B_KING,
                _ => crate::nnue::position::NO_PIECE,
            };
            board[s as usize] = pc;
            if pt == crate::position::PieceType::King {
                king_square[c.idx()] = s as u32;
            }
        }
    }
    crate::nnue::position::Position {
        board,
        side_to_move: if pos.side_to_move == crate::position::Color::White { 0 } else { 1 },
        king_square,
    }
}


// wasm32 has no filesystem, so the .nnue file is embedded directly into the
// binary at compile time. Native builds still read it from disk, matching
// however you actually run the UCI binary. Adjust the relative path below
// if the .nnue file does not sit at the crate root alongside Cargo.toml.
#[cfg(target_arch = "wasm32")]
static NNUE_BYTES: &[u8] = include_bytes!("../nn-5af11540bbfe.nnue");

fn load_nnue() -> Result<NnueEvaluator, String> {
    #[cfg(target_arch = "wasm32")]
    {
        NnueEvaluator::load_from_bytes(NNUE_BYTES)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        NnueEvaluator::load("nn-5af11540bbfe.nnue")
    }
}

/// Loads (once) and caches the NNUE evaluator. We store the full `Result`,
/// not just an `Option`, so the real error message survives for reporting
/// over UCI instead of being silently discarded.
fn nnue_result() -> &'static Result<NnueEvaluator, String> {
    static EVAL: std::sync::OnceLock<Result<NnueEvaluator, String>> = std::sync::OnceLock::new();
    EVAL.get_or_init(load_nnue)
}

/// pub(crate): crate::position::Position now also calls this directly --
/// both to reach `.feature_transformer` for incremental accumulator
/// updates (`toggle_piece_feature`) and full refreshes on king moves
/// (`refresh_dirty_nnue_perspectives`), and to decide whether NNUE is even
/// loaded (if not, accumulator bookkeeping is skipped entirely and
/// classical_evaluate() is used, same as before).
pub(crate) fn nnue() -> Option<&'static NnueEvaluator> {
    nnue_result().as_ref().ok()
}

/// Forces the NNUE network to load (if it hasn't already) and reports
/// whether it succeeded. Cheap to call again afterwards -- the underlying
/// `OnceLock` only ever loads once. Intended to be called once at startup
/// (e.g. from wasm `init()`) so load failures can be reported immediately
/// over UCI instead of surfacing silently on the first `evaluate()` call.
pub fn warm_up_nnue() -> Result<(), String> {
    match nnue_result() {
        Ok(_) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

/// Simple material + piece-square-table evaluation, phase-blended between
/// the middlegame and endgame tables. Used as a fallback whenever the NNUE
/// network isn't available (failed to load, hash mismatch, etc.) so the
/// engine can keep playing -- gracelessly, but without crashing -- instead
/// of having no evaluation at all.
const PIECE_VALUE_MG: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const PIECE_VALUE_EG: [i32; 6] = [94, 281, 297, 512, 936, 0];

fn classical_evaluate(pos: &Position) -> i32 {
    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut phase = 0i32;

    for s in 0..64u8 {
        if let Some((color, pt)) = pos.piece_at(s) {
            let idx = pt as usize;
            let r = rank_of(s);
            let f = file_of(s);
            let (pst_mg, pst_eg): (&Pst, &Pst) = match pt {
                PieceType::Pawn => (&PAWN_MG, &PAWN_EG),
                PieceType::Knight => (&KNIGHT_MG, &KNIGHT_EG),
                PieceType::Bishop => (&BISHOP_MG, &BISHOP_EG),
                PieceType::Rook => (&ROOK_MG, &ROOK_EG),
                PieceType::Queen => (&QUEEN_MG, &QUEEN_EG),
                PieceType::King => (&KING_MG, &KING_EG),
                // Sentinel/empty-square variant -- piece_at() returning
                // Some(...) with this shouldn't happen in practice, but
                // skip it defensively rather than assume that invariant.
                PieceType::None => continue,
            };
            let c = color.idx();
            mg[c] += PIECE_VALUE_MG[idx] + pst_lookup(pst_mg, color, r, f);
            eg[c] += PIECE_VALUE_EG[idx] + pst_lookup(pst_eg, color, r, f);
            phase += PHASE_WEIGHT[idx];
        }
    }

    let phase = phase.min(MAX_PHASE);
    let stm = pos.side_to_move.idx();
    let opp = 1 - stm;
    let mg_score = mg[stm] - mg[opp];
    let eg_score = eg[stm] - eg[opp];
    (mg_score * phase + eg_score * (MAX_PHASE - phase)) / MAX_PHASE
}

/// PERF: previously built a fresh `nnue::position::Position` and ran a full
/// accumulator refresh (both perspectives) on every single call -- the
/// call site the incremental-accumulator work targeted. Now reads
/// `pos.nnue_accum[WHITE]` / `pos.nnue_accum[BLACK]`, which
/// `crate::position::Position` keeps current incrementally across
/// make_move/unmake_move (full refresh only on king moves; see
/// `Position::toggle_piece_feature`). No behavior change -- same raw score,
/// same `* 100 / 328` centipawn scaling -- purely how the accumulators get
/// there.
pub fn evaluate(pos: &Position, mobility_moves: Option<&MoveList>) -> i32 {
    let _ = mobility_moves; // reserved for future mobility terms
    match nnue() {
        Some(evaluator) => {
            let stm: u32 = if pos.side_to_move == Color::White { 0 } else { 1 };
            let piece_count = pos.occ_all.count_ones();
            let raw = evaluator.evaluate_with_accumulators(
                &pos.nnue_accum[0],
                &pos.nnue_accum[1],
                stm,
                piece_count,
            );
            raw * 100 / 328
        }
        None => classical_evaluate(pos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn startpos_eval_is_near_zero() {
        let pos = Position::startpos();
        let mut list = MoveList::new();
        crate::movegen::generate_legal_moves(&pos, &mut list);
        let e = evaluate(&pos, Some(&list));
        assert!(e.abs() < 100, "startpos eval too far from 0: {}", e);
    }
}
