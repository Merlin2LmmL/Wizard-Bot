use super::thread_state::SearchState;
use super::MAX_MATE_PLY;
use crate::bitboard::*;
use crate::eval::MATE_SCORE;
use crate::movegen::is_in_check;
use crate::position::*;

pub(crate) fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_SCORE - MAX_MATE_PLY {
        score + ply
    } else if score <= -(MATE_SCORE - MAX_MATE_PLY) {
        score - ply
    } else {
        score
    }
}

pub(crate) fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_SCORE - MAX_MATE_PLY {
        score - ply
    } else if score <= -(MATE_SCORE - MAX_MATE_PLY) {
        score + ply
    } else {
        score
    }
}

pub(crate) fn is_insufficient_material(pos: &Position) -> bool {
    let total_pieces: u32 = (0..2)
        .map(|c| (0..6).map(|pt| popcount(pos.pieces[c][pt])).sum::<u32>())
        .sum();
    if total_pieces == 2 {
        return true;
    }
    if total_pieces == 3 {
        for c in 0..2 {
            if popcount(pos.pieces[c][PieceType::Bishop as usize]) == 1
                || popcount(pos.pieces[c][PieceType::Knight as usize]) == 1
            {
                return true;
            }
        }
        return false;
    }
    // all non-king pieces are bishops, same color square
    let mut all_bishops = true;
    let mut bishop_count = 0i32;
    let mut color_sum = 0i32;
    for c in 0..2 {
        for pt in 0..6 {
            let bb = pos.pieces[c][pt];
            if pt == PieceType::King as usize {
                continue;
            }
            if bb != 0 {
                if pt != PieceType::Bishop as usize {
                    all_bishops = false;
                } else {
                    let mut b = bb;
                    while b != 0 {
                        let s = pop_lsb(&mut b);
                        bishop_count += 1;
                        color_sum += ((rank_of(s) + file_of(s)) % 2) as i32;
                    }
                }
            }
        }
    }
    if all_bishops && bishop_count > 0 && (color_sum == 0 || color_sum == bishop_count) {
        return true;
    }
    false
}

/// Terminal / draw score, or None if the position is not terminal/drawn
/// (i.e. caller should call evaluate()).
pub(crate) fn terminal_score(
    pos: &Position,
    state: &SearchState,
    ply: i32,
    has_legal_moves: bool,
) -> Option<i32> {
    if !has_legal_moves {
        return Some(if is_in_check(pos, pos.side_to_move) {
            -(MATE_SCORE - ply)
        } else {
            0
        });
    }
    if is_insufficient_material(pos) || pos.halfmove_clock >= 100 {
        return Some(0);
    }
    if state.is_search_repetition(pos.zobrist_key, pos.halfmove_clock) {
        return Some(0);
    }
    None
}
