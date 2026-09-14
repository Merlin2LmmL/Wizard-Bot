use super::helpers::terminal_score;
use super::thread_state::SearchState;
use super::QUIESCENCE_MAX_PLY;
use crate::eval::{evaluate, MATE_SCORE};
use crate::movegen::{compute_checking_squares, generate_legal_moves, gives_check, is_in_check};
use crate::position::*;

pub(crate) fn quiescence(
    pos: &mut Position,
    state: &mut SearchState,
    mut alpha: i32,
    beta: i32,
    qply: i32,
    allow_checks: bool,
    ply: i32,
) -> i32 {
    state.local.nodes += 1;
    if state.time_up() {
        return evaluate(pos, None);
    }

    let mut list = MoveList::new();
    generate_legal_moves(pos, &mut list);
    let in_check = is_in_check(pos, pos.side_to_move);

    if let Some(s) = terminal_score(pos, state, ply, list.count > 0) {
        return s;
    }

    if in_check {
        if qply <= 0 {
            return evaluate(pos, Some(&list));
        }
        let mut best = -MATE_SCORE;
        for &m in list.as_slice() {
            state.push_rep(pos.zobrist_key);
            pos.make_move(m);
            let score = -quiescence(pos, state, -beta, -alpha, qply - 1, allow_checks, ply + 1);
            pos.unmake_move(m);
            state.pop_rep();
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }
        return best;
    }

    let stand_pat = evaluate(pos, Some(&list));
    if stand_pat >= beta {
        return stand_pat;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }
    if qply <= 0 {
        return stand_pat;
    }

    let allow_checks_this_ply = allow_checks && qply > (QUIESCENCE_MAX_PLY - 2);

    // Per-node checking-squares table (see movegen.rs::CheckingSquares),
    // computed once here and reused for every quiet candidate considered
    // below instead of each gives_check() call redoing its own attack scan.
    let check_ctx = compute_checking_squares(pos, pos.side_to_move);

    // Candidate moves: captures always; promotions and (optionally) quiet
    // checks in the last 2 plies of the qsearch window.
    let mut candidates: Vec<Move> = Vec::new();
    for &m in list.as_slice() {
        if m.flag().is_capture() || m.flag().is_promotion() {
            candidates.push(m);
        } else if allow_checks_this_ply {
            // Use gives_check() with the precomputed table rather than
            // make/unmake; answers same question without touching board.
            let check_start = std::time::Instant::now();
            let move_gives_check = gives_check(pos, m, &check_ctx);
            state.local.time_in_check_detect_ns += check_start.elapsed().as_nanos() as u64;
            if move_gives_check {
                candidates.push(m);
            }
        }
    }
    // Order by SEE (best exchange first) instead of raw victim value, so a
    // "big victim" that's actually defended doesn't get tried before a
    // smaller but genuinely winning capture.
    let see_sort_start = std::time::Instant::now();
    candidates.sort_by_key(|&m| -crate::movegen::see(pos, m));
    state.local.time_in_see_ns += see_sort_start.elapsed().as_nanos() as u64;

    let mut best = stand_pat;
    for m in candidates {
        if !m.flag().is_promotion() && m.flag().is_capture() {
            // SEE-based pruning: skip captures that lose material
            // once already below alpha by more than a small margin.
            let see_start = std::time::Instant::now();
            let see_val = crate::movegen::see(pos, m);
            state.local.time_in_see_ns += see_start.elapsed().as_nanos() as u64;
            if stand_pat + see_val + 100 <= alpha {
                continue;
            }
        }
        let next_allow_checks = allow_checks_this_ply && m.flag().is_capture();
        state.push_rep(pos.zobrist_key);
        pos.make_move(m);
        let score = -quiescence(pos, state, -beta, -alpha, qply - 1, next_allow_checks, ply + 1);
        pos.unmake_move(m);
        state.pop_rep();
        if score > best {
            best = score;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}
