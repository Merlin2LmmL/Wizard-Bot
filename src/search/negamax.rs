use super::helpers::{score_from_tt, score_to_tt, terminal_score};
use super::ordering::MoveOrderer;
use super::quiescence::quiescence;
use super::thread_state::SearchState;
use super::tt::Bound;
use super::{MAX_LINE_EXTENSIONS, MAX_PLY, QUIESCENCE_MAX_PLY};
use crate::eval::{evaluate, MATE_SCORE};
use crate::movegen::{compute_checking_squares, gives_check, is_in_check};
use crate::position::*;

/// `excluded`: the move this call must pretend doesn't exist, or
/// `Move::NULL` for a normal search. Only ever non-null when this call is
/// itself the singular-extension verification search of its own parent
/// node (same `pos`, same `ply` -- no move has been made relative to the
/// caller). See the singular-extension block below for why this replaced
/// the old `state.local.excluded_move` field, which nothing ever read.
///
/// `ext`: total extensions (check extensions + singular extensions)
/// already spent along the current line, from the root move onward. Capped
/// at `MAX_LINE_EXTENSIONS` so that a chain of checks (or checks stacked
/// with singular extensions) can't keep handing out un-shrunk depth
/// indefinitely -- see the comment on `MAX_LINE_EXTENSIONS` in mod.rs.
pub(crate) fn negamax(
    pos: &mut Position,
    state: &mut SearchState,
    mut depth: i32,
    mut alpha: i32,
    mut beta: i32,
    ply: i32,
    prev_move: Move,
    excluded: Move,
    mut ext: i32,
) -> i32 {
    state.local.nodes += 1;
    if state.time_up() {
        return evaluate(pos, None);
    }

    let orig_alpha = alpha;

    let mut list = MoveList::new();
    crate::movegen::generate_legal_moves(pos, &mut list);

    if let Some(s) = terminal_score(pos, state, ply, list.count > 0) {
        return s;
    }

    if depth <= 0 {
        return quiescence(pos, state, alpha, beta, QUIESCENCE_MAX_PLY, true, ply);
    }

    // Hard ply ceiling: check extensions and singular extensions can both
    // keep `depth` from shrinking to zero along a sufficiently forcing
    // line, so `ply` itself is the only thing guaranteed to grow every
    // recursive call. Bailing out one ply before the killers-table bound
    // avoids an out-of-bounds panic that -- under this crate's
    // `panic = "abort"` release profile -- would kill the whole process
    // with no unwinding, looking exactly like a hang from the UCI client's
    // side.
    if ply as usize >= MAX_PLY - 1 {
        return evaluate(pos, Some(&list));
    }

    let mut tt_move = Move::NULL;
    let mut tt_score_for_singular: i32 = 0;
    let tt_probe_start = std::time::Instant::now();
    let tt_probe_result = state.shared.probe(pos.zobrist_key);
    state.local.time_in_tt_probe_ns += tt_probe_start.elapsed().as_nanos() as u64;
    if let Some(entry) = tt_probe_result {
        tt_move = entry.best_move;
        let adjusted_score = score_from_tt(entry.score, ply);
        tt_score_for_singular = adjusted_score;
        // Singular-extension verification skips excluded `tt_move`.
        if excluded.is_null() && entry.depth >= depth {
            match entry.bound {
                Bound::Exact => return adjusted_score,
                Bound::Lower => {
                    if adjusted_score > alpha {
                        alpha = adjusted_score;
                    }
                }
                Bound::Upper => {
                    if adjusted_score < beta {
                        beta = adjusted_score;
                    }
                }
            }
            if alpha >= beta {
                return adjusted_score;
            }
        }
    }

    let in_check = is_in_check(pos, pos.side_to_move);
    let is_pv = beta - alpha > 1;

    // Per-node checking-squares table (see movegen.rs::CheckingSquares).
    // Computed once here and reused by every gives_check() call below (the
    // move loop and ProbCut) instead of each call redoing its own
    // bitboard-copy-and-attack-lookup from scratch.
    let check_ctx = compute_checking_squares(pos, pos.side_to_move);

    // Internal iterative reduction: with no TT move to anchor ordering,
    // search this node one ply shallower rather than paying full width for
    // a weakly-ordered node.
    if tt_move.is_null() && depth >= 4 && !in_check {
        depth -= 1;
    }

    // Null-move pruning
    if depth >= 3
        && !in_check
        && beta < MATE_SCORE - 1000
        && pos.non_pawn_king_material(pos.side_to_move)
    {
        let prev_ep = pos.make_null_move();
        let score = -negamax(pos, state, depth - 1 - 2, -beta, -beta + 1, ply + 1, Move::NULL, Move::NULL, ext);
        pos.unmake_null_move(prev_ep);
        if score >= beta {
            return beta;
        }
    }

    // Singular extensions: verify `tt_move` is singular by searching same
    // node without it, using a null window just below the TT score.
    if excluded.is_null()
        && !tt_move.is_null()
        && depth >= 4
        && !in_check
        && beta.abs() < MATE_SCORE - 1000
        && tt_probe_result.map_or(false, |e| e.depth >= depth - 3)
    {
        let singular_beta = tt_score_for_singular - 3 * depth;
        let singular_depth = (depth / 2).max(1);
        let singular_ext_start = std::time::Instant::now();
        // Verification search doesn't consume an extension itself (it's not
        // adding depth to any line yet) -- pass `ext` through unchanged.
        let sing_score = negamax(
            pos,
            state,
            singular_depth,
            singular_beta - 1,
            singular_beta,
            ply,
            prev_move,
            tt_move,
            ext,
        );
        state.local.time_in_singular_ext_ns += singular_ext_start.elapsed().as_nanos() as u64;
        // Singular extension budget capped (same shared per-line cap).
        if sing_score < singular_beta && ext < MAX_LINE_EXTENSIONS {
            depth += 1;
            ext += 1;
        }
    }

    // Static eval of the current node, reused by reverse-futility and
    // frontier futility pruning below. Not computed while in check (a
    // check-evasion static eval is meaningless / can't safely be used to
    // prune since the position is inherently tactical).
    let static_eval = if !in_check {
        let eval_start = std::time::Instant::now();
        let val = evaluate(pos, Some(&list));
        state.local.time_in_static_eval_ns += eval_start.elapsed().as_nanos() as u64;
        val
    } else {
        0
    };

    // Reverse futility / static null-move pruning
    if !in_check && !is_pv && depth <= 8 && beta.abs() < MATE_SCORE - 1000 {
        let margin = 85 * depth;
        if static_eval - margin >= beta {
            return static_eval - margin;
        }
    }

    let killers_snapshot = state.local.killers[ply as usize];
    let countermove = if !prev_move.is_null() {
        state.local.countermove[prev_move.from_sq() as usize][prev_move.to_sq() as usize]
    } else {
        Move::NULL
    };
    let mut orderer = MoveOrderer::new(pos, &list, tt_move, &killers_snapshot, countermove, &state.local.history, &check_ctx);

    let mut best_score = -MATE_SCORE;
    let mut best_move = Move::NULL;

    while let Some((move_index, m)) = orderer.next_move(&mut list) {
        // Singular-verification calls skip the move they're excluding.
        if !excluded.is_null() && m == excluded {
            continue;
        }

        let is_quiet = !m.flag().is_capture();
        // `gives_check` computed before gates; checking moves exempt
        // from LMP/futility since they're forcing.
        let gives_check = gives_check(pos, m, &check_ctx);

        // Late move pruning
        if !is_pv && !in_check && is_quiet && !gives_check && depth <= 8 {
            let lmp_threshold = 4 + (depth * depth) as usize;
            if move_index >= lmp_threshold {
                continue;
            }
        }

        // Futility pruning at frontier nodes
        if !is_pv && !in_check && is_quiet && !gives_check && depth <= 8 && move_index > 0 {
            let margin = 90 + 80 * depth;
            if static_eval + margin <= alpha {
                continue;
            }
        }

        // SEE-based pruning for losing captures
        if m.flag().is_capture() && !is_pv && depth <= 8 && !in_check && ply > 0 && move_index > 0 {
            let see_margin = 10 * depth + 15;
            let see_val = crate::movegen::see(pos, m);
            if see_val < -see_margin {
                continue;
            }
        }

        // Bounded ProbCut insertion: early-cut noisy alternatives on
        // non-PV nodes.
        //
        // ProbCut gate: exclude mate-range scores (`beta.abs() < MATE_SCORE - 1000`).
        if !is_pv
            && !in_check
            && depth <= 8
            && !is_quiet
            && move_index > 0
            && !tt_move.is_null()
            && tt_move.flag().is_capture()
            && beta.abs() < MATE_SCORE - 1000
        {
            let improving = if best_score > alpha { 1 } else { 0 };
            let probcut_beta = beta + 254 - 85 * improving;
            if static_eval >= probcut_beta {
                let reduced_depth = (depth - ((static_eval - probcut_beta) / 319) as i32).max(1);
                pos.make_move(m);
                // `gives_check` is the value computed above the pruning
                // gates, before this move was made -- no need to ask
                // is_in_check() again now that it's already known.
                let mut child_depth_reduced = reduced_depth - 1;
                let mut probcut_ext = ext;
                // Shared per-line extension budget for ProbCut check bump.
                if gives_check && ext < MAX_LINE_EXTENSIONS {
                    child_depth_reduced += 1;
                    probcut_ext += 1;
                }
                state.push_rep(pos.zobrist_key);
                let probcut_start = std::time::Instant::now();
                let probcut_score = -negamax(
                    pos,
                    state,
                    child_depth_reduced,
                    -beta - 1,
                    -alpha,
                    ply + 1,
                    m,
                    Move::NULL,
                    probcut_ext,
                );
                state.local.time_in_probcut_ns += probcut_start.elapsed().as_nanos() as u64;
                state.pop_rep();
                pos.unmake_move(m);
                if probcut_score >= probcut_beta {
                    let interpolated = ((probcut_score as f32) * 0.2695 + (beta as f32) * 0.7305) as i32;
                    return interpolated;
                }
                // Falls through to the normal full-depth path below if the
                // reduced-depth score doesn't confirm the cut.
            }
        }

        pos.make_move(m);
        // Reusing the `gives_check` computed above the pruning gates --
        // same move, same position, no need to ask is_in_check() again.
        let mut child_depth = depth - 1;
        let mut child_ext = ext;
        // Check-extension budget capped per line (shared with singular/ProbCut).
        if gives_check && ext < MAX_LINE_EXTENSIONS {
            child_depth += 1;
            child_ext += 1;
        }

        let mut score;
        if move_index == 0 {
            state.push_rep(pos.zobrist_key);
            score = -negamax(pos, state, child_depth, -beta, -alpha, ply + 1, m, Move::NULL, child_ext);
            state.pop_rep();
        } else {
            let mut reduced_depth = child_depth;
            let do_lmr = is_quiet && !gives_check && depth >= 3 && move_index >= 3;
            if do_lmr {
                let reduction = if move_index >= 9 { 2 } else { 1 };
                reduced_depth = (child_depth - reduction).max(1);
            }

            state.push_rep(pos.zobrist_key);
            score = -negamax(pos, state, reduced_depth, -alpha - 1, -alpha, ply + 1, m, Move::NULL, child_ext);

            if do_lmr && score > alpha {
                score = -negamax(pos, state, child_depth, -alpha - 1, -alpha, ply + 1, m, Move::NULL, child_ext);
                if score > alpha && score < beta {
                    score = -negamax(pos, state, child_depth, -beta, -alpha, ply + 1, m, Move::NULL, child_ext);
                }
            } else if !do_lmr && score > alpha && score < beta {
                score = -negamax(pos, state, child_depth, -beta, -alpha, ply + 1, m, Move::NULL, child_ext);
            }
            state.pop_rep();
        }

        pos.unmake_move(m);

        if score > best_score {
            best_score = score;
            best_move = m;
        }
        if best_score > alpha {
            alpha = best_score;
        }
        if alpha >= beta {
            if is_quiet {
                let killers = &mut state.local.killers[ply as usize];
                if killers[0] != m {
                    killers[1] = killers[0];
                    killers[0] = m;
                }
                state.local.history[m.from_sq() as usize][m.to_sq() as usize] += depth * depth;
                if !prev_move.is_null() {
                    state.local.countermove[prev_move.from_sq() as usize][prev_move.to_sq() as usize] = m;
                }
            }
            break;
        }
    }

    // If `best_score` is still -MATE_SCORE here, every move considered was
    // the excluded move -- only possible inside a singular-verification
    // call where tt_move turned out to be the position's only legal move.
    // That's not a bug: it means the position is genuinely forced, so
    // reporting a maximally-low score here (which makes the caller's
    // `sing_score < singular_beta` check extend) is the correct answer.
    let bound = if best_score <= orig_alpha {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };

    // Gate: singular-verification calls must not store to shared TT.
    if excluded.is_null() {
        state.shared.store(pos.zobrist_key, depth, score_to_tt(best_score, ply), bound, best_move);
    }

    best_score
}
