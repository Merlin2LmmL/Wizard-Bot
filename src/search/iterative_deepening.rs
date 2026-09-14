use super::negamax::negamax;
use super::ordering::score_move;
use super::thread_state::{SearchLimits, SearchState, ThreadLocalSearch};
use super::tt::{Bound, SharedSearch};
use super::{now_ms, GO_TOKEN, MAX_LINE_EXTENSIONS, STOP_FLAG};
use crate::eval::MATE_SCORE;
use crate::movegen::{compute_checking_squares, generate_legal_moves};
use crate::position::*;
use std::sync::atomic::Ordering;

#[allow(dead_code)]
pub struct RootResult {
    pub best_move: Move,
    pub score: i32,
}

/// Walk shared TT from pos through best_move entries to reconstruct PV.
fn extract_pv(pos: &Position, shared: &SharedSearch, first_move: Move, max_len: usize) -> Vec<Move> {
    let mut pv = Vec::with_capacity(max_len.max(1));
    if first_move.is_null() {
        return pv;
    }
    let mut walk = pos.clone();
    walk.make_move(first_move);
    pv.push(first_move);
    let mut seen = std::collections::HashSet::new();
    seen.insert(walk.zobrist_key);
    while pv.len() < max_len {
        let entry = match shared.probe(walk.zobrist_key) {
            Some(e) => e,
            None => break,
        };
        // Only chain PV off Bound::Exact entries.
        if entry.best_move.is_null() || entry.bound != Bound::Exact {
            break;
        }
        let mut list = MoveList::new();
        generate_legal_moves(&walk, &mut list);
        if !list.as_slice().iter().any(|&mv| mv == entry.best_move) {
            break; // stale/colliding TT entry -- don't trust it past this point
        }
        walk.make_move(entry.best_move);
        if !seen.insert(walk.zobrist_key) {
            break; // would loop forever through a repeated position
        }
        pv.push(entry.best_move);
    }
    pv
}

pub fn iterative_deepening<F: FnMut(&str)>(
    pos: &mut Position,
    state: &mut SearchState,
    limits: SearchLimits,
    go_token: u32,
    mut send: F,
) -> Move {
    state.local.go_token = go_token;
    state.local.deadline_ms = limits.deadline_ms;
    state.local.max_depth = limits.max_depth;
    state.local.stopped = false;
    state.local.nodes = 0;
    state.shared.nodes_aggregate.store(0, Ordering::Relaxed);

    let mut root_list = MoveList::new();
    generate_legal_moves(pos, &mut root_list);
    if root_list.count == 0 {
        return Move::NULL;
    }

    // Checking-squares table for root-move ordering; computed once (root pos fixed).
    let root_check_ctx = compute_checking_squares(pos, pos.side_to_move);

    // Filter root moves to book candidates once, before search.
    let book_start = std::time::Instant::now();
    let book_candidates: Vec<&'static str> = crate::book::get_book_candidates(pos);
    if !book_candidates.is_empty() {
        let book_moves: std::collections::HashSet<String> =
            book_candidates.iter().map(|u| u.to_string()).collect();
        let original_count = root_list.count;
        let mut filtered_moves = [Move::NULL; 218]; // MAX_MOVES
        let mut filtered_count = 0;
        for i in 0..original_count {
            let m = root_list.moves[i];
            if book_moves.contains(&m.to_uci()) {
                filtered_moves[filtered_count] = m;
                filtered_count += 1;
            }
        }
        // Keep full list if filtering would empty it.
        if filtered_count > 0 {
            root_list.moves = filtered_moves;
            root_list.count = filtered_count;
        }
    }
    state.local.time_in_book_filter_ns = book_start.elapsed().as_nanos() as u64;

    let mut best_move = root_list.as_slice()[0];
    let mut best_score = -MATE_SCORE;
    let mut prev_best_move = Move::NULL;
    let start_ms = now_ms();

    // Per-root-move state lives across depth/retry loops; move-order tables persist.
    let num_root_moves = root_list.count;
    let per_root_state: Vec<std::sync::Mutex<ThreadLocalSearch>> =
        (0..num_root_moves).map(|_| std::sync::Mutex::new(ThreadLocalSearch::new())).collect();

    // Repetition seed from real game history.
    let mut ancestor_keys: Vec<u64> = pos.undo_stack.iter().map(|u| u.prev_zobrist).collect();
    ancestor_keys.push(pos.zobrist_key);
    let look_back = pos.halfmove_clock as usize;
    let seed_start = ancestor_keys.len().saturating_sub(look_back);
    let rep_seed: Vec<u64> = ancestor_keys[seed_start..].to_vec();

    for depth in 1..=limits.max_depth {
        if state.local.stopped || STOP_FLAG.load(Ordering::Relaxed) || go_token != GO_TOKEN.load(Ordering::Relaxed) {
            break;
        }
        if now_ms() >= limits.deadline_ms {
            break;
        }

        let root_killers = state.local.killers[0];

        let mut window = 50;
        let mut alpha;
        let mut beta;
        if depth <= 2 || best_score.abs() >= MATE_SCORE - 1000 {
            alpha = -MATE_SCORE;
            beta = MATE_SCORE;
        } else {
            alpha = best_score - window;
            beta = best_score + window;
        }

        let mut depth_best_move = best_move;
        #[allow(unused_assignments)]
        let mut depth_best_score = -MATE_SCORE;

        #[allow(unused_assignments)]
        let mut depth_had_valid_result = false;
        'aspiration: loop {
            depth_best_score = -MATE_SCORE;
            depth_had_valid_result = false;
            let root_moves: Vec<Move> = root_list.as_slice().to_vec();
            let mut aggregated_root_history = [[0i32; 64]; 64];
            for slot in per_root_state.iter() {
                let guard = slot.lock().unwrap();
                for f in 0..64 {
                    for t in 0..64 {
                        aggregated_root_history[f][t] += guard.history[f][t];
                    }
                }
            }
            let mut spawn_order: Vec<usize> = (0..root_moves.len()).collect();
            spawn_order.sort_by_key(|&i| {
                -score_move(pos, root_moves[i], prev_best_move, &root_killers, Move::NULL, &aggregated_root_history, &root_check_ctx)
            });

            let round_start_ms = now_ms();
            eprintln!(
                "TIMING: depth={} aspiration_round_start elapsed_since_go={:.1}ms window={} alpha={} beta={} deadline_in={:.1}ms",
                depth, round_start_ms - start_ms, window, alpha, beta, limits.deadline_ms - round_start_ms
            );

            let _shared_clone = state.shared.clone();
            let results_arc = std::sync::Arc::new(std::sync::Mutex::new(vec![(Move::NULL, -MATE_SCORE, false); root_moves.len()]));
            let _base_pos_clone = (*pos).clone();

            #[cfg(feature = "parallel-search")]
            {
                let per_root_state_ref = &per_root_state;
                let rep_seed_ref = &rep_seed;
                rayon::scope(|s| {
                    for &i in spawn_order.iter() {
                        if STOP_FLAG.load(Ordering::Relaxed) || state.local.stopped {
                            break;
                        }
                        let m = root_moves[i];
                        let shared = _shared_clone.clone();
                        let results_ref = results_arc.clone();
                        let pos_for_this = _base_pos_clone.clone();
                        let deadline_for_this = state.local.deadline_ms;
                        let go_token_for_this = state.local.go_token;
                        let spawn_queued_ms = now_ms();
                        s.spawn(move |_| {
                            // Queue lag: time between queuing and worker start.
                            let actual_start_ms = now_ms();
                            let queue_lag_ms = actual_start_ms - spawn_queued_ms;
                            if queue_lag_ms > 20.0 {
                                eprintln!(
                                    "TIMING: depth={} move={} QUEUE_LAG={:.1}ms (deadline_in={:.1}ms at start)",
                                    depth, m.to_uci(), queue_lag_ms, deadline_for_this - actual_start_ms
                                );
                            }
                            let shared_for_aggregate = shared.clone();
                            let mut local_tls = {
                                let mut guard = per_root_state_ref[i].lock().unwrap();
                                std::mem::replace(&mut *guard, ThreadLocalSearch::new())
                            };
                            local_tls.deadline_ms = deadline_for_this;
                            local_tls.go_token = go_token_for_this;
                            local_tls.stopped = false;
                            local_tls.nodes = 0;
                            local_tls.rep_stack.clear();
                            local_tls.rep_stack.extend_from_slice(rep_seed_ref);
                            let mut local_state = SearchState {
                                shared,
                                local: local_tls,
                            };
                            let mut temp_pos = pos_for_this.clone();
                            temp_pos.make_move(m);
                            let gives_check = crate::movegen::is_in_check(&temp_pos, temp_pos.side_to_move);
                            // Cap check-extension depth; do not let it grow unbounded.
                            let (child_depth, child_ext) = if gives_check && 0 < MAX_LINE_EXTENSIONS {
                                (depth, 1)
                            } else {
                                (depth - 1, 0)
                            };
                            local_state.push_rep(temp_pos.zobrist_key);
                            let search_start_ms = now_ms();
                            let score = -negamax(&mut temp_pos, &mut local_state, child_depth, -beta, -alpha, 1, m, Move::NULL, child_ext);
                            let search_duration_ms = now_ms() - search_start_ms;
                            local_state.pop_rep();
                            temp_pos.unmake_move(m);
                            let this_move_stopped = local_state.local.stopped;
                            // TIMING: wall time this thread's own negamax
                            // Duration/nodes timing for debugging overrun.
                            eprintln!(
                                "TIMING: depth={} move={} duration={:.1}ms nodes={} stopped={} finished_after_deadline_by={:.1}ms",
                                depth, m.to_uci(), search_duration_ms, local_state.local.nodes, this_move_stopped,
                                now_ms() - deadline_for_this
                            );
                            results_ref.lock().unwrap()[i] = (m, score, this_move_stopped);
                            shared_for_aggregate.nodes_aggregate.fetch_add(local_state.local.nodes, Ordering::Relaxed);
                            *per_root_state_ref[i].lock().unwrap() = local_state.local;
                        });
                    }
                });
                eprintln!(
                    "TIMING: depth={} scope_returned elapsed_since_round_start={:.1}ms",
                    depth, now_ms() - round_start_ms
                );
            }
            #[cfg(not(feature = "parallel-search"))]
            {
                // Sequential fallback without rayon spawn.
                for &i in spawn_order.iter() {
                    if STOP_FLAG.load(Ordering::Relaxed) || state.local.stopped {
                        break;
                    }
                    let m = root_moves[i];
                    let shared_for_aggregate = _shared_clone.clone();
                    let mut local_tls = {
                        let mut guard = per_root_state[i].lock().unwrap();
                        std::mem::replace(&mut *guard, ThreadLocalSearch::new())
                    };
                    local_tls.deadline_ms = state.local.deadline_ms;
                    local_tls.go_token = state.local.go_token;
                    local_tls.stopped = false;
                    local_tls.nodes = 0;
                    local_tls.rep_stack.clear();
                    local_tls.rep_stack.extend_from_slice(&rep_seed);
                    let mut local_state = SearchState {
                        shared: shared_for_aggregate.clone(),
                        local: local_tls,
                    };
                    let mut temp_pos = _base_pos_clone.clone();
                    temp_pos.make_move(m);
                    let gives_check = crate::movegen::is_in_check(&temp_pos, temp_pos.side_to_move);
                    // Same fix as the parallel-search spawn closure above.
                    let (child_depth, child_ext) = if gives_check && 0 < MAX_LINE_EXTENSIONS {
                        (depth, 1)
                    } else {
                        (depth - 1, 0)
                    };
                    local_state.push_rep(temp_pos.zobrist_key);
                    let score = -negamax(&mut temp_pos, &mut local_state, child_depth, -beta, -alpha, 1, m, Move::NULL, child_ext);
                    local_state.pop_rep();
                    temp_pos.unmake_move(m);
                    let this_move_stopped = local_state.local.stopped;
                    results_arc.lock().unwrap()[i] = (m, score, this_move_stopped);
                    shared_for_aggregate.nodes_aggregate.fetch_add(local_state.local.nodes, Ordering::Relaxed);
                    *per_root_state[i].lock().unwrap() = local_state.local;
                }
            }

            let results = std::sync::Arc::try_unwrap(results_arc).unwrap().into_inner().unwrap();
            // Skip stopped moves; don't discard completed results.
            let mut any_move_stopped = false;
            // If incumbent best move was starved, don't trust this depth.
            let mut incumbent_starved = false;
            for (m, score, this_stopped) in results {
                if this_stopped {
                    any_move_stopped = true;
                    if m == prev_best_move {
                        incumbent_starved = true;
                    }
                    continue;
                }
                depth_had_valid_result = true;
                if score > depth_best_score {
                    depth_best_score = score;
                    depth_best_move = m;
                }
                if depth_best_score > alpha {
                    alpha = depth_best_score;
                }
            }
            if any_move_stopped {
                state.local.stopped = true;
            }
            if incumbent_starved {
                eprintln!(
                    "TIMING: depth={} INCUMBENT_STARVED prev_best_move={} elapsed_since_go={:.1}ms",
                    depth, prev_best_move.to_uci(), now_ms() - start_ms
                );
                // Starved incumbent means this depth isn't trustworthy; keep previous result.
                depth_had_valid_result = false;
            }

            // Direct clock check prevents extra full-width re-search when time is already past deadline.
            if state.local.stopped || now_ms() >= limits.deadline_ms {
                eprintln!(
                    "TIMING: depth={} aspiration_abort_on_deadline elapsed_since_go={:.1}ms any_move_stopped={}",
                    depth, now_ms() - start_ms, any_move_stopped
                );
                state.local.stopped = true;
                break 'aspiration;
            }
            if depth > 2 && best_score.abs() < MATE_SCORE - 1000 {
                if depth_best_score <= alpha - 0 && depth_best_score <= (best_score - window) {
                    window *= 4;
                    if window > 2000 {
                        alpha = -MATE_SCORE;
                        beta = MATE_SCORE;
                    } else {
                        alpha = best_score - window;
                        beta = best_score + window;
                    }
                    continue 'aspiration;
                }
                if depth_best_score >= beta {
                    window *= 4;
                    if window > 2000 {
                        alpha = -MATE_SCORE;
                        beta = MATE_SCORE;
                    } else {
                        alpha = best_score - window;
                        beta = best_score + window;
                    }
                    continue 'aspiration;
                }
            }
            break 'aspiration;
        }

        // Report this depth's result if it completed, regardless of stopped state.
        if depth_had_valid_result {
            let commit_start_ms = now_ms();
            best_score = depth_best_score;
            best_move = depth_best_move;
            prev_best_move = best_move;

            let total_nodes = state.local.nodes + state.shared.nodes_aggregate.load(Ordering::Relaxed);
            let elapsed_ms = (now_ms() - start_ms).max(1.0);
            let nps = (total_nodes as f64 / (elapsed_ms / 1000.0)) as u64;

            // TEMPDEBUG: bookns is one-time root-level; others summed across root moves.
            let bookns_acc = state.local.time_in_book_filter_ns;
            let mut ttns_acc = 0u64;
            let mut singns_acc = 0u64;
            let mut evalns_acc = 0u64;
            let mut probcutns_acc = 0u64;
            for slot in per_root_state.iter() {
                let guard = slot.lock().unwrap();
                ttns_acc += guard.time_in_tt_probe_ns;
                singns_acc += guard.time_in_singular_ext_ns;
                evalns_acc += guard.time_in_static_eval_ns;
                probcutns_acc += guard.time_in_probcut_ns;
            }
            eprintln!(
                "TIMING: depth={} per_root_state_lock_sweep_took={:.1}ms",
                depth, now_ms() - commit_start_ms
            );

            if best_score == -MATE_SCORE {
                send(&format!(
                    "info depth {} score cp -32768 nodes {} nps {} time {} pv (none) bookns {} ttns {} singns {} evalns {} probcutns {}",
                    depth, total_nodes, nps, elapsed_ms as u64, bookns_acc, ttns_acc, singns_acc, evalns_acc, probcutns_acc
                ));
            } else if best_score.abs() >= MATE_SCORE - 1000 {
                let mate_in = ((MATE_SCORE - best_score.abs() + 1) / 2) * best_score.signum();
                let pv_start_ms = now_ms();
                let pv = extract_pv(pos, &state.shared, best_move, depth as usize);
                eprintln!("TIMING: depth={} extract_pv_took={:.1}ms pv_len={}", depth, now_ms() - pv_start_ms, pv.len());
                let pv_str: Vec<String> = pv.iter().map(|m| m.to_uci()).collect();
                send(&format!(
                    "info depth {} score mate {} nodes {} nps {} time {} pv {} bookns {} ttns {} singns {} evalns {} probcutns {}",
                    depth, mate_in, total_nodes, nps, elapsed_ms as u64, pv_str.join(" "), bookns_acc, ttns_acc, singns_acc, evalns_acc, probcutns_acc
                ));
            } else {
                let pv_start_ms = now_ms();
                let pv = extract_pv(pos, &state.shared, best_move, depth as usize);
                eprintln!("TIMING: depth={} extract_pv_took={:.1}ms pv_len={}", depth, now_ms() - pv_start_ms, pv.len());
                let pv_str: Vec<String> = pv.iter().map(|m| m.to_uci()).collect();
                send(&format!(
                    "info depth {} score cp {} nodes {} nps {} time {} pv {} bookns {} ttns {} singns {} evalns {} probcutns {}",
                    depth, best_score, total_nodes, nps, elapsed_ms as u64, pv_str.join(" "), bookns_acc, ttns_acc, singns_acc, evalns_acc, probcutns_acc
                ));
            }

            // Stop after committing result; don't discard it for a straggler.
            if state.local.stopped {
                break;
            }
        } else {
            // No completed results at this depth; nothing usable to report.
            break;
        }
    }

    best_move
}
