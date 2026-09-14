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

/// Reconstructs the expected principal variation by walking the shared TT
/// forward from `pos` through `first_move` and then whatever best_move each
/// successive position's TT entry holds, validating legality at each step.
/// This is best-effort (TT entries can be missing or stale from a shallower
/// search / different node) but is far more useful for debugging than
/// printing only the root move.
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
        // FIX: this used to trust `entry.best_move` regardless of
        // `entry.bound`. On a fail-low (Bound::Upper) store, best_move is
        // whichever move happened to be tried first/best-of-a-bad-bunch --
        // the search never established it as genuinely best, only that
        // nothing beat alpha. A fail-high (Bound::Lower) move is real but
        // the position's value there is only a lower bound, not confirmed
        // exact -- still not safe to chain further PV off of. Since the
        // shared TT is written by every root-move thread's subtree
        // concurrently (all against the same aspiration window), a
        // transposition into this same position from a totally unrelated
        // line can and does leave behind exactly this kind of unvalidated
        // entry. Only Bound::Exact certifies "this move and this
        // subtree's backed-up value are the real answer for this
        // position" -- anything else, stop rather than display it as if
        // it were part of a calculated line.
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

    // Per-node checking-squares table (see movegen.rs::CheckingSquares),
    // needed by score_move() below to order root moves before each
    // aspiration round's spawn. `pos` is the fixed root position for this
    // whole call (make_move/unmake_move pairs inside the search always
    // restore it), so this is computed once here rather than per depth or
    // per aspiration retry.
    let root_check_ctx = compute_checking_squares(pos, pos.side_to_move);

    // FIX: opening-book filtering used to live inside negamax() and run at
    // EVERY node of the entire search tree, not just here at the true
    // root. That caused two separate problems:
    //   1. Correctness: any interior node the search reached whose position
    //      happened to have book coverage got its legal-move list silently
    //      cut down to "whatever get_book_candidates() returns" for the
    //      rest of that subtree -- tactically blind to non-book replies
    //      deep in the tree, any time a transposition wandered back into
    //      book territory, not just at the real opening.
    //   2. Performance: get_book_candidates() plus building a fresh
    //      HashSet<String> (via to_uci() on every candidate) ran once per
    //      *node* with coverage, not once per search. If that lookup isn't
    //      microseconds-cheap, this is a far better explanation for a
    //      "10 seconds wall clock, ~77 nodes searched" anomaly than
    //      anything inside negamax/quiescence itself.
    // Filtering here instead -- once, on the real root move list, before
    // any recursive search starts -- gets the intended "prefer book moves"
    // behavior for the move the engine actually plays, without either
    // problem. bookns should now read close to zero in the info line;
    // if the wall-clock/node-count anomaly still shows up after this
    // change, that's a real signal it's *not* the book after all and the
    // other TEMPDEBUG counters (ttns/singns/evalns/probcutns) are the next
    // place to look.
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
        // FIX (defensive, new): the old per-node version had no fallback
        // here. If none of the generated legal moves' UCI strings matched
        // anything the book returned (stale data, notation mismatch,
        // whatever), `list.count` would silently become 0 and the caller
        // would treat the position as checkmate/stalemate. At the root
        // that would mean the engine reports "no legal moves" in a normal
        // position. Falling back to the full legal list if filtering would
        // empty it out keeps a book data bug from ever becoming a game-
        // ending bug.
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

    // Persistent per-root-move search state: index i always corresponds to
    // root_list.as_slice()[i] (root move order never changes once computed
    // above), and this Vec lives for the whole iterative-deepening call --
    // outside both the depth loop and the aspiration-window retry loop
    // inside it, so move-ordering tables built at shallower depth carry
    // forward into deeper iterations instead of resetting every call.
    let num_root_moves = root_list.count;
    let per_root_state: Vec<std::sync::Mutex<ThreadLocalSearch>> =
        (0..num_root_moves).map(|_| std::sync::Mutex::new(ThreadLocalSearch::new())).collect();

    // Seed for repetition detection, built from the *real* game history --
    // not just whatever this search happens to revisit on its own. See the
    // original file's comment history for the full rationale; unchanged
    // here.
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
                            // TIMING: gap between when this task was queued
                            // (rayon::scope's for-loop reached it) and when a
                            // worker thread actually started running it. On a
                            // pool with fewer workers than root moves, a
                            // large gap here means this move sat waiting
                            // while other moves' searches ran -- and if that
                            // gap alone exceeds the remaining time to
                            // deadline, this thread has already lost before
                            // doing a single node of real work.
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
                            // BUGFIX: same uncapped check-extension pattern as
                            // negamax's main loop (see MAX_LINE_EXTENSIONS in
                            // mod.rs) -- was `depth < 40`, which let every
                            // root move that gives check start its own line
                            // at full, un-shrunk depth with nothing tracking
                            // how many times that had already happened.
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
                            // call actually took, vs. how many nodes it did
                            // and how late it finished relative to the
                            // deadline. Low nodes + high duration = stalled
                            // (blocked on something, not computing). High
                            // nodes + high duration = genuinely expensive
                            // subtree (extensions/singular/probcut chain).
                            // finished_after_deadline_by should be near-zero
                            // or negative if time_up() is catching things
                            // promptly; a large positive value here is
                            // exactly the "invisible overrun" we're hunting.
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
                // Sequential single-threaded fallback: evaluate each root move
                // directly without rayon spawn, sharing the same TT via a
                // cloned Arc and writing into the same results_arc so
                // downstream aggregation is unchanged.
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
            // A root move whose thread hit its own deadline mid-search never
            // completed a real negamax result -- time_up() short-circuited
            // every recursive call under it to evaluate(pos, None), a raw
            // static eval with zero search behind it. That score must never
            // win the comparison below against a move that DID complete a
            // full search, so it's still skipped entirely (`continue`).
            //
            // BUGFIX: this used to *also* set `state.local.stopped = true`
            // whenever ANY move was stopped, which discarded every other
            // root move's fully-completed, correct result for this depth
            // and fell back to the previous (shallower) depth's answer.
            // With root-move-per-thread parallelism on a pool sized to the
            // core count, any position with more legal moves than cores --
            // i.e. almost every real middlegame position -- guarantees a
            // late-queued task gets scheduled only once the deadline has
            // already passed, at which point it calls time_up() instantly
            // and reports stopped=true with zero real search behind it.
            // `spawn_order` already runs the best-looking moves first, so
            // the straggler is typically one of the worst-ordered, least
            // relevant candidates -- yet it was enough to throw away the
            // fully-searched result for every OTHER move too, including
            // whichever one actually found the winning continuation or a
            // forced mate. That's the direct cause of "search says mate,
            // engine plays something else": the depth that found the mate
            // got discarded because an unrelated move was starved for CPU
            // time by rayon's scheduler, not because anything was wrong
            // with the mate-finding thread's own result.
            //
            // Fix: keep and report whatever this depth's completed root
            // moves actually produced. We still stop searching any deeper
            // (there's no time left regardless), but we no longer throw
            // away correct, finished work over an unrelated straggler.
            let mut any_move_stopped = false;
            // BUGFIX: excluding a stopped move's own score from the max isn't
            // enough. If the move whose thread got starved is specifically
            // the *incumbent* best move carried in from the previous
            // completed depth, every other, cheaper move still "wins" this
            // depth's comparison by elimination -- even though the real best
            // move never got a fair, finished score at this depth. This is
            // the exact mechanism behind the g6f8 game: a tactically
            // demanding move is disproportionately likely to be the
            // straggler precisely because it's the one worth searching
            // deepest. Track that case specifically.
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
                // The move we already trusted never finished at this depth,
                // so whatever "won" among the finished moves is not a
                // trustworthy improvement over it. Don't report or commit
                // this depth's result; fall through to the "nothing usable"
                // path below, which leaves best_move/best_score exactly as
                // the previous depth left them.
                depth_had_valid_result = false;
            }

            // BUGFIX: `state.local.stopped` only becomes true via
            // `any_move_stopped`, which itself only fires if some thread's
            // own periodic (every-4096-node) check happened to notice the
            // deadline. That's an indirect proxy for "is there time left,"
            // not the deadline itself -- on a genuine fail-high/fail-low
            // (common in sharp positions: a hanging piece just got found),
            // every currently-running thread can easily finish its own
            // subtree cleanly, without ever hitting another checkpoint,
            // even though real wall-clock time has already blown well past
            // `deadline_ms`. Without a direct clock check here, that lets
            // the loop respawn all root-move threads again at a wider
            // window -- a full-cost re-search of every move -- with zero
            // regard for whether any time is actually left. This is the
            // exact mechanism behind large chunks of a move's budget going
            // "unaccounted for": not a discarded deeper depth, but 2-3
            // extra full-width research rounds at the SAME depth, none of
            // which produce a reported result until the loop finally exits.
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

        // BUGFIX: previously gated on `!state.local.stopped`, which meant a
        // depth where every OTHER root move completed correctly but one
        // straggler timed out (see the comment above) reported nothing at
        // all for this depth -- silently keeping the previous, shallower
        // depth's move as `best_move` even though we have a better, fully
        // -searched answer sitting right here. Gate on whether this depth
        // actually produced a completed result instead; `state.local.stopped`
        // still ends the outer depth loop below (correctly -- we're out of
        // time either way), it just no longer suppresses reporting the
        // result we already have.
        if depth_had_valid_result {
            let commit_start_ms = now_ms();
            best_score = depth_best_score;
            best_move = depth_best_move;
            prev_best_move = best_move;

            let total_nodes = state.local.nodes + state.shared.nodes_aggregate.load(Ordering::Relaxed);
            let elapsed_ms = (now_ms() - start_ms).max(1.0);
            let nps = (total_nodes as f64 / (elapsed_ms / 1000.0)) as u64;

            // TEMPDEBUG: bookns is now a one-time root-level cost (see the
            // fix above), so it's read directly off the coordinator's own
            // ThreadLocalSearch rather than summed across per_root_state.
            // The other counters still measure per-node work inside
            // negamax/quiescence, so they're still summed across every
            // root move's persisted state as before.
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

            // We still stop here if this depth also had a starved straggler
            // (state.local.stopped) -- there's genuinely no time left for
            // another full iteration -- but unlike before, we now do so
            // *after* committing this depth's valid result above, instead
            // of discarding it.
            if state.local.stopped {
                break;
            }
        } else {
            // Nothing at all completed this depth (e.g. even the
            // highest-priority root move never got a real search in before
            // the deadline) -- there's genuinely nothing usable to report.
            break;
        }
    }

    // NOTE (historical): a "root-level blunder guard" previously lived
    // here, re-checking best_move with a shallow 1-ply capture-only
    // heuristic and silently substituting a different root move whenever
    // that heuristic didn't like the result. It had no way to see mating
    // nets, promotion follow-up, or positional compensation the full
    // alpha-beta search already accounts for to full depth, so it was
    // strictly worse than trusting the search. Removed; the code for it
    // (`net_swing_after_move`) was dead and has been dropped from this
    // split rather than carried forward -- shout if you want it back.

    best_move
}
