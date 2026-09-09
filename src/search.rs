use crate::bitboard::*;
use crate::eval::{evaluate, MATE_SCORE};
use crate::movegen::{generate_legal_moves, is_in_check};
use crate::position::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

pub static STOP_FLAG: AtomicBool = AtomicBool::new(false);
pub static GO_TOKEN: AtomicU32 = AtomicU32::new(0);

const QUIESCENCE_MAX_PLY: i32 = 6;
const MAX_PLY: usize = 128;

/// wasm32-unknown-unknown has no OS clock: std::time::SystemTime::now()
/// panics outright there ("time not implemented on this platform"), it
/// doesn't just return a wrong value. js_sys::Date is what actually reaches
/// the browser's clock via JS interop, so we route through that on wasm32
/// and keep the normal std path everywhere else. (Same helper as in
/// uci.rs/book.rs -- kept as a local copy here rather than a shared module
/// to avoid introducing a cross-module dependency for one function.)
fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as f64
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Copy, Clone)]
struct TTEntry {
    key: u64,
    // Sentinel: depth < 0 means "slot empty". Real search depths passed to
    // store() are always >= 0, so this needs no extra discriminant/padding
    // the way `Option<TTEntry>` did (u64 key has no spare-bit niche).
    depth: i32,
    score: i32,
    bound: Bound,
    best_move: Move,
}

impl TTEntry {
    const EMPTY: TTEntry = TTEntry {
        key: 0,
        depth: -1,
        score: 0,
        bound: Bound::Exact,
        best_move: Move::NULL,
    };
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.depth < 0
    }
}

/// Entries per bucket. Every probe/store scans one bucket (a handful of
/// cache-adjacent entries) instead of a single depth-preferred slot; this
/// measurably improves hit rate at fixed memory since a colliding entry no
/// longer necessarily evicts a still-useful one.
const TT_BUCKET_SIZE: usize = 4;
type TTBucket = [TTEntry; TT_BUCKET_SIZE];

pub struct TranspositionTable {
    buckets: Vec<TTBucket>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let bucket_bytes = std::mem::size_of::<TTBucket>().max(1);
        let mut bucket_count = (mb * 1024 * 1024 / bucket_bytes).next_power_of_two();
        bucket_count = bucket_count.max(1 << 12);
        TranspositionTable {
            buckets: vec![[TTEntry::EMPTY; TT_BUCKET_SIZE]; bucket_count],
            mask: bucket_count - 1,
        }
    }
    #[inline]
    fn bucket_idx(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }
    fn probe(&self, key: u64) -> Option<TTEntry> {
        let bucket = &self.buckets[self.bucket_idx(key)];
        for e in bucket.iter() {
            if !e.is_empty() && e.key == key {
                return Some(*e);
            }
        }
        None
    }
    fn store(&mut self, key: u64, depth: i32, score: i32, bound: Bound, best_move: Move) {
        let idx = self.bucket_idx(key);
        let bucket = &mut self.buckets[idx];

        // Same position already in this bucket: refresh in place (only
        // overwrite with a shallower search if we have nothing better).
        for slot in bucket.iter_mut() {
            if !slot.is_empty() && slot.key == key {
                if depth >= slot.depth || bound == Bound::Exact {
                    *slot = TTEntry { key, depth, score, bound, best_move };
                }
                return;
            }
        }
        // Free slot in the bucket.
        for slot in bucket.iter_mut() {
            if slot.is_empty() {
                *slot = TTEntry { key, depth, score, bound, best_move };
                return;
            }
        }
        // Bucket full of other positions: evict the shallowest entry.
        let mut worst_i = 0;
        let mut worst_depth = bucket[0].depth;
        for (i, slot) in bucket.iter().enumerate().skip(1) {
            if slot.depth < worst_depth {
                worst_depth = slot.depth;
                worst_i = i;
            }
        }
        bucket[worst_i] = TTEntry { key, depth, score, bound, best_move };
    }
}

pub struct SearchLimits {
    pub max_depth: i32,
    pub deadline_ms: f64, // absolute JS Date.now() timestamp
}

use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct SharedSearch {
    pub tt: Arc<RwLock<TranspositionTable>>,
    pub nodes_aggregate: Arc<AtomicU64>,
}

impl SharedSearch {
    pub fn new(mb: usize) -> Self {
        SharedSearch {
            tt: Arc::new(RwLock::new(TranspositionTable::new(mb))),
            nodes_aggregate: Arc::new(AtomicU64::new(0)),
        }
    }
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        self.tt.read().unwrap().probe(key)
    }
    pub fn store(&self, key: u64, depth: i32, score: i32, bound: Bound, best_move: Move) {
        self.tt.write().unwrap().store(key, depth, score, bound, best_move);
    }
}

pub struct ThreadLocalSearch {
    pub killers: [[Move; 2]; MAX_PLY],
    pub history: [[i32; 64]; 64],
    pub countermove: [[Move; 64]; 64],
    pub excluded_move: Move, // singular-extension excluded best-move (scalar for bounded insertion)
    pub rep_stack: Vec<u64>,
    pub nodes: u64,
    pub go_token: u32,
    pub stopped: bool,
    pub deadline_ms: f64,
    pub max_depth: i32,
}

impl ThreadLocalSearch {
    pub fn new() -> Self {
        ThreadLocalSearch {
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            countermove: [[Move::NULL; 64]; 64],
            rep_stack: Vec::with_capacity(512),
            excluded_move: Move::NULL,
            nodes: 0,
            go_token: 0,
            stopped: false,
            deadline_ms: 0.0,
            max_depth: 32,
        }
    }
}

pub struct SearchState {
    pub shared: SharedSearch,
    pub local: ThreadLocalSearch,
}

impl SearchState {
    pub fn new() -> Self {
        SearchState {
            shared: SharedSearch::new(64),
            local: ThreadLocalSearch::new(),
        }
    }

    pub fn nodes(&self) -> u64 {
        self.local.nodes
    }

    fn time_up(&mut self) -> bool {
        if self.local.nodes % 4096 != 0 {
            return self.local.stopped;
        }
        if STOP_FLAG.load(Ordering::Relaxed) {
            self.local.stopped = true;
        }
        if self.local.go_token != GO_TOKEN.load(Ordering::Relaxed) {
            self.local.stopped = true;
        }
        if now_ms() >= self.local.deadline_ms {
            self.local.stopped = true;
        }
        self.local.stopped
    }

    fn push_rep(&mut self, key: u64) {
        self.local.rep_stack.push(key);
    }
    fn pop_rep(&mut self) {
        self.local.rep_stack.pop();
    }
    fn is_search_repetition(&self, key: u64, halfmove_clock: u16) -> bool {
        let look_back = halfmove_clock as usize;
        let len = self.local.rep_stack.len();
        if len == 0 {
            return false;
        }
        let start = len.saturating_sub(look_back);
        let mut matches = 0;
        for i in (start..len).rev() {
            if self.local.rep_stack[i] == key {
                matches += 1;
                if matches >= 2 {
                    return true;
                }
            }
        }
        false
    }
}

fn is_insufficient_material(pos: &Position) -> bool {
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
fn terminal_score(pos: &Position, state: &SearchState, ply: i32, has_legal_moves: bool) -> Option<i32> {
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

const WINNING_CAPTURE_BASE: i32 = 500_000;
const KILLER_1_SCORE: i32 = 90_000;
const KILLER_2_SCORE: i32 = 89_000;
const COUNTERMOVE_SCORE: i32 = 88_000;
const TT_MOVE_SCORE: i32 = 1_000_000;

fn score_move(
    pos: &Position,
    m: Move,
    tt_move: Move,
    killers: &[Move; 2],
    countermove: Move,
    history: &[[i32; 64]; 64],
) -> i32 {
    if !tt_move.is_null() && m == tt_move {
        return TT_MOVE_SCORE;
    }
    if m.flag().is_capture() {
        // SEE-based ordering: captures that win material (or are at worst
        // equal) are searched before killers/quiets; captures that lose
        // material (SEE < 0) sink below quiet moves instead of getting
        // MVV-LVA's optimistic "biggest victim first" treatment, which is
        // wrong whenever the victim is defended.
        let see_val = crate::movegen::see(pos, m);
        if see_val >= 0 {
            return WINNING_CAPTURE_BASE + see_val;
        }
        return see_val;
    }
    if m == killers[0] {
        return KILLER_1_SCORE;
    }
    if m == killers[1] {
        return KILLER_2_SCORE;
    }
    if !countermove.is_null() && m == countermove {
        return COUNTERMOVE_SCORE;
    }
    history[m.from_sq() as usize][m.to_sq() as usize]
}

/// Scores every move once into a fixed-size (stack) buffer, then does a
/// partial selection sort: each call to `next_move` finds the best-scoring
/// remaining move and swaps it to the front. This means a node that beta-
/// cuts after 1-2 moves never pays for ordering the rest of the list, and
/// there is no per-node heap allocation (no Vec, no sort_by closure).
struct MoveOrderer {
    scores: [i32; MAX_MOVES],
    len: usize,
    next: usize,
}

impl MoveOrderer {
    #[inline]
    fn new(pos: &Position, list: &MoveList, tt_move: Move, killers: &[Move; 2], countermove: Move, history: &[[i32; 64]; 64]) -> Self {
        let mut scores = [0i32; MAX_MOVES];
        let slice = list.as_slice();
        for (i, &m) in slice.iter().enumerate() {
            scores[i] = score_move(pos, m, tt_move, killers, countermove, history);
        }
        MoveOrderer { scores, len: slice.len(), next: 0 }
    }

    /// Selects the highest-scoring move among the unconsumed tail of
    /// `list` and swaps it into position `self.next`, then returns it.
    /// Returns None once every move has been produced.
    #[inline]
    fn next_move(&mut self, list: &mut MoveList) -> Option<(usize, Move)> {
        if self.next >= self.len {
            return None;
        }
        let mut best_i = self.next;
        let mut best_score = self.scores[self.next];
        for i in (self.next + 1)..self.len {
            if self.scores[i] > best_score {
                best_score = self.scores[i];
                best_i = i;
            }
        }
        if best_i != self.next {
            self.scores.swap(self.next, best_i);
            list.moves.swap(self.next, best_i);
        }
        let idx = self.next;
        let m = list.moves[idx];
        self.next += 1;
        Some((idx, m))
    }
}

fn quiescence(pos: &mut Position, state: &mut SearchState, mut alpha: i32, beta: i32, qply: i32, allow_checks: bool, ply: i32) -> i32 {
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

    // Candidate moves: captures always; promotions and (optionally) quiet
    // checks in the last 2 plies of the qsearch window.
    let mut candidates: Vec<Move> = Vec::new();
    for &m in list.as_slice() {
        if m.flag().is_capture() || m.flag().is_promotion() {
            candidates.push(m);
        } else if allow_checks_this_ply {
            // cheap check test: does this move give check?
            pos.make_move(m);
            let gives_check = is_in_check(pos, pos.side_to_move);
            pos.unmake_move(m);
            if gives_check {
                candidates.push(m);
            }
        }
    }
    // Order by SEE (best exchange first) instead of raw victim value, so a
    // "big victim" that's actually defended doesn't get tried before a
    // smaller but genuinely winning capture.
    candidates.sort_by_key(|&m| -crate::movegen::see(pos, m));

    let mut best = stand_pat;
    for m in candidates {
        if !m.flag().is_promotion() && m.flag().is_capture() {
            // SEE-based pruning: a capture that loses material even after
            // the full exchange sequence can't recover once we're already
            // below alpha by more than a small margin, so don't bother
            // searching it. This replaces the old flat
            // "stand_pat + captured_value + 200 <= alpha" delta margin,
            // which could badly overestimate a capture's real value when
            // the victim was defended.
            let see_val = crate::movegen::see(pos, m);
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

fn negamax(pos: &mut Position, state: &mut SearchState, mut depth: i32, mut alpha: i32, beta: i32, ply: i32, prev_move: Move) -> i32 {
    state.local.nodes += 1;
    if state.time_up() {
        return evaluate(pos, None);
    }

    let orig_alpha = alpha;

    // FIX: this previously called generate_qsearch_captures(), a
    // captures-only pseudo-legal generator meant for quiescence search (see
    // its own doc comment in movegen.rs: "Qsearch-specific pseudo-legal
    // capture generator"). Two things broke as a result:
    //   1. terminal_score() below was told has_legal_moves = (list.count >
    //      0), but list only ever contained captures -- so any position
    //      where the side to move has zero captures but DOES have quiet
    //      moves (extremely common in sparse endgames) was misjudged as
    //      checkmate (if in check) or stalemate (if not), independent of
    //      whether real legal moves existed. This is what produced the
    //      phantom "score cp 99999" / mate-in-1 result for a plain king
    //      move in a bare K+R vs K endgame.
    //   2. This same `list` is also what MoveOrderer and the move loop
    //      below actually iterate over -- meaning the entire main search
    //      tree only ever considered captures at every node and every
    //      depth, never quiet moves. Quiet moves were only ever visible via
    //      quiescence()'s own (correct) generate_legal_moves() call at
    //      depth <= 0. This explains high depth/nps with very poor play:
    //      the search was fast and "deep" because it was only ever
    //      expanding a handful of capture moves per node, essentially
    //      blind to normal developing/defensive/positional moves.
    let mut list = MoveList::new();
    crate::movegen::generate_legal_moves(pos, &mut list);

    if let Some(s) = terminal_score(pos, state, ply, list.count > 0) {
        return s;
    }

    if depth <= 0 {
        return quiescence(pos, state, alpha, beta, QUIESCENCE_MAX_PLY, true, ply);
    }

    let mut tt_move = Move::NULL;
    let mut tt_score_for_singular: i32 = 0;
    if let Some(entry) = state.shared.probe(pos.zobrist_key) {
        tt_move = entry.best_move;
        tt_score_for_singular = entry.score;
        tt_move = entry.best_move;
        if entry.depth >= depth {
            match entry.bound {
                Bound::Exact => return entry.score,
                Bound::Lower => {
                    if entry.score > alpha {
                        alpha = entry.score;
                    }
                }
                Bound::Upper => {
                    if entry.score < beta {
                        // effectively lowers beta bound only if it beats current beta
                    }
                }
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    let in_check = is_in_check(pos, pos.side_to_move);
    let is_pv = beta - alpha > 1;

    // Internal iterative reduction: with no TT move to anchor ordering,
    // search this node one ply shallower rather than paying full width for
    // a weakly-ordered node. Compensates for the missing hash move without
    // an explicit separate IID search.
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
        let score = -negamax(pos, state, depth - 1 - 2, -beta, -beta + 1, ply + 1, Move::NULL);
        pos.unmake_null_move(prev_ep);
        if score >= beta {
            return beta;
        }
    }

    // Singular extensions: reduced excluded loop using singular_beta/reduced_depth.
    #[allow(unused_assignments)]
    let mut singular_extension_depth = 0i32;
    if !tt_move.is_null() && depth >= 4 && !in_check && beta.abs() < MATE_SCORE - 1000 {
        let singular_beta_margin = 3 * depth; // depth-scaled singular margin (Stockfish style)
        let singular_beta = tt_score_for_singular - singular_beta_margin;
        let singular_depth = (depth / 2 + 1).max(2);
        let reduced_depth = singular_depth.max(1);
        // Reduced excluded search: search first non-TT alternative at reduced depth.
        // If it fails badly (< singular_beta), TT move is singular -> extend.
        state.local.excluded_move = tt_move;
        let mut singular_failed_badly = true;
        // Search first alternative excluding tt_move with reduced depth / singular_beta window
        for alt in list.as_slice() {
            if *alt != tt_move {
                pos.make_move(*alt);
                let alt_score = -negamax(pos, state, reduced_depth, -singular_beta, -singular_beta, ply + 1, *alt);
                pos.unmake_move(*alt);
                if alt_score >= singular_beta {
                    singular_failed_badly = false; // alternative holds -> not singular
                    break;
                }
                // if alt_score < singular_beta, keep singular_failed_badly = true
                break; // bounded insertion: check only first alternative
            }
        }
        state.local.excluded_move = Move::NULL;
        if singular_failed_badly {
            singular_extension_depth = 1; // positive singular extension
        } else {
            singular_extension_depth = 0; // not singular, no extension
        }
        depth += singular_extension_depth;
    }

    // Static eval of the current node, reused by reverse-futility and
    // frontier futility pruning below. Not computed while in check (a
    // check-evasion static eval is meaningless / can't safely be used to
    // prune since the position is inherently tactical).
    let static_eval = if !in_check { evaluate(pos, Some(&list)) } else { 0 };

    // Reverse futility / static null-move pruning: if we're already far
    // above beta according to the static eval, assume a real search would
    // confirm it and cut immediately.
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
    let mut orderer = MoveOrderer::new(pos, &list, tt_move, &killers_snapshot, countermove, &state.local.history);

    let mut best_score = -MATE_SCORE;
    let mut best_move = Move::NULL;

    while let Some((move_index, m)) = orderer.next_move(&mut list) {
        let is_quiet = !m.flag().is_capture();

        // Late move pruning: at shallow depth, once we've already tried a
        // generous number of quiet moves without a cutoff, stop searching
        // further quiet moves entirely (no reduction, just skip).
        if !is_pv && !in_check && is_quiet && depth <= 8 {
            let lmp_threshold = 4 + (depth * depth) as usize;
            if move_index >= lmp_threshold {
                continue;
            }
        }

        // Futility pruning at frontier nodes: a quiet move can't recover
        // from being this far below alpha at low remaining depth, so don't
        // even make it.
        if !is_pv && !in_check && is_quiet && depth <= 8 && move_index > 0 {
            let margin = 90 + 80 * depth;
            if static_eval + margin <= alpha {
                continue;
            }
        }

        // SEE-based pruning for losing captures: distinct from ordering SEE (search.rs:310-314).
        // Modeled on Reckless/repo/src/search.rs:841-850 (SEE Pruning, depth-scaled threshold)
        // and Stockfish/src/search.cpp: depth-scaled margin then !pos.see_ge(move, -margin).
        if m.flag().is_capture() && !is_pv && depth <= 8 && !in_check && ply > 0 && move_index > 0 {
            let see_margin = 10 * depth + 15;
            let see_val = crate::movegen::see(pos, m);
            if see_val < -see_margin {
                continue;
            }
        }

        // Bounded ProbCut insertion (batch4 design): early-cut noisy alternatives on non-PV nodes.
        // Only triggers when static eval exceeds the interpolated cut threshold and the
        // best TT move is noisy (!tt_move.is_quiet => tt_move.flag().is_capture()).
        // Noisy current move (!is_quiet) is searched at reduced depth; if it exceeds
        // probcut_beta we return a conservative interpolated lower bound.
        if !is_pv && !in_check && depth <= 8 && !is_quiet && !tt_move.is_null() && tt_move.flag().is_capture() {
            let improving = if best_score > alpha { 1 } else { 0 };
            let probcut_beta = beta + 254 - 85 * improving;
            if static_eval >= probcut_beta {
                let reduced_depth = (depth - ((static_eval - probcut_beta) / 319) as i32).max(1);
                pos.make_move(m);
                let gives_check = is_in_check(pos, pos.side_to_move);
                let mut child_depth_reduced = reduced_depth - 1;
                if gives_check && reduced_depth + ply < 40 {
                    child_depth_reduced += 1;
                }
                state.push_rep(pos.zobrist_key);
                let probcut_score = -negamax(pos, state, child_depth_reduced, -beta - 1, -alpha, ply + 1, m);
                state.pop_rep();
                pos.unmake_move(m);
                if probcut_score >= probcut_beta {
                    let interpolated = ((probcut_score as f32) * 0.2695 + (beta as f32) * 0.7305) as i32;
                    return interpolated;
                }
                // If reduced-depth score does not exceed probcut_beta, fall through
                // to the normal full-depth path below (bounded: full-depth preserved).
            }
        }

        pos.make_move(m);
        let gives_check = is_in_check(pos, pos.side_to_move);
        let mut child_depth = depth - 1;
        if gives_check && depth + ply < 40 {
            child_depth += 1;
        }

        let mut score;
        if move_index == 0 {
            state.push_rep(pos.zobrist_key);
            score = -negamax(pos, state, child_depth, -beta, -alpha, ply + 1, m);
            state.pop_rep();
        } else {
            let mut reduced_depth = child_depth;
            let do_lmr = is_quiet && !gives_check && depth >= 3 && move_index >= 3;
            if do_lmr {
                let reduction = if move_index >= 9 { 2 } else { 1 };
                reduced_depth = (child_depth - reduction).max(1);
            }

            state.push_rep(pos.zobrist_key);
            score = -negamax(pos, state, reduced_depth, -alpha - 1, -alpha, ply + 1, m);

            if do_lmr && score > alpha {
                score = -negamax(pos, state, child_depth, -alpha - 1, -alpha, ply + 1, m);
                if score > alpha && score < beta {
                    score = -negamax(pos, state, child_depth, -beta, -alpha, ply + 1, m);
                }
            } else if !do_lmr && score > alpha && score < beta {
                score = -negamax(pos, state, child_depth, -beta, -alpha, ply + 1, m);
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

    let bound = if best_score <= orig_alpha {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    state.shared.store(pos.zobrist_key, depth, best_score, bound, best_move);

    best_score
}

#[allow(dead_code)]
fn net_swing_after_move(pos: &mut Position, mv: Move) -> i32 {
    let mut credit = 0;
    if mv.flag().is_capture() {
        credit += piece_value(mv.captured());
    }
    pos.make_move(mv);

    let mut list = MoveList::new();
    generate_legal_moves(pos, &mut list);
    let mut worst_opponent_gain = 0;

    for &reply in list.as_slice() {
        if !reply.flag().is_capture() {
            continue;
        }
        let gain_raw = piece_value(reply.captured());
        let target_sq = reply.to_sq();
        pos.make_move(reply);

        let mut our_replies = MoveList::new();
        generate_legal_moves(pos, &mut our_replies);
        let mut can_recapture = false;
        let mut recapture_attacker_value = 0;
        for &our_m in our_replies.as_slice() {
            if our_m.flag().is_capture() && our_m.to_sq() == target_sq {
                can_recapture = true;
                recapture_attacker_value = piece_value(reply.piece());
                break;
            }
        }
        pos.unmake_move(reply);

        let net_gain = if can_recapture {
            gain_raw - recapture_attacker_value
        } else {
            gain_raw
        };
        if net_gain > worst_opponent_gain {
            worst_opponent_gain = net_gain;
        }
    }

    pos.unmake_move(mv);
    credit - worst_opponent_gain
}

#[allow(dead_code)]
pub struct RootResult {
    pub best_move: Move,
    pub score: i32,
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

    let mut root_list = MoveList::new();
    generate_legal_moves(pos, &mut root_list);
    if root_list.count == 0 {
        return Move::NULL;
    }

    let mut best_move = root_list.as_slice()[0];
    let mut best_score = -MATE_SCORE;
    let mut prev_best_move = Move::NULL;
    let start_ms = now_ms();

    for depth in 1..=limits.max_depth {
        if state.local.stopped || STOP_FLAG.load(Ordering::Relaxed) || go_token != GO_TOKEN.load(Ordering::Relaxed) {
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

            'aspiration: loop {
            depth_best_score = -MATE_SCORE;
            let mut _unused_root_order = MoveOrderer::new(pos, &root_list, prev_best_move, &root_killers, Move::NULL, &state.local.history);
            // Bounded batch 2: flat root-level split (not two-phase best-guess-first).
            // Each root move evaluated in rayon::scope; threads share SharedSearch.tt (Arc<RwLock>),
            // each gets its own ThreadLocalSearch.
            let root_moves = root_list.as_slice().to_vec();
            let shared_clone = state.shared.clone();
            let results_arc = std::sync::Arc::new(std::sync::Mutex::new(vec![(Move::NULL, -MATE_SCORE); root_moves.len()]));
            let base_pos_clone = (*pos).clone();
            #[cfg(feature = "parallel-search")]
            {
                use rayon::prelude::*;
                rayon::scope(|s| {
                    for (i, m) in root_moves.iter().enumerate() {
                        if STOP_FLAG.load(Ordering::Relaxed) || state.local.stopped {
                            break;
                        }
                        let shared = shared_clone.clone();
                        let results_ref = results_arc.clone();
                        let pos_for_this = base_pos_clone.clone();
                        s.spawn(move |_| {
                            let shared_for_aggregate = shared.clone();
                            let mut local_state = SearchState {
                                shared: shared,
                                local: ThreadLocalSearch::new(),
                            };
                            let mut temp_pos = pos_for_this.clone();
                            temp_pos.make_move(*m);
                            let _z = temp_pos.zobrist_key;
                            let score = -negamax(&mut temp_pos, &mut local_state, depth, -beta, -alpha, 1, *m);
                            temp_pos.unmake_move(*m);
                            results_ref.lock().unwrap()[i] = (*m, score);
                            shared_for_aggregate.nodes_aggregate.fetch_add(local_state.local.nodes, Ordering::Relaxed);
                        });
                    }
                });
            }
            let results = std::sync::Arc::try_unwrap(results_arc).unwrap().into_inner().unwrap();
            // Aggregate parallel results back into sequential best tracking.
            for (m, score) in results {
                if score > depth_best_score {
                    depth_best_score = score;
                    depth_best_move = m;
                }
                if depth_best_score > alpha {
                    alpha = depth_best_score;
                }
            }

            if state.local.stopped {
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

        if !state.local.stopped {
            best_score = depth_best_score;
            best_move = depth_best_move;
            prev_best_move = best_move;

            let total_nodes = state.local.nodes + state.shared.nodes_aggregate.load(Ordering::Relaxed);
            let elapsed_ms = (now_ms() - start_ms).max(1.0);
            let nps = (total_nodes as f64 / (elapsed_ms / 1000.0)) as u64;
            if best_score.abs() >= MATE_SCORE - 1000 {
                let mate_in = ((MATE_SCORE - best_score.abs() + 1) / 2) * best_score.signum();
                send(&format!(
                    "info depth {} score mate {} nodes {} nps {} time {} pv {}",
                    depth, mate_in, total_nodes, nps, elapsed_ms as u64, best_move.to_uci()
                ));
            } else {
                send(&format!(
                    "info depth {} score cp {} nodes {} nps {} time {} pv {}",
                    depth, best_score, total_nodes, nps, elapsed_ms as u64, best_move.to_uci()
                ));
            }
        } else {
            break;
        }
    }

    // NOTE: a "root-level blunder guard" previously lived here, re-checking
    // best_move with a shallow 1-ply capture-only heuristic
    // (net_swing_after_move) and silently substituting a different root
    // move -- picked in raw move-generation order, not by search quality --
    // whenever that heuristic didn't like the result. It had no way to see
    // mating nets, promotion follow-up, or positional compensation that the
    // full alpha-beta search above already accounts for to full depth, so
    // it was strictly worse than the search itself: it vetoed sound
    // sacrifices/promotions and replaced them with whatever unrelated move
    // happened to come first and pass its loose threshold. This is why
    // `bestmove` was disagreeing with the `pv` printed during the loop.
    // Removed; trust the search's own best_move.

    best_move
}
#[cfg(feature = "parallel-search")]
pub fn parallel_search_stub() {
    use rayon::prelude::*;
    // Bounded batch-2: root-level split only (no recursive/interior split).
    // Modeled on Reckless/repo/src/threadpool.rs: rayon::scope over root moves,
    // sharing SharedSearch.tt (RwLock) across threads.
    // Each thread creates its own ThreadLocalSearch.
}
// WASM excluded (single-threaded spawn_local). multi-threaded root search: rayon::scope over root moves with per-thread SearchState clones.
// WASM build excludes this (single-threaded spawn_local). Feature enabled with: cargo build --features parallel-search
#[cfg(feature = "parallel-search")]
#[allow(unused_imports)]
use rayon::prelude::*;
