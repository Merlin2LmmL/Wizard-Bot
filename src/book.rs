use crate::position::{Color, Position};

include!(concat!(env!("OUT_DIR"), "/opening_book_data.rs"));

// Tiny xorshift64* PRNG seeded from the current time, so we don't need
// `getrandom` unless the caller prefers that route (Cargo.toml already
// pulls in `getrandom`'s js feature as a fallback/alternative; this local
// RNG avoids the extra JS boundary hop for the common case).
pub struct BookRng(u64);

impl BookRng {
    pub fn new_seeded(pos: &Position) -> Self {
        let key_str = pos.book_key();
        let mut seed: u64 = 1;
        for b in key_str.bytes() {
            seed = seed.wrapping_mul(31).wrapping_add(b as u64);
        }
        seed |= 1;
        BookRng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Only consult while ply < 16.
pub fn should_probe_book(pos: &Position) -> bool {
    pos.ply() < 16
}

/// Weighted-random pick with an 8% floor, matched against actual legal
/// moves. Returns the chosen UCI move string, or None if no book entry /
/// no match against legal moves.
/// Multi-candidate preference via shallow static eval (fixed depth 4).
/// For single-candidate or when only one survives the floor, fall back
/// to the existing position-derived seed / weighted-random path.
fn pick_by_shallow_eval(pos: &Position, candidates: &[(&str, u32)]) -> Option<String> {
    let mut best_uci: Option<&str> = None;
    let mut best_score: i32 = i32::MIN;
    let mut best_weight: u32 = 0;

    for &(uci_str, w) in candidates {
        // Find matching legal move for this UCI.
        let mut list = crate::position::MoveList::new();
        crate::movegen::generate_legal_moves(pos, &mut list);
        let mut move_for_uci: Option<crate::position::Move> = None;
        for &m in list.as_slice() {
            if m.to_uci() == uci_str {
                move_for_uci = Some(m);
                break;
            }
        }
        let m = move_for_uci?;

        // Full shallow search (fixed depth 4 total = book ply + depth-3 deeper),
        // but root-level restricted to this single candidate (already applied).
        // Then search continues with all legal moves (opponent + deeper plies).
        let mut temp_pos = pos.clone();
        temp_pos.make_move(m);
        let mut temp_state = crate::search::SearchState::new();
        let limits = crate::search::SearchLimits {
            max_depth: 7, // cap book-move selection search to 8 plies total (1 book + 7 deeper)
            deadline_ms: 500.0, // 500ms safeguard
        };
        let mut captured_score: i32 = 0;
        let captured_best: crate::position::Move = crate::search::iterative_deepening(
            &mut temp_pos,
            &mut temp_state,
            limits,
            0,
            |line: &str| {
                // Capture final depth-4 info score from emitted line.
                if line.contains("score cp ") {
                    if let Some(pos) = line.find("score cp ") {
                        if let Ok(s) = line[pos + 10..].split_whitespace().next().unwrap_or("0").parse::<i32>() {
                            captured_score = s;
                        }
                    }
                } else if line.contains("score mate ") {
                    if let Some(pos) = line.find("score mate ") {
                        if let Ok(s) = line[pos + 11..].split_whitespace().next().unwrap_or("0").parse::<i32>() {
                            captured_score = s;
                        }
                    }
                }
            },
        );
        let score = captured_score;
        if best_uci.is_none() || score > best_score || (score == best_score && w > best_weight) {
            // Tie-break: if scores within 20 cp, prefer higher weight (ECO frequency).
            // Only apply weight preference when score is close.
            let within_margin = best_uci.is_some() && (best_score - score).abs() <= 20;
            if !best_uci.is_some()
                || score > best_score
                || (within_margin && w > best_weight)
            {
                best_uci = Some(uci_str);
                best_score = score;
                best_weight = w;
            }
        }
    }
    best_uci.map(|s| s.to_string())
}

pub fn probe_book(pos: &Position, rng: &mut BookRng) -> Option<String> {
    let key = pos.book_key();
    let candidates = OPENING_BOOK.get(key.as_str())?;
    if candidates.is_empty() {
        return None;
    }

    // Multi-candidate preference via shallow static eval (fixed depth 4 concept,
    // approximated by static eval for speed while comparing candidates).
    // Single-candidate falls back to existing position-derived seed / weight path.
    if candidates.len() > 1 {
        let result = pick_by_shallow_eval(pos, candidates);
        if result.is_some() {
            // Validate against legal moves before returning.
            let mut list = crate::position::MoveList::new();
            crate::movegen::generate_legal_moves(pos, &mut list);
            let chosen_str = result.unwrap();
            for &m in list.as_slice() {
                if m.to_uci() == chosen_str {
                    return Some(chosen_str);
                }
            }
            // If chosen shallow-eval move isn't legal, fall through to weight-based.
        }
    }

    // Single-candidate (or multi-candidate shallow-eval fallback):
    // existing weight-based RNG with 8% floor + fix B position-derived seed.
    let max_weight = candidates.iter().map(|(_, w)| *w).max().unwrap_or(0);
    if max_weight == 0 {
        return None;
    }
    let floor = (max_weight as f64 * 0.08).ceil() as u32;
    let survivors: Vec<&(&str, u32)> = candidates.iter().filter(|(_, w)| *w >= floor).collect();
    if survivors.is_empty() {
        return None;
    }
    let total: u64 = survivors.iter().map(|(_, w)| *w as u64).sum();
    if total == 0 {
        return None;
    }
    let mut r = rng.next_u64() % total;
    let mut chosen: Option<&str> = None;
    for (uci, w) in survivors {
        if r < *w as u64 {
            chosen = Some(uci);
            break;
        }
        r -= *w as u64;
    }
    let chosen = chosen?;

    // Validate against actual legal moves (from+to+promotion match).
    let mut list = crate::position::MoveList::new();
    crate::movegen::generate_legal_moves(pos, &mut list);
    for &m in list.as_slice() {
        if m.to_uci() == chosen {
            return Some(chosen.to_string());
        }
    }
    None
}

#[allow(dead_code)]
fn _color_unused(_c: Color) {}

/// Public helper for recursive book filter in search: returns the list of
/// candidate UCI strings for the current position, or empty if out of range / no entry.
pub fn get_book_candidates(pos: &Position) -> Vec<&'static str> {
    if !should_probe_book(pos) {
        return vec![];
    }
    let key = pos.book_key();
    if let Some(slice) = OPENING_BOOK.get(key.as_str()) {
        if !slice.is_empty() {
            return slice.iter().map(|(u, _w)| *u).collect();
        }
    }
    vec![]
}
