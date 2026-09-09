use crate::position::{Color, Position};

include!(concat!(env!("OUT_DIR"), "/opening_book_data.rs"));

/// wasm32-unknown-unknown has no OS clock: std::time::SystemTime::now()
/// panics outright there ("time not implemented on this platform"), it
/// doesn't just return a wrong value. js_sys::Date is what actually reaches
/// the browser's clock via JS interop, so we route through that on wasm32
/// and keep the normal std path everywhere else. (This mirrors the same
/// helper in uci.rs -- kept as a local copy here to avoid introducing a
/// cross-module dependency just for one function.)
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

// Tiny xorshift64* PRNG seeded from the current time, so we don't need
// `getrandom` unless the caller prefers that route (Cargo.toml already
// pulls in `getrandom`'s js feature as a fallback/alternative; this local
// RNG avoids the extra JS boundary hop for the common case).
pub struct BookRng(u64);

impl BookRng {
    pub fn new_seeded() -> Self {
        let seed = (now_ms() as u64) | 1;
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
pub fn probe_book(pos: &Position, rng: &mut BookRng) -> Option<String> {
    let key = pos.book_key();
    let candidates = OPENING_BOOK.get(key.as_str())?;
    if candidates.is_empty() {
        return None;
    }
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
