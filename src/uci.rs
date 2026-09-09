use crate::position::*;
use crate::search::{iterative_deepening, SearchLimits, SearchState, GO_TOKEN, STOP_FLAG};
use crate::book;
use std::sync::atomic::Ordering;

/// wasm32-unknown-unknown has no OS clock: std::time::SystemTime::now() and
/// std::time::Instant::now() both panic outright there (not just return a
/// wrong value) with "time not implemented on this platform". js_sys::Date
/// is what actually reaches the browser clock via JS interop, so we route
/// through that on wasm32 and keep the normal std path everywhere else.
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

pub struct EngineState {
    pub pos: Position,
    pub search: SearchState,
    pub book_rng: book::BookRng,
}

impl EngineState {
    pub fn new() -> Self {
        EngineState {
            pos: Position::startpos(),
            search: SearchState::new(),
            book_rng: book::BookRng::new_seeded(),
        }
    }
}

pub fn handle_uci_line<F: FnMut(&str)>(state: &mut EngineState, line: &str, mut send: F) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    let mut parts = line.split_whitespace();
    let cmd = match parts.next() {
        Some(c) => c,
        None => return,
    };

    match cmd {
        "uci" => {
            send("id name WizardBot");
            send("id author Merlin2LmmL");
            send("uciok");
            send(&format!("info string THREAD_DEBUG threading={:?} cores={:?}", cfg!(feature = "parallel-search"), std::thread::available_parallelism()));
        }
        "isready" => {
            send("readyok");
        }
        "ucinewgame" => {
            GO_TOKEN.fetch_add(1, Ordering::SeqCst);
            state.pos = Position::startpos();
            state.search = SearchState::new();
        }
        "position" => {
            GO_TOKEN.fetch_add(1, Ordering::SeqCst);
            handle_position(state, &mut parts);
        }
        "go" => {
            handle_go(state, &mut parts, send);
        }
        "stop" => {
            STOP_FLAG.store(true, Ordering::SeqCst);
            GO_TOKEN.fetch_add(1, Ordering::SeqCst);
        }
        "quit" => {
            // no-op, matches original
        }
        _ if cmd.starts_with("setoption") || line.starts_with("setoption") => {
            // no-op
        }
        _ => {
            // unknown line: ignore
        }
    }
}

fn handle_position<'a, I: Iterator<Item = &'a str>>(state: &mut EngineState, parts: &mut I) {
    let mut tokens: Vec<&str> = parts.collect();
    if tokens.is_empty() {
        return;
    }
    let mut idx = 0;
    let new_pos = if tokens[idx] == "startpos" {
        idx += 1;
        Position::startpos()
    } else if tokens[idx] == "fen" {
        idx += 1;
        let mut fen_parts = Vec::new();
        while idx < tokens.len() && tokens[idx] != "moves" {
            fen_parts.push(tokens[idx]);
            idx += 1;
        }
        let fen = fen_parts.join(" ");
        match Position::from_fen(&fen) {
            Ok(p) => p,
            Err(_) => return,
        }
    } else {
        return;
    };

    state.pos = new_pos;

    if idx < tokens.len() && tokens[idx] == "moves" {
        idx += 1;
        while idx < tokens.len() {
            let uci = tokens[idx];
            let mut list = MoveList::new();
            crate::movegen::generate_legal_moves(&state.pos, &mut list);
            let mut applied = false;
            for &m in list.as_slice() {
                if m.to_uci() == uci {
                    state.pos.make_move(m);
                    applied = true;
                    break;
                }
            }
            if !applied {
                break;
            }
            idx += 1;
        }
    }
    let _ = tokens.drain(..0); // silence unused mut warning in some configs
}

fn handle_go<'a, I: Iterator<Item = &'a str>, F: FnMut(&str)>(state: &mut EngineState, parts: &mut I, mut send: F) {
    STOP_FLAG.store(false, Ordering::SeqCst);
    let token = GO_TOKEN.fetch_add(1, Ordering::SeqCst) + 1;

    let mut max_depth = 32i32;
    let mut movetime_ms: Option<f64> = None;
    let mut infinite = false;
    let mut wtime_ms: Option<f64> = None;
    let mut btime_ms: Option<f64> = None;
    let mut winc_ms: Option<f64> = None;
    let mut binc_ms: Option<f64> = None;

    let tokens: Vec<&str> = parts.collect();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                if let Some(v) = tokens.get(i + 1).and_then(|s| s.parse::<i32>().ok()) {
                    max_depth = v;
                }
                i += 2;
            }
            "movetime" => {
                if let Some(v) = tokens.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                    movetime_ms = Some(v);
                }
                i += 2;
            }
            "infinite" => {
                infinite = true;
                i += 1;
            }
            "wtime" => {
                if let Some(v) = tokens.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                    wtime_ms = Some(v);
                }
                i += 2;
            }
            "btime" => {
                if let Some(v) = tokens.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                    btime_ms = Some(v);
                }
                i += 2;
            }
            "winc" => {
                if let Some(v) = tokens.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                    winc_ms = Some(v);
                }
                i += 2;
            }
            "binc" => {
                if let Some(v) = tokens.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                    binc_ms = Some(v);
                }
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    const HARD_CAP_MS: f64 = 20.0 * 60.0 * 1000.0;
    let now = now_ms();
    let deadline_ms = if infinite {
        now + HARD_CAP_MS
    } else if let Some(mt) = movetime_ms {
        now + mt.min(HARD_CAP_MS)
    } else if movetime_ms.is_none() && !infinite && max_depth == 32 {
        // Fallback: use time controls only when movetime/infinite/depth not explicitly overriding.
        let (time_left, inc) = if state.pos.side_to_move == Color::White {
            (wtime_ms.unwrap_or(0.0), winc_ms.unwrap_or(0.0))
        } else {
            (btime_ms.unwrap_or(0.0), binc_ms.unwrap_or(0.0))
        };
        if time_left > 0.0 {
            let budget = (time_left / 20.0) + (inc * 0.8) - 50.0;
            let budget = budget.max(50.0).min(HARD_CAP_MS);
            now + budget
        } else {
            now + 8000.0_f64.min(HARD_CAP_MS)
        }
    } else {
        now + 8000.0_f64.min(HARD_CAP_MS)
    };

    // Opening book: bypass search entirely.
    if book::should_probe_book(&state.pos) {
        if let Some(uci) = book::probe_book(&state.pos, &mut state.book_rng) {
            send("info string book move");
            send(&format!("bestmove {}", uci));
            return;
        }
    }

    let limits = SearchLimits { max_depth, deadline_ms };
    let best_move = iterative_deepening(&mut state.pos, &mut state.search, limits, token, |line| send(line));

    if best_move.is_null() {
        send("bestmove (none)");
    } else {
        send(&format!("bestmove {}", best_move.to_uci()));
    }
}
