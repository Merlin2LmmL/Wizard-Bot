pub mod nnue;
pub mod bitboard;
pub mod book;
pub mod eval;
pub mod movegen;
pub mod position;
pub mod search;
pub mod uci;
pub mod zobrist;

pub fn run_perft_suite() {
    use position::Position;
    use movegen::perft;

    let cases: &[(&str, &[u64])] = &[
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &[20, 400, 8902, 197281, 4865609]),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", &[48, 2039, 97862]),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2812, 43238]),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &[6, 264, 9467]),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[44, 1486, 62379]),
    ];

    let mut all_ok = true;
    for (fen, expected) in cases {
        let mut pos = Position::from_fen(fen).unwrap();
        for (i, &exp) in expected.iter().enumerate() {
            let depth = (i + 1) as u32;
            let t0 = std::time::Instant::now();
            let nodes = perft(&mut pos, depth);
            let dt = t0.elapsed();
            let ok = nodes == exp;
            all_ok &= ok;
            println!(
                "{} depth={} nodes={} expected={} {} ({:?})",
                fen, depth, nodes, exp, if ok { "OK" } else { "MISMATCH" }, dt
            );
        }
    }
    if !all_ok {
        std::process::exit(1);
    }
    println!("ALL PERFT OK");
}

pub fn run_search_smoke() {
    use position::Position;
    use search::{iterative_deepening, SearchLimits, SearchState};

    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", // simple rook endgame
        "6k1/8/6K1/8/8/8/8/R7 w - - 0 1", // mate in a few
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", // fool's mate threat
    ];

    // Known mate puzzles: (fen, expected best move in UCI, mate distance hint)
    let mates = [
        ("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", "e1e8"), // mate in 1
        ("1k6/1p6/8/8/8/8/6R1/2R3K1 w - - 0 1", "g2g8"), // mate in 1 (back rank)
        ("r1b1kb1r/pppp1ppp/2n2n2/1B2p2q/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5", "f3g5"), // legal-ish tactical (not forced mate, just sanity it doesn't crash)
    ];
    for (fen, expect) in mates {
        let mut pos = Position::from_fen(fen).unwrap();
        let mut state = SearchState::new();
        let token = search::GO_TOKEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let limits = SearchLimits { max_depth: 8, deadline_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as f64 + 3000.0 };
        let bm = iterative_deepening(&mut pos, &mut state, limits, token, |_line| {});
        let uci = if bm.is_null() { "none".to_string() } else { bm.to_uci() };
        println!("MATE-CHECK {} -> {} (expected {}) nodes={}", fen, uci, expect, state.nodes());
    }

    for fen in fens {
        let mut pos = Position::from_fen(fen).unwrap();
        let mut state = SearchState::new();
        let token = search::GO_TOKEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let limits = SearchLimits { max_depth: 6, deadline_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as f64 + 3000.0 };
        let t0 = std::time::Instant::now();
        let bm = iterative_deepening(&mut pos, &mut state, limits, token, |_line| {});
        println!("{} -> best={} nodes={} time={:?}", fen, if bm.is_null() { "none".to_string() } else { bm.to_uci() }, state.nodes(), t0.elapsed());
    }
}

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use uci::EngineState;

#[cfg(not(target_arch = "wasm32"))]
pub fn native_init() -> uci::EngineState { uci::EngineState::new() }

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static SEND_CB: RefCell<Option<js_sys::Function>> = RefCell::new(None);
    static ENGINE: RefCell<Option<EngineState>> = RefCell::new(None);
}
#[cfg(target_arch = "wasm32")]
fn send_line(line: &str) {
    SEND_CB.with(|cb| {
        if let Some(f) = cb.borrow().as_ref() {
            let _ = f.call1(&JsValue::NULL, &JsValue::from_str(line));
        }
    });
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn init(send_cb: js_sys::Function) {
    console_error_panic_hook::set_once();
    SEND_CB.with(|cb| { *cb.borrow_mut() = Some(send_cb); });
    ENGINE.with(|e| { *e.borrow_mut() = Some(EngineState::new()); });

    // eval.rs owns NNUE loading (it's the only real consumer), and embeds
    // the .nnue bytes at compile time via include_bytes! since wasm32 has
    // no filesystem. We just trigger the load here so any failure is
    // reported immediately over UCI instead of surfacing silently on the
    // first `go` command -- and instead of the old behavior of an
    // unhandled panic (which compiles to an `unreachable` trap and kills
    // the whole wasm instance before it can answer any UCI command at all).
    match crate::eval::warm_up_nnue() {
        Ok(()) => send_line("info string nnue=loaded"),
        Err(e) => send_line(&format!("info string nnue=failed error={e}")),
    }
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn handle_line(line: &str) {
    let line = line.to_string();
    let cmd_word = line.split_whitespace().next().unwrap_or("").to_string();
    if cmd_word == "go" {
        ENGINE.with(|e| {
            let mut engine_ref = e.borrow_mut();
            if let Some(engine) = engine_ref.as_mut() {
                uci::handle_uci_line(engine, &line, |out| send_line(out));
            }
        });
    } else if cmd_word == "position" || cmd_word == "isready" || cmd_word == "uci" || cmd_word == "quit" || cmd_word == "stop" {
        ENGINE.with(|e| {
            let mut engine_ref = e.borrow_mut();
            if let Some(engine) = engine_ref.as_mut() {
                uci::handle_uci_line(engine, &line, |out| send_line(out));
            } else if cmd_word == "uci" {
                send_line("id name WizardBot");
                send_line("uciok");
            }
        });
    }
}
