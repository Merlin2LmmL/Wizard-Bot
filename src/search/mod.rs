//! Search module entry point.
//!
//! Split out of a single ~1200-line `search.rs` into focused submodules:
//! - `tt`             - transposition table + cross-thread shared search state
//! - `thread_state`   - per-thread search state (killers/history/etc.) and SearchLimits
//! - `ordering`        - move scoring / selection-sort move ordering
//! - `helpers`         - small shared helpers (terminal detection, mate-score TT adjustment)
//! - `quiescence`       - quiescence search
//! - `negamax`          - the main alpha-beta negamax search
//! - `iterative_deepening` - outer iterative-deepening/aspiration driver, root
//!    move parallelization, opening-book filtering, and PV extraction
//!
//! Public API is unchanged from the old single-file module: everything that
//! used to be reachable as `crate::search::X` still is, via the re-exports
//! below. If your crate root has `mod search;` pointing at the old
//! `src/search.rs`, just replace that file with this `src/search/` directory
//! (Rust treats `search/mod.rs` as equivalent to `search.rs`) -- no changes
//! needed elsewhere.

mod helpers;
mod iterative_deepening;
mod negamax;
mod ordering;
mod quiescence;
mod thread_state;
mod tt;

pub use iterative_deepening::{iterative_deepening, RootResult};
pub use thread_state::{SearchLimits, SearchState, ThreadLocalSearch};
pub use tt::{Bound, SharedSearch, TTEntry, TranspositionTable};

use std::sync::atomic::{AtomicBool, AtomicU32};

pub static STOP_FLAG: AtomicBool = AtomicBool::new(false);
pub static GO_TOKEN: AtomicU32 = AtomicU32::new(0);

pub(crate) const QUIESCENCE_MAX_PLY: i32 = 6;
pub(crate) const MAX_PLY: usize = 128;
pub(crate) const MAX_MATE_PLY: i32 = 1000;

// Cap total line extensions to bound worst-case depth collapse independently
// of remaining search depth.
pub(crate) const MAX_LINE_EXTENSIONS: i32 = 8;

/// wasm32-unknown-unknown has no OS clock: std::time::SystemTime::now()
/// panics outright there ("time not implemented on this platform"), it
/// doesn't just return a wrong value. js_sys::Date is what actually reaches
/// the browser's clock via JS interop, so we route through that on wasm32
/// and keep the normal std path everywhere else. (Same helper as in
/// uci.rs/book.rs -- kept as a local copy here rather than a shared module
/// to avoid introducing a cross-module dependency for one function.)
pub(crate) fn now_ms() -> f64 {
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
