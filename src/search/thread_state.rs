use super::tt::SharedSearch;
use super::{now_ms, GO_TOKEN, MAX_PLY, STOP_FLAG};
use crate::position::Move;
use std::sync::atomic::Ordering;

pub struct SearchLimits {
    pub max_depth: i32,
    pub deadline_ms: f64, // absolute JS Date.now() timestamp
}

pub struct ThreadLocalSearch {
    pub killers: [[Move; 2]; MAX_PLY],
    pub history: [[i32; 64]; 64],
    pub countermove: [[Move; 64]; 64],
    // Replaced mutable `excluded_move` with explicit `excluded: Move` param
    // on negamax() (see negamax.rs) — avoids mutable side-channel.
    pub rep_stack: Vec<u64>,
    pub nodes: u64,
    pub go_token: u32,
    pub stopped: bool,
    pub deadline_ms: f64,
    pub max_depth: i32,
    // TEMPDEBUG: diagnostic timing counters (ns accumulated per-thread)
    pub time_in_book_filter_ns: u64,
    pub time_in_tt_probe_ns: u64,
    pub time_in_singular_ext_ns: u64,
    pub time_in_static_eval_ns: u64,
    pub time_in_probcut_ns: u64,
    pub time_in_check_detect_ns: u64,
    pub time_in_see_ns: u64,
}

impl ThreadLocalSearch {
    pub fn new() -> Self {
        ThreadLocalSearch {
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            countermove: [[Move::NULL; 64]; 64],
            rep_stack: Vec::with_capacity(512),
            nodes: 0,
            go_token: 0,
            stopped: false,
            deadline_ms: 0.0,
            max_depth: 32,
            time_in_book_filter_ns: 0,
            time_in_tt_probe_ns: 0,
            time_in_singular_ext_ns: 0,
            time_in_static_eval_ns: 0,
            time_in_probcut_ns: 0,
            time_in_check_detect_ns: 0,
            time_in_see_ns: 0,
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

    pub(crate) fn time_up(&mut self) -> bool {
        // Check deadline on node 1 (not just every 4096) so late-started threads
        // discover timeout immediately before doing real work.
        if self.local.nodes % 4096 != 0 && self.local.nodes != 1 {
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

    pub(crate) fn push_rep(&mut self, key: u64) {
        self.local.rep_stack.push(key);
    }
    pub(crate) fn pop_rep(&mut self) {
        self.local.rep_stack.pop();
    }
    pub(crate) fn is_search_repetition(&self, key: u64, halfmove_clock: u16) -> bool {
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
