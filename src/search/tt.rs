use crate::position::Move;
use std::sync::atomic::{AtomicU64, AtomicU8};
use std::sync::{Arc, RwLock};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Copy, Clone)]
pub struct TTEntry {
    pub key: u64,
    // Sentinel: depth < 0 means "slot empty". Real search depths passed to
    // store() are always >= 0, so this needs no extra discriminant/padding
    // the way `Option<TTEntry>` did (u64 key has no spare-bit niche).
    pub depth: i32,
    pub score: i32,
    pub bound: Bound,
    pub best_move: Move,
    pub generation: u8,
}

impl TTEntry {
    const EMPTY: TTEntry = TTEntry {
        key: 0,
        depth: -1,
        score: 0,
        bound: Bound::Exact,
        best_move: Move::NULL,
        generation: 0,
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

// BUGFIX: this table used to be wrapped in a single, table-wide
// `RwLock<TranspositionTable>` (see SharedSearch below, previously
// `Arc<RwLock<TranspositionTable>>`). With root-parallel search that means
// *every* node, on *every* one of the ~dozens of root-move threads, takes
// either a read or a write lock on the exact same mutex to probe/store the
// TT -- a store anywhere serializes every other thread's probe or store
// anywhere else in the table, table-wide, regardless of whether they touch
// the same bucket. That's a real scalability ceiling independent of the
// check-extension bug (see MAX_LINE_EXTENSIONS in mod.rs) and a plausible
// contributor to the nps swings across otherwise-similar positions (more
// legal root moves -> more threads -> more contention on the one lock).
//
// Fix: lock per-bucket instead of table-wide. Two threads only actually
// contend if their positions hash into the *same* bucket, which with
// thousands of buckets is rare. `generation` moves to an AtomicU8 so
// `new_generation()` needs no lock at all.
pub struct TranspositionTable {
    buckets: Vec<RwLock<TTBucket>>,
    mask: usize,
    generation: AtomicU8,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let bucket_bytes = std::mem::size_of::<TTBucket>().max(1);
        let mut bucket_count = (mb * 1024 * 1024 / bucket_bytes).next_power_of_two();
        bucket_count = bucket_count.max(1 << 12);
        let mut buckets = Vec::with_capacity(bucket_count);
        buckets.resize_with(bucket_count, || RwLock::new([TTEntry::EMPTY; TT_BUCKET_SIZE]));
        TranspositionTable {
            buckets,
            mask: bucket_count - 1,
            generation: AtomicU8::new(0),
        }
    }
    pub fn new_generation(&self) {
        self.generation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[inline]
    fn bucket_idx(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }
    pub(crate) fn probe(&self, key: u64) -> Option<TTEntry> {
        let bucket = self.buckets[self.bucket_idx(key)].read().unwrap();
        for e in bucket.iter() {
            if !e.is_empty() && e.key == key {
                return Some(*e);
            }
        }
        None
    }
    pub(crate) fn store(&self, key: u64, depth: i32, score: i32, bound: Bound, best_move: Move) {
        let idx = self.bucket_idx(key);
        let mut bucket = self.buckets[idx].write().unwrap();
        let gen = self.generation.load(std::sync::atomic::Ordering::Relaxed);

        // Same position already in this bucket: refresh in place (only
        // overwrite with a shallower search if we have nothing better).
        for slot in bucket.iter_mut() {
            if !slot.is_empty() && slot.key == key {
                if depth >= slot.depth || bound == Bound::Exact || slot.generation != gen {
                    *slot = TTEntry { key, depth, score, bound, best_move, generation: gen };
                }
                return;
            }
        }
        // Free slot in the bucket.
        for slot in bucket.iter_mut() {
            if slot.is_empty() {
                *slot = TTEntry { key, depth, score, bound, best_move, generation: gen };
                return;
            }
        }
        // Bucket full of other positions: evict the shallowest entry.
        let mut worst_i = 0;
        let mut worst_value = bucket[0].depth as i32 - 8 * gen.wrapping_sub(bucket[0].generation) as i32;
        for (i, slot) in bucket.iter().enumerate().skip(1) {
            let value = slot.depth as i32 - 8 * gen.wrapping_sub(slot.generation) as i32;
            if value < worst_value {
                worst_value = value;
                worst_i = i;
            }
        }
        bucket[worst_i] = TTEntry { key, depth, score, bound, best_move, generation: gen };
    }
}

#[derive(Clone)]
pub struct SharedSearch {
    pub tt: Arc<TranspositionTable>,
    pub nodes_aggregate: Arc<AtomicU64>,
}

impl SharedSearch {
    pub fn new(mb: usize) -> Self {
        SharedSearch {
            tt: Arc::new(TranspositionTable::new(mb)),
            nodes_aggregate: Arc::new(AtomicU64::new(0)),
        }
    }
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        self.tt.probe(key)
    }
    pub fn store(&self, key: u64, depth: i32, score: i32, bound: Bound, best_move: Move) {
        self.tt.store(key, depth, score, bound, best_move);
    }
    pub fn new_generation(&self) {
        self.tt.new_generation();
    }
}
