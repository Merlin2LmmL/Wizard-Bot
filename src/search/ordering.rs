use crate::position::*;

const WINNING_CAPTURE_BASE: i32 = 500_000;
const KILLER_1_SCORE: i32 = 90_000;
const KILLER_2_SCORE: i32 = 89_000;
const COUNTERMOVE_SCORE: i32 = 88_000;
const TT_MOVE_SCORE: i32 = 1_000_000;
// Checking moves score above ordinary quiets but below killers/countermoves.
const CHECK_SCORE: i32 = 50_000;

pub(crate) fn score_move(
    pos: &Position,
    m: Move,
    tt_move: Move,
    killers: &[Move; 2],
    countermove: Move,
    history: &[[i32; 64]; 64],
    check_ctx: &crate::movegen::CheckingSquares,
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
    // O(1) check lookup via per-node context (see CheckingSquares).
    if crate::movegen::gives_check(pos, m, check_ctx) {
        return CHECK_SCORE;
    }
    history[m.from_sq() as usize][m.to_sq() as usize]
}

/// Scores every move once into a fixed-size (stack) buffer, then does a
/// partial selection sort: each call to `next_move` finds the best-scoring
/// remaining move and swaps it to the front. This means a node that beta-
/// cuts after 1-2 moves never pays for ordering the rest of the list, and
/// there is no per-node heap allocation (no Vec, no sort_by closure).
pub(crate) struct MoveOrderer {
    scores: [i32; MAX_MOVES],
    len: usize,
    next: usize,
}

impl MoveOrderer {
    #[inline]
    pub(crate) fn new(
        pos: &Position,
        list: &MoveList,
        tt_move: Move,
        killers: &[Move; 2],
        countermove: Move,
        history: &[[i32; 64]; 64],
        check_ctx: &crate::movegen::CheckingSquares,
    ) -> Self {
        let mut scores = [0i32; MAX_MOVES];
        let slice = list.as_slice();
        for (i, &m) in slice.iter().enumerate() {
            scores[i] = score_move(pos, m, tt_move, killers, countermove, history, check_ctx);
        }
        MoveOrderer { scores, len: slice.len(), next: 0 }
    }

    /// Selects the highest-scoring move among the unconsumed tail of
    /// `list` and swaps it into position `self.next`, then returns it.
    /// Returns None once every move has been produced.
    #[inline]
    pub(crate) fn next_move(&mut self, list: &mut MoveList) -> Option<(usize, Move)> {
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
