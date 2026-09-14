// Ported from Stockfish sf_16 (68e1e9b3811e16cad014b590d7443b9063b3eb52),
// see Stockfish/src/nnue/nnue_feature_transformer.h,
// update_accumulator_refresh(), scalar (#else, non-VECTOR) branch, lines 591-607.
//
// Two accumulator update paths now live here:
//
//   - compute_accumulator(): a full "refresh" from scratch, summing every
//     active feature's weight column. Used at position construction and
//     whenever a king moves (HalfKAv2_hm features are king-relative, so a
//     king move invalidates that perspective's entire feature set, not
//     just one square -- see Position::toggle_piece_feature).
//
//   - apply_feature(): an incremental single-feature update (+= or -= one
//     weight column), used for every other piece event during
//     make_move/unmake_move. Two's-complement wrapping add/sub are exact
//     exact inverses of each other bit-for-bit, so a sequence of
//     apply_feature calls and their inverses always round-trips back to
//     the original accumulator with no precision loss -- unlike floating
//     point, where +x then -x can drift.
//
// Both paths are mathematically equivalent for any given board state
// (summation of int16/int32 weight columns is associative/commutative
// under fixed-width wraparound), which is what makes it safe for
// Position::refresh_dirty_nnue_perspectives() to fall back to a full
// compute_accumulator() on a king move rather than trying to track
// incremental deltas through a change of king-relative orientation.

use crate::nnue::features;
use crate::nnue::transformer::{FeatureTransformer, HALF_DIMENSIONS, PSQT_BUCKETS};
use crate::nnue::position::{Color, Position};

#[derive(Clone)]
pub struct Accumulator {
    pub accumulation: [i16; HALF_DIMENSIONS],
    pub psqt_accumulation: [i32; PSQT_BUCKETS],
}

impl Accumulator {
    /// A placeholder accumulator with no features applied. Used as the
    /// initial value before the first real refresh (e.g. at Position
    /// construction, or when NNUE isn't loaded and the value is simply
    /// never read because classical_evaluate() is used instead).
    pub fn new_zeroed() -> Self {
        Accumulator {
            accumulation: [0; HALF_DIMENSIONS],
            psqt_accumulation: [0; PSQT_BUCKETS],
        }
    }
}

/// update_accumulator_refresh<Perspective>(pos), scalar branch:
///   std::memcpy(accumulator.accumulation[Perspective], biases, HalfDimensions * sizeof(BiasType));
///   for k in 0..PSQTBuckets: accumulator.psqtAccumulation[Perspective][k] = 0;
///   for index in active:
///     for j in 0..HalfDimensions: accumulator.accumulation[Perspective][j] += weights[offset + j];
///     for k in 0..PSQTBuckets: accumulator.psqtAccumulation[Perspective][k] += psqtWeights[index * PSQTBuckets + k];
pub fn compute_accumulator(
    ft: &FeatureTransformer,
    pos: &Position,
    perspective: Color,
) -> Accumulator {
    let mut accumulation = [0i16; HALF_DIMENSIONS];
    accumulation.copy_from_slice(&ft.biases);

    let mut psqt_accumulation = [0i32; PSQT_BUCKETS];

    let mut active = features::ActiveIndices::new();
    features::append_active_indices(pos, perspective, &mut active);

    for &index in active.as_slice() {
        let col = ft.weight_column(index);
        for j in 0..HALF_DIMENSIONS {
            // Same wrapping semantics as C++ int16_t += int16_t (both wrap
            // silently on overflow in practice for this trained network;
            // Rust's `+` on i16 panics in debug on overflow instead of
            // silently wrapping like C++, so use wrapping_add to stay
            // faithful to the exact bit-for-bit C++ behavior in all builds).
            accumulation[j] = accumulation[j].wrapping_add(col[j]);
        }

        let psqt_row = ft.psqt_row(index);
        for k in 0..PSQT_BUCKETS {
            psqt_accumulation[k] = psqt_accumulation[k].wrapping_add(psqt_row[k]);
        }
    }

    Accumulator {
        accumulation,
        psqt_accumulation,
    }
}

/// Applies (or removes) a single active feature's contribution to an
/// already-computed accumulator. This is the incremental counterpart to
/// compute_accumulator()'s inner loop: same per-column wrapping add over
/// weights/psqt_weights, just for one feature index instead of the whole
/// active set, with `add` controlling direction. `add = false` is used
/// when a feature leaves the active set (a piece moved off a square or
/// was captured); `add = true` when a feature enters it (a piece arrived
/// on a square). Calling this with `add = true` for some index and later
/// with `add = false` for the same index always restores the accumulator
/// to its prior value exactly.
#[inline]
pub fn apply_feature(accum: &mut Accumulator, ft: &FeatureTransformer, index: u32, add: bool) {
    let col = ft.weight_column(index);
    let psqt_row = ft.psqt_row(index);

    if add {
        for j in 0..HALF_DIMENSIONS {
            accum.accumulation[j] = accum.accumulation[j].wrapping_add(col[j]);
        }
        for k in 0..PSQT_BUCKETS {
            accum.psqt_accumulation[k] = accum.psqt_accumulation[k].wrapping_add(psqt_row[k]);
        }
    } else {
        for j in 0..HALF_DIMENSIONS {
            accum.accumulation[j] = accum.accumulation[j].wrapping_sub(col[j]);
        }
        for k in 0..PSQT_BUCKETS {
            accum.psqt_accumulation[k] = accum.psqt_accumulation[k].wrapping_sub(psqt_row[k]);
        }
    }
}
