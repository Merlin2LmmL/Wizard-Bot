// Ported from Stockfish sf_16 (68e1e9b3811e16cad014b590d7443b9063b3eb52),
// see Stockfish/src/nnue/nnue_feature_transformer.h,
// update_accumulator_refresh(), scalar (#else, non-VECTOR) branch, lines 591-607.
//
// We only ever need a single, from-scratch evaluation per FEN (no search
// tree / incremental updates), so we always do a full "refresh" computation
// per perspective. This is mathematically identical to Stockfish's
// incremental update path: both are just sums of int16 (resp. int32)
// weight columns over the active-feature set, and two's-complement
// fixed-width addition is associative/commutative regardless of the order
// features are added in, so a from-scratch refresh always yields the same
// bit-for-bit accumulator as any sequence of incremental updates would.

use crate::nnue::features;
use crate::nnue::transformer::{FeatureTransformer, HALF_DIMENSIONS, PSQT_BUCKETS};
use crate::nnue::position::{Color, Position};

pub struct Accumulator {
    pub accumulation: [i16; HALF_DIMENSIONS],
    pub psqt_accumulation: [i32; PSQT_BUCKETS],
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

    let mut active: Vec<u32> = Vec::with_capacity(features::MAX_ACTIVE_DIMENSIONS);
    features::append_active_indices(pos, perspective, &mut active);

    for &index in &active {
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
