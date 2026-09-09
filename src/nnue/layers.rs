// Ported from Stockfish sf_16 (68e1e9b3811e16cad014b590d7443b9063b3eb52),
// see Stockfish/src/nnue/layers/affine_transform.h,
//     Stockfish/src/nnue/layers/affine_transform_sparse_input.h,
//     Stockfish/src/nnue/layers/clipped_relu.h,
//     Stockfish/src/nnue/layers/sqr_clipped_relu.h
//
// Only the plain scalar fallback (the #if !defined(USE_SSSE3) /
// affine_transform_non_ssse3 path, and the "Start = 0" scalar tails of the
// ReLU layers) is ported -- all AVX2/AVX512/SSE/NEON intrinsic branches are
// intentionally skipped per the task's scalar-only translation rule. Both
// the AffineTransform "small input" specialization and
// AffineTransformSparseInput fall back, in the non-SSSE3 case, to this same
// dense scalar routine (verified by reading both header files: each calls
// `affine_transform_non_ssse3<InputDimensions, PaddedInputDimensions,
// OutputDimensions>(output, weights, biases, input)` in its #else branch).

use crate::nnue::loader::Reader;

/// ceil_to_multiple, Stockfish/src/nnue/nnue_common.h:84-87
#[inline]
pub const fn ceil_to_multiple(n: usize, base: usize) -> usize {
    (n + base - 1) / base * base
}

/// MaxSimdWidth is always 32, regardless of target architecture.
/// Stockfish/src/nnue/nnue_common.h:77
pub const MAX_SIMD_WIDTH: usize = 32;

/// A fully-connected (affine) layer with InputDimensions -> OutputDimensions,
/// generic over both the AffineTransform "small input" specialization and
/// AffineTransformSparseInput, since -- as noted above -- both reduce to the
/// identical scalar dense routine and identical (non-scrambled) weight
/// layout for a scalar-only (non-SSSE3) build.
///
/// Field/layout notes (verbatim from the C++, see file headers cited above):
///   - BiasType = OutputType = i32
///   - WeightType = i8
///   - PaddedInputDimensions = ceil_to_multiple(InputDimensions, MaxSimdWidth)
///   - weights has OutputDimensions * PaddedInputDimensions entries, stored
///     row-major (row = output index, PaddedInputDimensions columns per
///     row), because get_weight_index(i) == i when USE_SSSE3 is not defined
///     (no scrambling in a scalar build).
///   - propagate() only sums the first InputDimensions columns of each row
///     (the padding columns exist only to keep the file's byte layout/
///     stream position correct and are otherwise unused mathematically).
pub struct AffineLayer {
    pub input_dimensions: usize,
    pub output_dimensions: usize,
    pub padded_input_dimensions: usize,
    pub biases: Vec<i32>,
    pub weights: Vec<i8>, // [output_dimensions * padded_input_dimensions]
}

impl AffineLayer {
    pub fn new_zeroed(input_dimensions: usize, output_dimensions: usize) -> Self {
        let padded_input_dimensions = ceil_to_multiple(input_dimensions, MAX_SIMD_WIDTH);
        AffineLayer {
            input_dimensions,
            output_dimensions,
            padded_input_dimensions,
            biases: vec![0; output_dimensions],
            weights: vec![0; output_dimensions * padded_input_dimensions],
        }
    }

    /// Hash value embedded in the evaluation file. Identical formula for
    /// both AffineTransform (small-input specialization) and
    /// AffineTransformSparseInput:
    ///   Stockfish/src/nnue/layers/affine_transform.h:420-427
    ///   Stockfish/src/nnue/layers/affine_transform_sparse_input.h:168-175
    ///     hashValue = 0xCC03DAE4u;
    ///     hashValue += OutputDimensions;
    ///     hashValue ^= prevHash >> 1;
    ///     hashValue ^= prevHash << 31;
    pub fn hash_value(output_dimensions: u32, prev_hash: u32) -> u32 {
        let mut hash_value: u32 = 0xCC03DAE4u32.wrapping_add(output_dimensions);
        hash_value ^= prev_hash >> 1;
        hash_value ^= prev_hash << 31;
        hash_value
    }

    /// read_parameters: biases as little-endian i32, then weights as
    /// little-endian i8 with identity index mapping (get_weight_index(i) ==
    /// i when USE_SSSE3 is not defined).
    /// Stockfish/src/nnue/layers/affine_transform.h:446-453
    /// Stockfish/src/nnue/layers/affine_transform_sparse_input.h:194-201
    pub fn read_parameters(&mut self, r: &mut Reader) -> std::io::Result<()> {
        for b in self.biases.iter_mut() {
            *b = r.read_i32_le()?;
        }
        for w in self.weights.iter_mut() {
            *w = r.read_i8_le()?;
        }
        Ok(())
    }

    /// affine_transform_non_ssse3, Stockfish/src/nnue/layers/affine_transform.h:147-153:
    ///   std::int32_t sum = biases[i];
    ///   for (j in 0..InputDimensions) sum += weights[offset + j] * input[j];
    ///   output[i] = sum;
    pub fn propagate(&self, input: &[u8]) -> Vec<i32> {
        debug_assert_eq!(input.len(), self.input_dimensions);
        let mut output = vec![0i32; self.output_dimensions];
        for i in 0..self.output_dimensions {
            let offset = i * self.padded_input_dimensions;
            let mut sum: i32 = self.biases[i];
            for j in 0..self.input_dimensions {
                sum = sum.wrapping_add((self.weights[offset + j] as i32) * (input[j] as i32));
            }
            output[i] = sum;
        }
        output
    }
}

/// ClippedReLU::propagate scalar tail.
/// Stockfish/src/nnue/layers/clipped_relu.h:169-172:
///   output[i] = clamp(input[i] >> WeightScaleBits, 0, 127) as uint8
pub fn clipped_relu(input: &[i32], weight_scale_bits: u32) -> Vec<u8> {
    input
        .iter()
        .map(|&v| {
            let shifted = v >> weight_scale_bits; // arithmetic shift, matches C++ on negative ints
            shifted.clamp(0, 127) as u8
        })
        .collect()
}

/// SqrClippedReLU::propagate scalar tail.
/// Stockfish/src/nnue/layers/sqr_clipped_relu.h:107-112:
///   output[i] = clamp( (((int64)input[i]*input[i]) >> (2*WeightScaleBits)) / 128, 0, 127 )
/// NOTE: the division by 128 happens BEFORE the clamp in the original
/// (`std::min(127ll, X / 128)`), not after -- order matters since it's
/// integer division, so we must match it exactly.
pub fn sqr_clipped_relu(input: &[i32], weight_scale_bits: u32) -> Vec<u8> {
    input
        .iter()
        .map(|&v| {
            let v64 = v as i64;
            let squared = v64.wrapping_mul(v64);
            let shifted = squared >> (2 * weight_scale_bits);
            let divided = shifted / 128; // C++ integer division truncates toward zero; shifted >= 0 here
            let clamped = divided.clamp(0, 127);
            clamped as u8
        })
        .collect()
}

/// ClippedReLU::get_hash_value, Stockfish/src/nnue/layers/clipped_relu.h:45-49
/// (identical formula used for SqrClippedReLU, since it's not hashed
/// separately in Network::get_hash_value -- see nnue_architecture.h -- but
/// we still expose it for completeness/documentation).
pub fn clipped_relu_hash_value(prev_hash: u32) -> u32 {
    0x538D24C7u32.wrapping_add(prev_hash)
}
