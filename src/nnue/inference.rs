// Ported from Stockfish sf_16 (68e1e9b3811e16cad014b590d7443b9063b3eb52),
// see Stockfish/src/nnue/nnue_architecture.h (struct Network, get_hash_value,
// read_parameters, propagate) and Stockfish/src/nnue/evaluate_nnue.cpp
// (Eval::NNUE::evaluate).

use crate::nnue::accum;
use crate::nnue::layers::{self, AffineLayer};
use crate::nnue::loader::{self, Reader};
use crate::nnue::transformer::{FeatureTransformer, HALF_DIMENSIONS};
use crate::nnue::position::{Position, WHITE, BLACK};

/// Stockfish/src/nnue/nnue_common.h:54-55
pub const OUTPUT_SCALE: i32 = 16;
pub const WEIGHT_SCALE_BITS: u32 = 6;

/// Stockfish/src/nnue/nnue_architecture.h:45
pub const LAYER_STACKS: usize = 8;

/// Stockfish/src/nnue/nnue_architecture.h:49-50
pub const FC_0_OUTPUTS: usize = 15;
pub const FC_1_OUTPUTS: usize = 32;

/// One "network" bucket: fc_0 -> (ac_sqr_0, ac_0) -> fc_1 -> ac_1 -> fc_2.
/// Stockfish/src/nnue/nnue_architecture.h:47-58
pub struct Network {
    pub fc_0: AffineLayer, // AffineTransformSparseInput<1536, 16>
    pub fc_1: AffineLayer, // AffineTransform<30, 32>
    pub fc_2: AffineLayer, // AffineTransform<32, 1>
}

impl Network {
    pub fn new_zeroed() -> Self {
        Network {
            fc_0: AffineLayer::new_zeroed(
                HALF_DIMENSIONS, // 1536 -- fc_0's input is the feature transformer's output
                FC_0_OUTPUTS + 1, // 16
            ),
            fc_1: AffineLayer::new_zeroed(FC_0_OUTPUTS * 2, FC_1_OUTPUTS), // 30 -> 32
            fc_2: AffineLayer::new_zeroed(FC_1_OUTPUTS, 1),                // 32 -> 1
        }
    }

    /// Network::get_hash_value(), Stockfish/src/nnue/nnue_architecture.h:60-72:
    ///   hashValue = 0xEC42E90Du;
    ///   hashValue ^= TransformedFeatureDimensions * 2;
    ///   hashValue = fc_0::get_hash_value(hashValue);
    ///   hashValue = ac_0::get_hash_value(hashValue);   // ac_sqr_0 is NOT hashed
    ///   hashValue = fc_1::get_hash_value(hashValue);
    ///   hashValue = ac_1::get_hash_value(hashValue);
    ///   hashValue = fc_2::get_hash_value(hashValue);
    pub fn hash_value() -> u32 {
        let mut hash_value: u32 = 0xEC42E90Du32;
        hash_value ^= (HALF_DIMENSIONS as u32) * 2;
        hash_value = AffineLayer::hash_value((FC_0_OUTPUTS + 1) as u32, hash_value); // fc_0
        hash_value = layers::clipped_relu_hash_value(hash_value); // ac_0
        hash_value = AffineLayer::hash_value(FC_1_OUTPUTS as u32, hash_value); // fc_1
        hash_value = layers::clipped_relu_hash_value(hash_value); // ac_1
        hash_value = AffineLayer::hash_value(1, hash_value); // fc_2
        hash_value
    }

    /// Network::read_parameters, Stockfish/src/nnue/nnue_architecture.h:75-81
    /// (ac_sqr_0 and ac_1/ac_0's read_parameters are no-ops that consume no
    /// bytes -- ClippedReLU/SqrClippedReLU have no learned parameters.)
    pub fn read_parameters(&mut self, r: &mut Reader) -> std::io::Result<()> {
        self.fc_0.read_parameters(r)?;
        // ac_0.read_parameters(stream) -- no-op, no bytes.
        self.fc_1.read_parameters(r)?;
        // ac_1.read_parameters(stream) -- no-op, no bytes.
        self.fc_2.read_parameters(r)?;
        Ok(())
    }

    /// Network::propagate, Stockfish/src/nnue/nnue_architecture.h:92-132.
    /// `transformed_features` is the 1536-wide u8 output of the feature
    /// transformer's transform(). Returns the "positional" int32 score.
    pub fn propagate(&self, transformed_features: &[u8]) -> i32 {
        let fc_0_out = self.fc_0.propagate(transformed_features); // len 16

        // ac_sqr_0.propagate(fc_0_out, ac_sqr_0_out) over all 16 outputs.
        let ac_sqr_0_out = layers::sqr_clipped_relu(&fc_0_out, WEIGHT_SCALE_BITS); // len 16
        // ac_0.propagate(fc_0_out, ac_0_out) over all 16 outputs.
        let ac_0_out = layers::clipped_relu(&fc_0_out, WEIGHT_SCALE_BITS); // len 16

        // memcpy(ac_sqr_0_out + FC_0_OUTPUTS, ac_0_out, FC_0_OUTPUTS bytes):
        // combined[0..FC_0_OUTPUTS)            = ac_sqr_0_out[0..FC_0_OUTPUTS)
        // combined[FC_0_OUTPUTS..2*FC_0_OUTPUTS) = ac_0_out[0..FC_0_OUTPUTS)
        // (fc_0_out[FC_0_OUTPUTS], the 16th/extra output, is deliberately
        // excluded from this concatenation -- it feeds the skip connection
        // below instead.)
        let mut combined = vec![0u8; FC_0_OUTPUTS * 2];
        combined[0..FC_0_OUTPUTS].copy_from_slice(&ac_sqr_0_out[0..FC_0_OUTPUTS]);
        combined[FC_0_OUTPUTS..FC_0_OUTPUTS * 2].copy_from_slice(&ac_0_out[0..FC_0_OUTPUTS]);

        let fc_1_out = self.fc_1.propagate(&combined); // len 32
        let ac_1_out = layers::clipped_relu(&fc_1_out, WEIGHT_SCALE_BITS); // len 32
        let fc_2_out = self.fc_2.propagate(&ac_1_out); // len 1

        // buffer.fc_0_out[FC_0_OUTPUTS] is such that 1.0 == 127*(1<<WeightScaleBits) in
        // quantized form, but we want 1.0 == 600*OutputScale.
        // Stockfish/src/nnue/nnue_architecture.h:126-129
        let fwd_out: i32 = (fc_0_out[FC_0_OUTPUTS] as i64 * (600 * OUTPUT_SCALE) as i64
            / (127 * (1i64 << WEIGHT_SCALE_BITS))) as i32;
        fc_2_out[0].wrapping_add(fwd_out)
    }
}

/// The full loaded NNUE evaluation function: one shared feature transformer
/// plus LAYER_STACKS independently-trained FC networks selected by piece
/// count (the "layer stack" bucket).
pub struct NnueEvaluator {
    pub feature_transformer: FeatureTransformer,
    pub networks: Vec<Network>, // len == LAYER_STACKS
    pub description: String,
}

impl NnueEvaluator {
    /// Combined top-level hash value.
    /// Stockfish/src/nnue/evaluate_nnue.h:31-32:
    ///   HashValue = FeatureTransformer::get_hash_value() ^ Network::get_hash_value();
    pub fn combined_hash_value() -> u32 {
        FeatureTransformer::hash_value() ^ Network::hash_value()
    }

    /// load_eval / read_parameters, Stockfish/src/nnue/evaluate_nnue.cpp:117-127.
    ///
    /// Takes raw bytes directly rather than a filesystem path, so it works
    /// identically whether the caller obtained them from disk (native
    /// builds, via `load()` below) or from `include_bytes!` at compile time
    /// (wasm32 builds, which have no real filesystem to read from).
    pub fn load_from_bytes(data: &[u8]) -> Result<NnueEvaluator, String> {
        let mut r = Reader::new(data);

        let header = loader::read_header(&mut r).map_err(|e| format!("bad NNUE header: {e}"))?;
        let expected = Self::combined_hash_value();
        if header.hash_value != expected {
            return Err(format!(
                "top-level NNUE hash mismatch: file has {:#010x}, port computes {:#010x} \
                 (FeatureTransformer::hash_value()={:#010x}, Network::hash_value()={:#010x})",
                header.hash_value,
                expected,
                FeatureTransformer::hash_value(),
                Network::hash_value()
            ));
        }

        let mut feature_transformer = FeatureTransformer::new_zeroed();
        loader::read_component_header(&mut r, FeatureTransformer::hash_value())
            .map_err(|e| format!("feature transformer header: {e}"))?;
        feature_transformer
            .read_parameters(&mut r)
            .map_err(|e| format!("feature transformer parameters: {e}"))?;

        let mut networks = Vec::with_capacity(LAYER_STACKS);
        for i in 0..LAYER_STACKS {
            loader::read_component_header(&mut r, Network::hash_value())
                .map_err(|e| format!("network[{i}] header: {e}"))?;
            let mut net = Network::new_zeroed();
            net.read_parameters(&mut r)
                .map_err(|e| format!("network[{i}] parameters: {e}"))?;
            networks.push(net);
        }

        if !r.at_eof() {
            return Err(format!(
                "trailing {} unread bytes after parsing all NNUE parameters",
                r.remaining()
            ));
        }

        Ok(NnueEvaluator {
            feature_transformer,
            networks,
            description: header.description,
        })
    }

    /// Native-only convenience wrapper that reads the file from disk first.
    /// Not available on wasm32 -- there is no filesystem there, so callers
    /// on that target must use `load_from_bytes` with e.g. `include_bytes!`
    /// or bytes fetched from JS instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &str) -> Result<NnueEvaluator, String> {
        let data = loader::read_file(path).map_err(|e| format!("failed to read {path}: {e}"))?;
        Self::load_from_bytes(&data)
    }

    /// Shared tail end of both `transform()` and `transform_from_accumulators()`:
    /// combines a pair of already-computed per-perspective accumulators
    /// (indexed by absolute perspective, [WHITE, BLACK]) plus a
    /// side-to-move-relative ordering into the psqt value and the
    /// 1536-wide transformed feature vector fc_0 reads.
    ///
    /// FeatureTransformer::transform(pos, output, bucket), scalar branch.
    /// Stockfish/src/nnue/nnue_feature_transformer.h:274-336 (the `#else`
    /// arm at 318-328; the `#if defined(VECTOR)` arm is SIMD-only and
    /// skipped per the scalar-only translation rule).
    fn transform_from_accumulators(
        &self,
        acc_white: &accum::Accumulator,
        acc_black: &accum::Accumulator,
        side_to_move: usize, // 0 = WHITE, 1 = BLACK
        bucket: usize,
    ) -> (i32, [u8; HALF_DIMENSIONS]) {
        let accumulation = [&acc_white.accumulation, &acc_black.accumulation];
        let psqt_accumulation = [&acc_white.psqt_accumulation, &acc_black.psqt_accumulation];

        let stm = side_to_move;
        let opp = 1 - stm;
        let perspectives = [stm, opp];

        // psqt = (psqtAccumulation[perspectives[0]][bucket] - psqtAccumulation[perspectives[1]][bucket]) / 2
        let psqt = (psqt_accumulation[perspectives[0]][bucket]
            - psqt_accumulation[perspectives[1]][bucket])
            / 2;

        let mut output = [0u8; HALF_DIMENSIONS];
        const HALF: usize = HALF_DIMENSIONS / 2;
        for p in 0..2 {
            let offset = HALF * p;
            let acc = accumulation[perspectives[p]];
            for j in 0..HALF {
                let sum0 = acc[j].clamp(0, 127) as i32;
                let sum1 = acc[j + HALF].clamp(0, 127) as i32;
                output[offset + j] = ((sum0 * sum1) / 128) as u8;
            }
        }

        (psqt, output)
    }

    /// Builds both perspectives' accumulators from scratch via
    /// `compute_accumulator` (a full refresh), then delegates to
    /// `transform_from_accumulators`. This is the one-shot path: still
    /// used by `evaluate()` below, and by anything else (tests, tooling)
    /// that only has a bare `Position` and no persisted accumulator state
    /// to reuse.
    fn transform(&self, pos: &Position) -> (i32, [u8; HALF_DIMENSIONS]) {
        let acc_white = accum::compute_accumulator(&self.feature_transformer, pos, WHITE);
        let acc_black = accum::compute_accumulator(&self.feature_transformer, pos, BLACK);
        let bucket = self.current_layer_stack_bucket(pos);
        self.transform_from_accumulators(&acc_white, &acc_black, pos.side_to_move as usize, bucket)
    }

    fn current_layer_stack_bucket(&self, pos: &Position) -> usize {
        // Stockfish/src/nnue/evaluate_nnue.cpp:165:
        //   const int bucket = (pos.count<ALL_PIECES>() - 1) / 4;
        ((pos.count_all_pieces() as i32 - 1) / 4) as usize
    }

    /// Eval::NNUE::evaluate(pos, /*adjusted=*/false), Stockfish/src/nnue/evaluate_nnue.cpp:144-177.
    /// Returns the raw Value (already divided by OutputScale), from the
    /// perspective of the side to move -- matching what our added debug
    /// UCI command `nnueraw` in the reference Stockfish binary prints via
    /// `Eval::NNUE::evaluate(pos, false)`.
    ///
    /// Unchanged from before: still builds a full `nnue::position::Position`
    /// and refreshes both accumulators from scratch every call. Kept as-is
    /// (rather than folded into `evaluate_with_accumulators`) so existing
    /// one-shot callers -- tests included -- see no behavior or signature
    /// change.
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let bucket = self.current_layer_stack_bucket(pos);
        let (psqt, transformed_features) = self.transform(pos);
        let positional = self.networks[bucket].propagate(&transformed_features);
        (psqt + positional) / OUTPUT_SCALE
    }

    /// Same as `evaluate`, but takes already-computed per-perspective
    /// accumulators instead of rebuilding a `nnue::position::Position` and
    /// running `compute_accumulator` fresh on every call.
    ///
    /// This is the entry point the incremental-accumulator path in
    /// `crate::position::Position` uses: `nnue_accum[WHITE]` /
    /// `nnue_accum[BLACK]` there are kept up to date incrementally across
    /// make_move/unmake_move (see `Position::toggle_piece_feature` and
    /// `Position::refresh_dirty_nnue_perspectives`), so evaluation just
    /// reads them instead of recomputing from scratch on every node.
    ///
    /// `side_to_move` and `piece_count` are the only remaining values this
    /// needs that used to come from a full `nnue::position::Position` --
    /// both are cheap for the caller to supply directly.
    pub fn evaluate_with_accumulators(
        &self,
        accum_white: &accum::Accumulator,
        accum_black: &accum::Accumulator,
        side_to_move: u32, // 0 = WHITE, 1 = BLACK, matches nnue::position::{WHITE, BLACK}
        piece_count: u32,
    ) -> i32 {
        let bucket = ((piece_count as i32 - 1) / 4) as usize;
        let (psqt, transformed_features) = self.transform_from_accumulators(
            accum_white,
            accum_black,
            side_to_move as usize,
            bucket,
        );
        let positional = self.networks[bucket].propagate(&transformed_features);
        (psqt + positional) / OUTPUT_SCALE
    }
}
