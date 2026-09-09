// Ported from Stockfish sf_16 (68e1e9b3811e16cad014b590d7443b9063b3eb52),
// see Stockfish/src/nnue/nnue_feature_transformer.h

use crate::nnue::features;
use crate::nnue::loader::Reader;

/// Number of output dimensions for one side (a.k.a. TransformedFeatureDimensions).
/// Stockfish/src/nnue/nnue_architecture.h:43
pub const HALF_DIMENSIONS: usize = 1536;

/// Number of PSQT output buckets.
/// Stockfish/src/nnue/nnue_architecture.h:44
pub const PSQT_BUCKETS: usize = 8;

/// Input dimensions (feature-set dimensions).
pub const INPUT_DIMENSIONS: usize = features::DIMENSIONS as usize;

pub type BiasType = i16;
pub type WeightType = i16;
pub type PsqtWeightType = i32;

pub struct FeatureTransformer {
    pub biases: Vec<BiasType>,           // [HALF_DIMENSIONS]
    pub weights: Vec<WeightType>,        // [HALF_DIMENSIONS * INPUT_DIMENSIONS]
    pub psqt_weights: Vec<PsqtWeightType>, // [INPUT_DIMENSIONS * PSQT_BUCKETS]
}

impl FeatureTransformer {
    pub fn new_zeroed() -> Self {
        FeatureTransformer {
            biases: vec![0; HALF_DIMENSIONS],
            weights: vec![0; HALF_DIMENSIONS * INPUT_DIMENSIONS],
            psqt_weights: vec![0; INPUT_DIMENSIONS * PSQT_BUCKETS],
        }
    }

    /// Hash value embedded in the evaluation file.
    /// Stockfish/src/nnue/nnue_feature_transformer.h:249-251:
    ///   return FeatureSet::HashValue ^ (OutputDimensions * 2);
    pub fn hash_value() -> u32 {
        features::HASH_VALUE ^ ((HALF_DIMENSIONS as u32) * 2)
    }

    /// Read network parameters.
    /// Stockfish/src/nnue/nnue_feature_transformer.h:254-261
    pub fn read_parameters(&mut self, r: &mut Reader) -> std::io::Result<()> {
        r.read_leb128_i16(&mut self.biases)?;
        r.read_leb128_i16(&mut self.weights)?;
        r.read_leb128_i32(&mut self.psqt_weights)?;
        Ok(())
    }

    /// biases[HalfDimensions * index .. HalfDimensions * (index + 1)]
    #[inline]
    pub fn weight_column(&self, index: u32) -> &[WeightType] {
        let offset = HALF_DIMENSIONS * index as usize;
        &self.weights[offset..offset + HALF_DIMENSIONS]
    }

    #[inline]
    pub fn psqt_row(&self, index: u32) -> &[PsqtWeightType] {
        let offset = PSQT_BUCKETS * index as usize;
        &self.psqt_weights[offset..offset + PSQT_BUCKETS]
    }
}
