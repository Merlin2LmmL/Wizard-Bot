// nnue module: Rust port of Stockfish sf_16's NNUE evaluation
// (commit 68e1e9b3811e16cad014b590d7443b9063b3eb52).
//
// Submodules mirror the Stockfish source layout:
//   loader.rs      -> nnue_common.h (readers) + evaluate_nnue.cpp (header/param framing)
//   features.rs    -> features/half_ka_v2_hm.{h,cpp}
//   transformer.rs -> nnue_feature_transformer.h (struct + read_parameters)
//   accum.rs       -> nnue_feature_transformer.h (accumulator refresh)
//   layers.rs      -> layers/{affine_transform,affine_transform_sparse_input,clipped_relu,sqr_clipped_relu}.h
//   inference.rs   -> nnue_architecture.h (Network) + evaluate_nnue.cpp (Eval::NNUE::evaluate)

pub mod accum;
pub mod features;
pub mod inference;
pub mod layers;
pub mod loader;
pub mod transformer;

#[cfg(test)]
mod nnue_test;

pub use inference::NnueEvaluator;
pub mod position;
