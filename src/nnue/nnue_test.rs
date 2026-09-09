// Mandatory validation harness against the 10 FEN cases in ground_truth.json
// Compares NNUE evaluation output (via nn-5af11540bbfe.nnue) against raw scores.

use crate::nnue::position::Position;
use crate::nnue::inference::NnueEvaluator;

fn load_evaluator() -> NnueEvaluator {
    NnueEvaluator::load("nn-5af11540bbfe.nnue").expect("NNUE load failed")
}

#[test]
fn nnue_validation_10_fens() {
    let evaluator = load_evaluator();
    // Cases from ai_work/ground_truth.json
    let cases: &[(&str, i32, &'static str)] = &[
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 85, "startpos"),
        ("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", 102, "early_opening"),
        ("r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1B3/PPPQBPPP/R4RK1 w - - 8 11", 46, "middlegame"),
        ("4k3/1p6/8/8/3Q4/8/6P1/K7 w - - 0 1", 2565, "king_a1"),
        ("4k3/2p5/8/8/3R4/8/6P1/7K w - - 0 1", 2567, "king_h1"),
        ("k7/6p1/8/3q4/8/8/1P6/4K3 b - - 0 1", 2565, "king_a8"),
        ("7k/6p1/8/3r4/8/8/2P5/4K3 b - - 0 1", 2567, "king_h8"),
        ("4k2r/pp3ppp/8/8/8/8/PP3PPP/R3K3 w - - 0 1", 75, "queenless_endgame"),
        ("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1", 4, "sparse_endgame"),
        ("rnbq1rk1/p1p1bpp1/1p2pn1p/3p4/2PP3B/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8", 183, "many_pieces_pos"),
    ];

    println!("\n=== NNUE VALIDATION BEFORE/AFTER ===");
    println!("{:25} {:>10} {:>12} {:>10}", "FEN case", "expected", "nnue_eval", "delta");
    for (fen, expected, name) in cases {
        let pos = Position::from_fen(fen).expect("bad fen");
        let val = evaluator.evaluate(&pos);
        println!("{:25} {:>10} {:>12} {:>10}", name, expected, val, val - expected);
    }
    println!("=== END VALIDATION ===\n");
}
