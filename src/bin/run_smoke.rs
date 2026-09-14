use wizardbot_engine::position::Position;
use wizardbot_engine::search::{SearchState, SearchLimits, iterative_deepening};
fn run_fen(fen: &str) {
    let mut pos = Position::from_fen(fen).expect("bad FEN");
    let mut state = SearchState::new();
    let limits = SearchLimits { max_depth: 4, deadline_ms: 5000.0 };
    let mut score: i32 = 0;
    let best = iterative_deepening(&mut pos, &mut state, limits, 0, |line: &str| {
        eprintln!("INFO_LINE: {}", line);
        if line.contains("score cp ") {
            if let Some(p) = line.find("score cp ") {
                score = line[p+10..].split_whitespace().next().unwrap_or("0").parse::<i32>().unwrap_or(0);
            }
        } else if line.contains("score mate ") {
            if let Some(p) = line.find("score mate ") {
                score = line[p+11..].split_whitespace().next().unwrap_or("0").parse::<i32>().unwrap_or(0);
            }
        }
    });
    let total_nodes = state.local.nodes + state.shared.nodes_aggregate.load(std::sync::atomic::Ordering::Relaxed);
    println!("FEN: {} | bestmove={:?} score={:?} total_nodes={:?}", fen.split_whitespace().next().unwrap_or(fen), best.to_uci(), score, total_nodes);
}

fn main() {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",
        "rnb1k2r/ppp2ppp/3p4/4q3/4P3/3B1N2/PPP2PPP/RNBQK2R w KQkq - 2 7",
        "8/8/8/3k4/3K4/8/8/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "rnb1kbnr/pppp1ppp/8/4P3/8/8/PPP2PPP/RNBQKBNR b KQkq - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/2P5/8/PP1P1PPP/RNBQKBNR b KQkq c6 0 1",
        "r1bqkb1r/ppp2ppp/2n2n2/3pp3/2B1P3/5N2/PPP3PP/RNBQK2R w KQkq d6 0 5",
        "r1bq1rk1/pp4pp/2np4/1p2p3/8/2PBBN2/P4PPP/RN1Q1RK1 w - - 0 12",
    ];
    for fen in fens {
        run_fen(fen);
    }
}
