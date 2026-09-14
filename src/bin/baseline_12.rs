use wizardbot_engine::position::Position;
use wizardbot_engine::search::{SearchState, SearchLimits, iterative_deepening};
fn main() {
    let fen = "r1bq1rk1/pp4pp/2np4/1p2p3/8/2PBBN2/P4PPP/RN1Q1RK1 w - - 0 12";
    let mut pos = Position::from_fen(fen).unwrap();
    let mut state = SearchState::new();
    let limits = SearchLimits { max_depth: 12, deadline_ms: 30000.0 };
    let start = std::time::Instant::now();
    let best = iterative_deepening(&mut pos, &mut state, limits, 0, |_line| {});
    let elapsed = start.elapsed();
    let total_nodes = state.local.nodes + state.shared.nodes_aggregate.load(std::sync::atomic::Ordering::Relaxed);
    println!("BASELINE depth=12 FEN={} best={} nodes={} time_ms={:?}", fen.split_whitespace().next().unwrap(), best.to_uci(), total_nodes, elapsed.as_millis());
}
