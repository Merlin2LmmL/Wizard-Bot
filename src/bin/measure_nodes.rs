use wizardbot_engine::position::Position;
use wizardbot_engine::search::{SearchState, SearchLimits, iterative_deepening};
fn main() {
    let mut state = SearchState::new();
    let mut pos = Position::startpos();
    let limits = SearchLimits { max_depth: 4, deadline_ms: 20000.0 };
    iterative_deepening(&mut pos, &mut state, limits, 0, |_line| {});
    println!("Done. state.local.nodes={}", state.local.nodes);
}
