use wizardbot_engine::position::{Position, MoveList};
use wizardbot_engine::movegen::generate_legal_moves;
fn main() {
    let pos = Position::from_fen("r1bqk2r/p1p1pnpp/3p4/np1b4/4P3/1PB5/P4PPP/RNBQK2R w KQkq - 0 9").unwrap();
    let mut list = MoveList::new();
    generate_legal_moves(&pos, &mut list);
    println!("Move count: {}", list.count);
    println!("Ply: {}  Fullmove: {}  Side: {:?}", pos.ply(), pos.fullmove_number, pos.side_to_move);
    println!("First 10 moves:");
    for i in 0..list.count.min(10) {
        println!("  {}: {}", i, list.as_slice()[i].to_uci());
    }
}
