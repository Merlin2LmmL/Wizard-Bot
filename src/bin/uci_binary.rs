use wizardbot_engine::uci::{EngineState, handle_uci_line};
use std::io::{self, BufRead};

fn main() {
    let mut engine = EngineState::new();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        handle_uci_line(&mut engine, &line, |s: &str| { println!("{}", s); });
    }
}
