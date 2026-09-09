#!/bin/bash
# Server variant: native binary with multi-core threading (rayon / parallel-search)
set -euo pipefail
cargo build --release --features parallel-search --bin uci_binary --jobs 24
echo "Server binary: ./target/release/uci_binary"
echo "Run server bridge: python3 hybrid_server.py"
