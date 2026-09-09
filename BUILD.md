1. Drop the real WizardBot openings.js (368KB) into the crate root,
   replacing the placeholder — build.rs parses this exact format
   (`var OPENING_BOOK = {...};`) with zero code changes needed.

2. cargo test                     # run perft + zobrist + unit tests natively
3. cargo build --release --target wasm32-unknown-unknown
4. wasm-bindgen --target no-modules --out-dir pkg \
     target/wasm32-unknown-unknown/release/wizardbot_engine.wasm
5. wasm-opt -O4 pkg/wizardbot_engine_bg.wasm -o pkg/wizardbot_engine_bg.wasm
6. cat pkg/wizardbot_engine.js entry.js > dist/entry.js
   cp pkg/wizardbot_engine_bg.wasm dist/wizardbot_engine_bg.wasm
   cp manifest.json dist/manifest.json
7. zip -j WizardBot.zip dist/entry.js dist/wizardbot_engine_bg.wasm dist/manifest.json
8. Load through the existing package-loader.js unchanged — manifest.wasmStrategy
   is "hash-fragment", so the loader appends the .wasm asset URL to entry.js's
   URL hash and boots it as a classic Worker.

Optional: enable +popcnt in .cargo/config.toml:
  [build]
  rustflags = ["-C", "target-feature=+bulk-memory"]
Verify your deployment target's wasm feature support before relying on this.
