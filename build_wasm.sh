#!/bin/bash
# 1. Compile to wasm
set -euo pipefail
cargo build --release --target wasm32-unknown-unknown

# 2. Run wasm-bindgen — --target no-modules matches the global
#    `wasm_bindgen` function style your current entry.js uses
wasm-bindgen target/wasm32-unknown-unknown/release/wizardbot_engine.wasm \
  --target no-modules --out-dir pkg --out-name wizardbot_engine

# 3. Copy the wasm straight into dist/
cp "pkg/wizardbot_engine_bg.wasm" "dist/wizardbot_engine_bg.wasm"

# copy the current manifest — it was never being refreshed, so dist/
# was shipping a stale kind/version indefinitely
cp "manifest.json" "dist/manifest.json"

# 4. Rebuild dist/entry.js = fresh glue + correct worker-bootstrap (with messageQueue)
cp "pkg/wizardbot_engine.js" "dist/entry.js"
cat "entry.js" >> "dist/entry.js"

# 5. Package the deployable zip — flat at root, matching manifest.json
(cd dist && zip -j ../wizardbot_engine.zip manifest.json entry.js wizardbot_engine_bg.wasm)

# Return back to the project dir
