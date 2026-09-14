#!/bin/bash
set -euo pipefail

# --- Threaded wasm build ---
#
# Requires:
#   1. Rust nightly with the rust-src component (std must be recompiled
#      with atomics enabled — the prebuilt stable std does not have this):
#        rustup toolchain install nightly
#        rustup component add rust-src --toolchain nightly
#   2. Cargo.toml: wasm-bindgen-rayon + rayon under a `wasm-threads` feature,
#      e.g.
#        [target.'cfg(target_arch = "wasm32")'.dependencies]
#        wasm-bindgen-rayon = { version = "1.2", optional = true }
#        rayon = "1.10"
#        [features]
#        wasm-threads = ["dep:wasm-bindgen-rayon"]
#   3. The HOST PAGE must be served with:
#        Cross-Origin-Opener-Policy: same-origin
#        Cross-Origin-Embedder-Policy: require-corp
#      or SharedArrayBuffer won't exist and the worker pool init will fail.
#      This is a hosting-side change, not something this script can do.
#   4. JS side must call and AWAIT `await wasm_bindgen.initThreadPool(n)`
#      (exported via `init_thread_pool` in uci.rs) before sending any UCI
#      commands — otherwise the engine silently runs single-threaded.
#
# Falls back to a normal single-threaded build if THREADED=0.

THREADED="${THREADED:-1}"

if [ "$THREADED" = "1" ]; then
  echo "Building THREADED wasm (nightly + atomics)"
  RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals" \
    rustup run nightly cargo build --release \
      --target wasm32-unknown-unknown \
      --features wasm-threads \
      -Z build-std=panic_abort,std
else
  echo "Building single-threaded wasm (stable)"
  cargo build --release --target wasm32-unknown-unknown
fi

# 2. Run wasm-bindgen — --target no-modules matches the global
#    `wasm_bindgen` function style your current entry.js uses.
#    NOTE: for threaded builds, wasm-bindgen must also be invoked with
#    a matching nightly-built wasm-bindgen-cli version compatible with
#    the wasm-bindgen-rayon release you pinned in Cargo.toml — mismatches
#    here are the most common source of "memory is not shared" errors.
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

if [ "$THREADED" = "1" ]; then
  echo ""
  echo "Built with threading support. Reminder: the deploy target must serve"
  echo "COOP: same-origin / COEP: require-corp headers, and the JS entry point"
  echo "must call 'await wasm_bindgen.initThreadPool(navigator.hardwareConcurrency)'"
  echo "before sending any UCI commands, or the engine falls back to 1 thread."
fi
