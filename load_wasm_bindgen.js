// Reads pkg/wizardbot_engine.js (a wasm-bindgen `--target no-modules` build,
// which only defines a top-level `let wasm_bindgen = ...` and never attaches
// it to module.exports) and re-evaluates it with one line appended so we can
// actually get the object out under Node's CommonJS loader.
//
// We do NOT edit pkg/wizardbot_engine.js itself -- this loads its source,
// appends `module.exports = wasm_bindgen;`, and runs the result in a fresh
// Node Module context via Module.wrap, so pkg/ stays untouched as a build
// artifact.

const fs = require('fs');
const path = require('path');
const Module = require('module');

function loadPatched(pkgJsPath) {
  const src = fs.readFileSync(pkgJsPath, 'utf8');
  const patched = src + '\nmodule.exports = wasm_bindgen;\n';

  const m = new Module(pkgJsPath, module.parent);
  m.filename = pkgJsPath;
  m.paths = Module._nodeModulePaths(path.dirname(pkgJsPath));
  m._compile(patched, pkgJsPath);

  return m.exports;
}

module.exports = { loadPatched };
