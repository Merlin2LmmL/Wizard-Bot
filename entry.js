var wasmUrl = decodeURIComponent(self.location.hash.slice(1));
var messageQueue = [];
var wasmReady = false;
self.onmessage = function (ev) {
  if (wasmReady) {
    wasm_bindgen.handle_line(String(ev.data));
  } else {
    messageQueue.push(ev.data);
  }
};
fetch(wasmUrl).then(r => r.arrayBuffer()).then(buf => {
  var module = new WebAssembly.Module(buf);
  wasm_bindgen.initSync({ module: module });
  wasm_bindgen.init(function (line) { postMessage(line); });
  wasmReady = true;
  while (messageQueue.length) {
    wasm_bindgen.handle_line(String(messageQueue.shift()));
  }
});
