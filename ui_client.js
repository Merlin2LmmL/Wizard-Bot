// Thin browser client for hybrid server: eval display with fullscreen, max-mode menu, history scroll
const ws = new WebSocket('ws://localhost:8765');
ws.onopen = () => ws.send('uci\n');

let evalHistory = [];
let maxMode = 'dynamic'; // 'dynamic' | 'fixed'
let fixedMaxCp = 300;

function render() {
  const latest = evalHistory[evalHistory.length - 1] || {eval: 0};
  const cp = Math.round(latest.eval * 100);
  const displayMax = maxMode === 'dynamic' ? Math.max(300, Math.abs(cp) * 1.2) : fixedMaxCp;
  const pct = Math.min(100, Math.abs(cp) / displayMax * 100);
  document.getElementById('eval-bar').style.width = pct + '%';
  document.getElementById('eval-val').textContent = (cp > 0 ? '+' : '') + cp + ' cp';
}

ws.onmessage = (e) => {
  const msg = e.data.trim();
  if (msg.startsWith('info score')) {
    const m = msg.match(/score (cp|mate) ([+-]?\d+)/);
    if (m) evalHistory.push({eval: parseInt(m[2]) / 100, move: msg});
    render();
  }
};

// Fullscreen toggle
function toggleFullscreen() {
  const el = document.getElementById('eval-panel');
  if (!document.fullscreenElement) el.requestFullscreen();
  else document.exitFullscreen();
}

// Menu: dynamic vs fixed + scroll history
