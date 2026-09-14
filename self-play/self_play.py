#!/usr/bin/env python3
"""
Self-play / engine-review harness for a UCI chess engine binary.

Launches two independent instances of the same (or two different) UCI
engine binaries, feeds them the game position after each move, and prints
a live per-move table plus an end-of-run review: accuracy, a rough Elo
estimate, blunder triage, and (if two different binaries are given) a
head-to-head Elo-difference estimate.

Usage:
    python3 self_play.py --engine ./target/release/uci_binary --movetime 8000
    python3 self_play.py --engine ./target/release/uci_binary --depth 12
    python3 self_play.py --white ./engine_a --black ./engine_b --movetime 5000
    python3 self_play.py --engine ./target/release/uci_binary --games 5

DEFAULT MODE (only --engine given): both colors are the SAME engine binary
("symmetric" self-play). There is no opponent to measure a game-result Elo
against in that mode, so strength is estimated from move quality vs.
Stockfish instead (see "Live Stockfish verification" below). Because both
colors are the same engine, each game is treated as TWO datapoints -- one
for the engine playing White, one for it playing Black -- and combined for
the overall accuracy/Elo figures. Pass distinct --white/--black binaries to
get a real head-to-head result-based Elo-difference estimate instead.

Live Stockfish verification (accuracy + rough Elo estimate):
    --stockfish PATH             Real Stockfish binary. Auto-detected at
                                  ../Stockfish/stockfish (relative to this
                                  script) or on PATH if omitted; skipped
                                  silently if none is found.
    --no-stockfish-check         Disable even if a binary is found.
    --stockfish-time-ms MS       Per-position think time for Stockfish's
                                  verification queries. Default 1000.
    --stockfish-side {white,black,both}
                                  Which side's moves to verify (default both).
    --stockfish-sample-every N   Only verify every Nth ply (default 1).
    --sf-cp-loss-threshold CP    cp lost vs. Stockfish's own eval above which
                                  a move is reported as a confirmed error.

    For each checked move, Stockfish is asked (single-threaded, its own
    NNUE net) for its top move + eval at fen_before, and for its own eval
    of the position after the engine's actual move. cp lost, a per-move
    "accuracy" figure (same win%-based formula family Lichess uses for its
    game-review accuracy score), and a coarse move-quality tier
    (Best/Excellent/Good/Inaccuracy/Mistake/Blunder, approximated from the
    win% drop) are all derived from that. None of this is a calibrated Elo
    measurement -- see estimate_elo()'s docstring. Get a real number from a
    multi-game SPRT match against a known-strength opponent, not this tool.

Live board viewer:
    --live-board PATH             Write an SVG of the current board to PATH
                                   after every move, for a browser viewer
                                   (e.g. live_view.html) to poll and display.
                                   Defaults to "live_board.svg" in this
                                   script's own directory (self_test/), NOT
                                   --out-dir (self_test/games/). Pass an
                                   empty string ("") to disable. Put your
                                   viewer HTML alongside this script (or
                                   pass an absolute --live-board path) so
                                   the viewer's relative reference resolves.

Other diagnostics:
    --full-log / --no-full-log   Every "info" line per move to
                                  <out-prefix>_infos.jsonl (on by default).
    --stderr-log DIR             Redirect each engine's stderr to DIR.
    --blunder-threshold CP       Self-inflicted score drop to flag.
    --overrun-threshold-ms MS    Wall-clock overrun vs. --movetime to flag
                                  as a deadline-enforcement bug.
    --plain                      ASCII table borders instead of Unicode
                                  box-drawing (for non-UTF-8 terminals or
                                  log files you'll grep later).

Output files: all written under --out-dir (default "games/", created if
missing). Each game within a run gets its own numbered PGN so games
within one run never overwrite each other --
    games/<out-prefix>_1.pgn, <out-prefix>_2.pgn, ...   one file per game
    games/<out-prefix>_moves.csv              per-move detail, all games
    games/<out-prefix>_infos.jsonl            every "info" line per move
    games/<out-prefix>_review.txt             every game review + the
                                               final summary panel, plain text
    games/<out-prefix>_stockfish_outliers.json   confirmed errors, if any
NOTE: there is no run number -- rerunning the same --out-prefix
overwrites all of the above from the previous run.
Exception: --live-board is NOT under --out-dir (default
self_test/live_board.svg, next to this script) -- it's meant to be a
fixed path a browser viewer polls continuously. Pass an absolute path
to --live-board if your viewer HTML expects the SVG somewhere else.
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

try:
    import chess
    import chess.pgn
    import chess.svg
except ImportError:
    sys.exit(
        "Missing dependency. Install with:\n"
        "  pip install python-chess --break-system-packages"
    )

# Trailing custom keys the engine appends after "pv <moves...>". The PV
# capture stops as soon as it sees one of these tokens, instead of
# swallowing the rest of the line.
TRAILING_KEYS = {"bookns", "ttns", "singns", "evalns", "probcutns"}

MATE_CP_PLACEHOLDER = 100000  # for sorting/diffing mate scores alongside cp scores


# ---------------------------------------------------------------------------
# Box-drawing table / panel printer
# ---------------------------------------------------------------------------

BOX = {
    True:  dict(tl="┌", tr="┐", bl="└", br="┘", h="─", v="│",
                lt="├", rt="┤", tt="┬", bt="┴", x="┼"),
    False: dict(tl="+", tr="+", bl="+", br="+", h="-", v="|",
                lt="+", rt="+", tt="+", bt="+", x="+"),
}


class MoveTable:
    """Live per-move table for one game: ply/turn/move/score/depth/nodes/
    nps/diff-to-Stockfish, one row per ply, opened before the game loop and
    closed (with a result line) after it."""

    COLUMNS = [
        ("Ply",   4, "r"),
        ("Turn",  5, "l"),
        ("Move",  7, "l"),
        ("Score", 9, "r"),
        ("Depth", 5, "r"),
        ("Nodes", 8, "r"),
        ("NPS",   8, "r"),
        ("SF \u0394", 8, "r"),
    ]

    def __init__(self, title, fancy=True):
        self.b = BOX[fancy]
        self.widths = [w for _, w, _ in self.COLUMNS]
        self.inner = sum(w + 2 for w in self.widths) + (len(self.widths) - 1)
        b = self.b
        print(b["tl"] + b["h"] * self.inner + b["tr"])
        print(b["v"] + " " + title.ljust(self.inner - 2) + " " + b["v"])
        self._hline(b["lt"], b["tt"], b["rt"])
        print(self._cols_line([name.center(w) for name, w, _ in self.COLUMNS]))
        self._hline(b["lt"], b["x"], b["rt"])

    def _hline(self, left, mid, right):
        b = self.b
        segs = [b["h"] * (w + 2) for w in self.widths]
        print(left + mid.join(segs) + right)

    def _cols_line(self, cells):
        b = self.b
        return b["v"] + " " + (" " + b["v"] + " ").join(cells) + " " + b["v"]

    def row(self, *values):
        cells = []
        for (_, w, align), v in zip(self.COLUMNS, values):
            s = str(v)
            cells.append(s.rjust(w) if align == "r" else s.ljust(w))
        print(self._cols_line(cells))

    def close(self, footer_lines=None):
        b = self.b
        self._hline(b["lt"], b["bt"], b["rt"])
        for line in (footer_lines or []):
            print(b["v"] + " " + line.ljust(self.inner - 2) + " " + b["v"])
        print(b["bl"] + b["h"] * self.inner + b["br"])


# Cap so a panel's total printed width (borders included) never exceeds an
# 80-column terminal. Panels used to grow to fit their longest line
# unbounded, which blew past 80 cols as soon as a line had a long Elo/
# accuracy caveat in it.
PANEL_MAX_WIDTH = 78


def teeprint(text, report_fh=None):
    """Print to stdout and, if a report file handle is given, mirror the
    same line (plain text, no ANSI) into it."""
    print(text)
    if report_fh is not None:
        report_fh.write(text + "\n")


def print_panel(title, lines, fancy=True, report_fh=None):
    """Free-form boxed panel (no columns) for review/summary text -- same
    visual family as MoveTable so the whole run reads as one style. Wraps
    long lines and caps total width at PANEL_MAX_WIDTH instead of growing
    to fit the longest line in the content. Mirrors output to report_fh
    if given, so reviews/summaries can be saved alongside stdout."""
    b = BOX[fancy]
    inner_cap = PANEL_MAX_WIDTH - 2

    def wrap_all(src_lines, cap):
        out = []
        for l in src_lines:
            if len(l) <= cap:
                out.append(l)
            else:
                out.extend(textwrap.wrap(l, cap) or [""])
        return out

    wrapped = wrap_all(lines, inner_cap)
    width = min(max([len(title)] + [len(l) for l in wrapped] + [40]) + 2, PANEL_MAX_WIDTH)
    inner = width - 2
    # Re-wrap: if the cap above pushed width up from a short longest-line
    # (padded to the 40-char floor), inner may now differ from inner_cap.
    wrapped = wrap_all(wrapped, inner)

    teeprint(b["tl"] + b["h"] * width + b["tr"], report_fh)
    teeprint(b["v"] + " " + title[:inner].ljust(inner) + " " + b["v"], report_fh)
    teeprint(b["lt"] + b["h"] * width + b["rt"], report_fh)
    for l in wrapped:
        teeprint(b["v"] + " " + l.ljust(inner) + " " + b["v"], report_fh)
    teeprint(b["bl"] + b["h"] * width + b["br"], report_fh)


# ---------------------------------------------------------------------------
# Accuracy / move-quality / Elo estimation
# ---------------------------------------------------------------------------

def win_percent(cp):
    """cp (mover's POV) -> an intuitive 0-100 'win%' via the same logistic
    curve Lichess's client-side analysis uses before computing accuracy."""
    cp = max(-1000, min(1000, cp))  # curve saturates well before this; clamps mate placeholders too
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(win_before, win_after):
    """Per-move accuracy %, same formula family Lichess uses for its
    game-review 'Accuracy' figure: how much of the win% you had got
    preserved by this move. Clamped to [0, 100]."""
    win_drop = max(0.0, win_before - win_after)
    acc = 103.1668 * math.exp(-0.04354 * win_drop) - 3.1669
    return max(0.0, min(100.0, acc))


def classify_move(sf_match, win_drop):
    """Coarse move-quality bucket from the win% drop, loosely mirroring
    Lichess's Best/Excellent/Good/Inaccuracy/Mistake/Blunder tiers. These
    thresholds are an approximation for triage, not a reimplementation of
    Lichess's exact (unpublished) constants."""
    if sf_match:
        return "Best"
    if win_drop < 2:
        return "Excellent"
    if win_drop < 5:
        return "Good"
    if win_drop < 10:
        return "Inaccuracy"
    if win_drop < 20:
        return "Mistake"
    return "Blunder"


# Rough, UNCALIBRATED average-cp-loss -> Elo lookup, loosely based on
# commonly cited ACPL/rating correlations from other analysis tools. This
# has NOT been fitted to WizardBot's own actual playing strength via real
# rated games or SPRT -- it's a ballpark triage number for "does this look
# like it moved the needle," not a rating claim. Get a real number from a
# multi-game match against known-strength opposition.
ACPL_TO_ELO = [
    (0, 3200), (10, 2700), (20, 2400), (30, 2100), (40, 1950),
    (60, 1750), (80, 1550), (100, 1400), (150, 1150), (200, 950),
    (300, 700), (500, 400),
]


def estimate_elo(avg_cp_loss):
    """Interpolate ACPL_TO_ELO. See that table's docstring-equivalent
    comment above for the (significant) caveats."""
    if avg_cp_loss is None:
        return None
    pts = ACPL_TO_ELO
    if avg_cp_loss <= pts[0][0]:
        return pts[0][1]
    if avg_cp_loss >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= avg_cp_loss <= x1:
            t = (avg_cp_loss - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0))
    return None


def elo_diff_from_score(score_fraction, games):
    """Standard score%->Elo-difference conversion (the same formula
    cutechess-cli/ordo use) for a head-to-head match between two DIFFERENT
    engines. NOT meaningful for symmetric self-play (identical binary both
    colors) -- there's no independent opponent to measure a gap against;
    use the accuracy/cp-loss estimate above instead. Returns None at a
    perfect or zero score (can't extrapolate a finite gap from zero
    observed losses/draws) or with no games played."""
    if games <= 0 or score_fraction <= 0 or score_fraction >= 1:
        return None
    return -400 * math.log10(1 / score_fraction - 1)


def new_side_stats():
    return {
        "checked": 0, "matches": 0,
        "cp_lost_sum": 0.0, "cp_lost_n": 0,
        "acc_sum": 0.0, "acc_n": 0,
        "class_counts": {},
    }


def merge_side_stats(a, b):
    m = new_side_stats()
    for k in ("checked", "matches", "cp_lost_n", "acc_n"):
        m[k] = a[k] + b[k]
    m["cp_lost_sum"] = a["cp_lost_sum"] + b["cp_lost_sum"]
    m["acc_sum"] = a["acc_sum"] + b["acc_sum"]
    for src in (a["class_counts"], b["class_counts"]):
        for k, v in src.items():
            m["class_counts"][k] = m["class_counts"].get(k, 0) + v
    return m


CLASS_ORDER = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]


def format_class_table(class_counts):
    """Render the move-classification tally as a small two-row table
    (labels, then counts) instead of one long inline string. Both figures
    are uncalibrated triage numbers -- see estimate_elo()'s comment for
    why a low avg-cp-loss game can still contain real blunders."""
    col_w = max(len(c) for c in CLASS_ORDER) + 2
    header = "".join(c.center(col_w) for c in CLASS_ORDER)
    values = "".join(str(class_counts.get(c, 0)).center(col_w) for c in CLASS_ORDER)
    return [header.rstrip(), values.rstrip()]


def side_stats_lines(label, st):
    if st["checked"] == 0:
        return [f"{label}: no Stockfish-checked moves"]
    avg_cp = st["cp_lost_sum"] / st["cp_lost_n"] if st["cp_lost_n"] else None
    avg_acc = st["acc_sum"] / st["acc_n"] if st["acc_n"] else None
    elo = estimate_elo(avg_cp)
    match_pct = 100.0 * st["matches"] / st["checked"]
    lines = [
        f"{label}: {st['checked']} moves checked vs SF "
        f"({st['matches']} matched top choice, {match_pct:.1f}%)",
    ]
    if avg_cp is not None:
        lines.append(
            f"  avg cp lost: {avg_cp:.1f}  accuracy: {avg_acc:.1f}%  "
            f"Elo est: ~{elo} (uncalibrated)"
        )
    if st["class_counts"]:
        lines.extend("  " + l for l in format_class_table(st["class_counts"]))
    return lines


# ---------------------------------------------------------------------------
# UCI engine plumbing
# ---------------------------------------------------------------------------

def find_stockfish(explicit_path=None):
    """Resolve a Stockfish binary for live move verification. Resolution
    order: explicit --stockfish path, a couple of likely names under
    ../Stockfish/, then PATH. Returns None (not an error) if nothing is
    found -- verification is opt-out-able and shouldn't block a plain run."""
    if explicit_path:
        p = Path(explicit_path)
        return str(p) if p.is_file() else None

    here = Path(__file__).resolve().parent
    candidates = [
        here / ".." / "Stockfish" / "stockfish",
        here / ".." / "Stockfish" / "src" / "stockfish",
        here / ".." / "Stockfish" / "stockfish.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    return shutil.which("stockfish")


class UciEngine:
    """Thin wrapper around a UCI engine subprocess."""

    def __init__(self, path, name="engine", stderr_dest=None, options=None):
        self.name = name
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_dest if stderr_dest is not None else subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        for opt_name, opt_value in (options or {}).items():
            self._send(f"setoption name {opt_name} value {opt_value}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token, timeout=10.0):
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            lines.append(line)
            if line.startswith(token) or token in line:
                return lines
        raise TimeoutError(f"{self.name}: timed out waiting for '{token}'")

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def go(self, moves=None, movetime=None, depth=None, timeout=60.0, fen=None, searchmoves=None):
        """Set position, run search, return (bestmove, last_info_dict, all_infos).

        Pass `fen` to search an arbitrary position directly (used for
        Stockfish verification queries); otherwise `moves` is applied on
        top of startpos."""
        if fen is not None:
            pos_cmd = f"position fen {fen}"
        else:
            pos_cmd = "position startpos"
            if moves:
                pos_cmd += " moves " + " ".join(moves)
        self._send(pos_cmd)

        go_cmd = "go"
        if depth is not None:
            go_cmd += f" depth {depth}"
        if movetime is not None:
            go_cmd += f" movetime {movetime}"
        if depth is None and movetime is None:
            go_cmd += " movetime 5000"
        if searchmoves:
            go_cmd += " searchmoves " + " ".join(searchmoves)

        t0 = time.time()
        self._send(go_cmd)

        all_infos = []
        last_info = {}
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if line.startswith("info"):
                parsed = self._parse_info(line)
                if parsed:
                    all_infos.append(parsed)
                    last_info = parsed
            elif line.startswith("bestmove"):
                elapsed = time.time() - t0
                bestmove = line.split()[1]
                last_info["wall_time_s"] = round(elapsed, 3)
                return bestmove, last_info, all_infos

        raise TimeoutError(f"{self.name}: no bestmove within {timeout}s")

    @staticmethod
    def _parse_info(line):
        """Parse a UCI 'info ...' line into a dict of the fields we care about."""
        parts = line.split()
        out = {}
        i = 1
        while i < len(parts):
            key = parts[i]
            if key == "depth" and i + 1 < len(parts):
                out["depth"] = int(parts[i + 1]); i += 2
            elif key == "nodes" and i + 1 < len(parts):
                out["nodes"] = int(parts[i + 1]); i += 2
            elif key == "nps" and i + 1 < len(parts):
                out["nps"] = int(parts[i + 1]); i += 2
            elif key == "time" and i + 1 < len(parts):
                out["engine_time_ms"] = int(parts[i + 1]); i += 2
            elif key == "score":
                if i + 2 < len(parts):
                    out["score_type"] = parts[i + 1]
                    try:
                        out["score_val"] = int(parts[i + 2])
                    except ValueError:
                        out["score_val"] = parts[i + 2]
                i += 3
            elif key == "pv" and i + 1 < len(parts):
                # Stops at the first recognized trailing key instead of
                # swallowing bookns/ttns/singns/evalns/probcutns into pv.
                pv_tokens = []
                j = i + 1
                while j < len(parts) and parts[j] not in TRAILING_KEYS:
                    pv_tokens.append(parts[j])
                    j += 1
                out["pv"] = " ".join(pv_tokens)
                i = j
            elif key in TRAILING_KEYS and i + 1 < len(parts):
                try:
                    out[key] = int(parts[i + 1])
                except ValueError:
                    out[key] = parts[i + 1]
                i += 2
            else:
                i += 1
        return out

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def score_to_cp(score_type, score_val):
    """Normalize score_type/score_val to a single centipawn-ish number.
    Mate scores map to a large placeholder with the mating side's sign."""
    if score_type == "mate":
        try:
            mv = int(score_val)
        except (TypeError, ValueError):
            return 0
        return (1 if mv > 0 else -1) * MATE_CP_PLACEHOLDER
    try:
        return int(score_val)
    except (TypeError, ValueError):
        return 0


def human_count(n):
    """Compact human-readable form of a large integer, e.g. 17564138 -> '17.6M'."""
    if n is None:
        return "-"
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{int(n)}"


def format_score(score_type, score_val):
    """Compact, sign-explicit score string: '+132cp', '-45cp', 'mate-3'."""
    if score_type is None or score_val is None:
        return "-"
    try:
        v = int(score_val)
        return f"mate{v:+d}" if score_type == "mate" else f"{v:+d}cp"
    except (TypeError, ValueError):
        return f"{score_type} {score_val}"


def format_sf_diff(sf_cp_lost, sf_mate_involved):
    if sf_cp_lost is None:
        return "-"
    if sf_mate_involved:
        return "mate*"
    return f"{sf_cp_lost:+d}"


def stockfish_probe(sf_engine, fen, movetime_ms):
    """Ask Stockfish for its top move + own eval of that move at `fen`
    (score is from the perspective of the side to move in `fen`)."""
    try:
        bestmove, info, _ = sf_engine.go(fen=fen, movetime=movetime_ms, timeout=movetime_ms / 1000 + 15)
    except TimeoutError:
        return None
    if not info.get("score_type"):
        return None
    return {
        "move": bestmove,
        "score_type": info.get("score_type"),
        "score_val": info.get("score_val"),
        "cp": score_to_cp(info.get("score_type"), info.get("score_val")),
    }


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_game(white_path, black_path, movetime, depth, max_moves, log_rows,
              game_num, total_games, full_log_fh, stderr_dir, fancy,
              sf_engine=None, sf_args=None, sf_stats=None, sf_outliers=None,
              live_board_path=None, report_fh=None):
    white_stderr = open(stderr_dir / f"white_{game_num}.log", "w") if stderr_dir else None
    black_stderr = open(stderr_dir / f"black_{game_num}.log", "w") if stderr_dir else None
    white = UciEngine(white_path, name="white", stderr_dest=white_stderr)
    black = UciEngine(black_path, name="black", stderr_dest=black_stderr)
    white.new_game()
    black.new_game()
    if sf_engine is not None:
        sf_engine.new_game()

    board = chess.Board()
    uci_moves = []
    game_stats = {"white": new_side_stats(), "black": new_side_stats()}

    if live_board_path is not None:
        # Seed the viewer with the start position before the first move
        # comes in, so it's not showing a stale board from a previous game.
        live_board_path.write_text(chess.svg.board(board, size=480))

    tc_desc = f"movetime={movetime}ms" if movetime is not None else f"depth={depth}"
    title = (f"Game {game_num}/{total_games}   "
             f"white={Path(white_path).name}   black={Path(black_path).name}   {tc_desc}")
    table = MoveTable(title, fancy=fancy)

    while not board.is_game_over() and len(uci_moves) < max_moves:
        engine = white if board.turn == chess.WHITE else black
        side = "white" if board.turn == chess.WHITE else "black"
        fen_before = board.fen()

        bestmove, info, all_infos = engine.go(uci_moves, movetime=movetime, depth=depth)

        move = chess.Move.from_uci(bestmove)
        if move not in board.legal_moves:
            table.close([f"!! {side} engine produced illegal move {bestmove} -- game stopped."])
            print(f"    fen_before: {fen_before}")
            break

        san = board.san(move)
        board.push(move)
        fen_after = board.fen()
        uci_moves.append(bestmove)
        ply = len(uci_moves)

        if live_board_path is not None:
            live_board_path.write_text(
                chess.svg.board(board, lastmove=move, size=480)
            )

        # --- Live Stockfish verification (optional) ---
        sf_best_move = sf_best_cp = sf_actual_cp = sf_cp_lost = None
        sf_match = sf_mate_involved = sf_accuracy = sf_class = None
        if sf_engine is not None and sf_args is not None and side in sf_args["sides"] \
                and ply % sf_args["sample_every"] == 0:
            best = stockfish_probe(sf_engine, fen_before, sf_args["time_ms"])
            after = stockfish_probe(sf_engine, fen_after, sf_args["time_ms"])
            if best is not None and after is not None:
                sf_best_move = best["move"]
                sf_best_cp = best["cp"]
                sf_actual_cp = -after["cp"]
                sf_cp_lost = sf_best_cp - sf_actual_cp
                sf_match = (sf_best_move == bestmove)
                sf_mate_involved = (best["score_type"] == "mate" or after["score_type"] == "mate")

                win_before = win_percent(sf_best_cp)
                win_after = win_percent(sf_actual_cp)
                sf_accuracy = move_accuracy(win_before, win_after)
                sf_class = classify_move(sf_match, max(0.0, win_before - win_after))

                for st in (sf_stats[side] if sf_stats is not None else None, game_stats[side]):
                    if st is None:
                        continue
                    st["checked"] += 1
                    st["matches"] += 1 if sf_match else 0
                    st["acc_sum"] += sf_accuracy
                    st["acc_n"] += 1
                    st["class_counts"][sf_class] = st["class_counts"].get(sf_class, 0) + 1
                    if not sf_mate_involved:
                        st["cp_lost_sum"] += sf_cp_lost
                        st["cp_lost_n"] += 1

                if sf_outliers is not None and not sf_mate_involved \
                        and sf_cp_lost is not None and sf_cp_lost >= sf_args["cp_loss_threshold"]:
                    sf_outliers.append({
                        "game": game_num, "ply": ply, "side": side, "san": san,
                        "fen_before": fen_before, "actual_move": bestmove,
                        "sf_best_move": sf_best_move,
                        "sf_best_cp": sf_best_cp, "sf_actual_cp": sf_actual_cp,
                        "sf_cp_lost": sf_cp_lost,
                    })

        wall_time_s = info.get("wall_time_s")
        engine_time_ms = info.get("engine_time_ms")
        time_gap_ms = None
        if wall_time_s is not None and engine_time_ms is not None:
            time_gap_ms = round(wall_time_s * 1000 - engine_time_ms)

        table.row(
            ply,
            side.capitalize(),
            san,
            format_score(info.get("score_type"), info.get("score_val")),
            info.get("depth", "-"),
            human_count(info.get("nodes")),
            human_count(info.get("nps")),
            format_sf_diff(sf_cp_lost, sf_mate_involved),
        )

        log_rows.append({
            "game": game_num,
            "ply": ply,
            "side": side,
            "san": san,
            "uci": bestmove,
            "fen_before": fen_before,
            "depth": info.get("depth"),
            "nodes": info.get("nodes"),
            "nps": info.get("nps"),
            "engine_time_ms": engine_time_ms,
            "wall_time_s": wall_time_s,
            "time_gap_ms": time_gap_ms,
            "score_type": info.get("score_type"),
            "score_val": info.get("score_val"),
            "score_cp": score_to_cp(info.get("score_type"), info.get("score_val")),
            "pv": info.get("pv"),
            "bookns": info.get("bookns"),
            "ttns": info.get("ttns"),
            "singns": info.get("singns"),
            "evalns": info.get("evalns"),
            "probcutns": info.get("probcutns"),
            "sf_best_move": sf_best_move,
            "sf_best_cp": sf_best_cp,
            "sf_actual_cp": sf_actual_cp,
            "sf_cp_lost": sf_cp_lost,
            "sf_match": sf_match,
            "sf_mate_involved": sf_mate_involved,
            "sf_accuracy": sf_accuracy,
            "sf_class": sf_class,
        })

        if full_log_fh is not None:
            full_log_fh.write(json.dumps({
                "game": game_num, "ply": ply, "side": side, "san": san,
                "uci": bestmove, "fen_before": fen_before, "infos": all_infos,
            }) + "\n")
            full_log_fh.flush()

    result = board.result() if board.is_game_over() else "*"
    table.close([f"Result: {result}"])

    review_lines = side_stats_lines("White", game_stats["white"]) \
        + side_stats_lines("Black", game_stats["black"])
    if review_lines:
        print()
        print_panel(f"Game {game_num} review", review_lines, fancy=fancy, report_fh=report_fh)

    white.quit()
    black.quit()
    if white_stderr:
        white_stderr.close()
    if black_stderr:
        black_stderr.close()

    game = chess.pgn.Game()
    game.headers["Event"] = "Self-play test"
    game.headers["White"] = f"engine({Path(white_path).name})"
    game.headers["Black"] = f"engine({Path(black_path).name})"
    game.headers["Result"] = result
    node = game
    for u in uci_moves:
        node = node.add_variation(chess.Move.from_uci(u))

    return game, result


# ---------------------------------------------------------------------------
# End-of-run reports
# ---------------------------------------------------------------------------

def print_blunder_scan(all_rows, blunder_threshold, movetime, overrun_threshold_ms, report_fh=None):
    """Heuristic triage: flag moves where the position swung sharply against
    the side that just moved, or where wall-clock time significantly
    exceeded --movetime (a real deadline-enforcement bug). Not authoritative
    -- always check the FEN."""
    teeprint("\n--- Possible blunders / anomalies (heuristic, verify manually) ---", report_fh)
    flagged = 0
    prev_cp_white_pov = None
    prev_was_mate = False
    for row in all_rows:
        cp = row["score_cp"]
        if cp is None:
            prev_cp_white_pov = None
            prev_was_mate = False
            continue
        white_pov = cp if row["side"] == "white" else -cp
        this_is_mate = row["score_type"] == "mate"

        reasons = []
        if prev_cp_white_pov is not None:
            mover_before = prev_cp_white_pov if row["side"] == "white" else -prev_cp_white_pov
            if this_is_mate or prev_was_mate:
                mover_now_being_mated = this_is_mate and cp < 0
                mover_was_okay_before = mover_before > -blunder_threshold
                if mover_now_being_mated and mover_was_okay_before:
                    reasons.append(
                        f"position went from roughly balanced/favorable ({mover_before}cp) "
                        f"for the mover to a forced mate against them"
                    )
            else:
                swing = cp - mover_before
                if swing <= -blunder_threshold:
                    reasons.append(f"score dropped {abs(swing)}cp for the mover after their own move")

        if movetime is not None and row["wall_time_s"] is not None:
            overrun_ms = row["wall_time_s"] * 1000 - movetime
            if overrun_ms >= overrun_threshold_ms:
                reasons.append(
                    f"move took {overrun_ms:.0f}ms longer than the {movetime}ms "
                    f"time budget -- possible deadline-enforcement bug"
                )

        if reasons:
            flagged += 1
            teeprint(f"  game {row['game']} ply {row['ply']:>3} {row['side']:5} {row['san']:8} "
                     f"score={row['score_type']} {row['score_val']} :: " + "; ".join(reasons), report_fh)
            teeprint(f"      fen_before: {row['fen_before']}", report_fh)

        prev_cp_white_pov = white_pov
        prev_was_mate = this_is_mate

    if flagged == 0:
        teeprint("  (none found above thresholds)", report_fh)


def print_stockfish_summary(sf_stats, sf_outliers, cp_loss_threshold, report_fh=None):
    """Match-rate / avg-cp-lost per side against real Stockfish, plus the
    concrete outlier moves worth root-causing."""
    any_checked = any(s["checked"] > 0 for s in sf_stats.values())
    if not any_checked:
        return

    teeprint("\n--- Live Stockfish verification ---", report_fh)
    for side in ("white", "black"):
        st = sf_stats[side]
        if st["checked"] == 0:
            continue
        match_pct = 100.0 * st["matches"] / st["checked"]
        avg_cp = (st["cp_lost_sum"] / st["cp_lost_n"]) if st["cp_lost_n"] else float("nan")
        teeprint(f"  {side:5}: {st['matches']}/{st['checked']} moves matched Stockfish's top choice "
                 f"({match_pct:.1f}%) | avg cp lost (non-mate positions, n={st['cp_lost_n']}) = {avg_cp:.1f}cp",
                 report_fh)

    if sf_outliers:
        teeprint(f"\n  Concrete errors (cp_lost >= {cp_loss_threshold}, confirmed against Stockfish's own "
                 f"independent eval -- not the engine's self-assessment):", report_fh)
        for o in sorted(sf_outliers, key=lambda r: -r["sf_cp_lost"]):
            teeprint(f"    game {o['game']} ply {o['ply']:>3} {o['side']:5} {o['san']:8} "
                     f":: Stockfish's {o['sf_best_move']} holds {o['sf_best_cp']}cp; "
                     f"played move drops to {o['sf_actual_cp']}cp ({o['sf_cp_lost']}cp lost)", report_fh)
            teeprint(f"        fen_before: {o['fen_before']}", report_fh)
    else:
        teeprint("  (no moves crossed the cp-loss threshold)", report_fh)


def print_engine_review(symmetric, white_path, black_path, sf_stats, results, fancy, report_fh=None):
    """Overall, run-wide review. Symmetric self-play (one binary, both
    colors): combine White+Black stats into one accuracy/Elo estimate for
    THE engine, since a game's White and Black moves are two datapoints for
    the same binary, not two competitors. Asymmetric (--white != --black):
    report each named binary separately, plus a real result-based
    Elo-difference estimate since there IS an opponent here."""
    any_checked = any(s["checked"] > 0 for s in sf_stats.values())
    games = len(results)

    if symmetric:
        combined = merge_side_stats(sf_stats["white"], sf_stats["black"])
        lines = [f"{games} game(s) x 2 colors (same engine both sides) = "
                 f"{combined['checked']} checked-move datapoints"]
        if any_checked:
            lines += side_stats_lines(Path(white_path).name, combined)
        else:
            lines.append("(no Stockfish verification data -- pass --stockfish to enable)")
        print()
        print_panel("Engine review (combined, symmetric self-play)", lines, fancy=fancy, report_fh=report_fh)
    else:
        lines = [f"{games} game(s), white={Path(white_path).name} vs. black={Path(black_path).name}"]
        if any_checked:
            lines += side_stats_lines(f"{Path(white_path).name} (as White)", sf_stats["white"])
            lines += side_stats_lines(f"{Path(black_path).name} (as Black)", sf_stats["black"])
        wins = sum(1 for r in results if r == "1-0")
        losses = sum(1 for r in results if r == "0-1")
        draws = sum(1 for r in results if r == "1/2-1/2")
        decided = wins + losses + draws
        if decided:
            score = (wins + 0.5 * draws) / decided
            diff = elo_diff_from_score(score, decided)
            lines.append(f"\nresult score for {Path(white_path).name} (as White): "
                          f"{wins}W {losses}L {draws}D / {decided} ({score * 100:.1f}%)")
            if diff is not None:
                lines.append(f"estimated Elo difference: {diff:+.0f} "
                              f"(white engine relative to black engine; small-n, wide error bars)")
            else:
                lines.append("estimated Elo difference: undefined (need at least one loss or "
                              "draw to bound it -- more games needed)")
        print()
        print_panel("Engine review (head-to-head)", lines, fancy=fancy, report_fh=report_fh)


def main():
    ap = argparse.ArgumentParser(description="Self-play / review harness for a UCI engine.")
    ap.add_argument("--engine", help="Path to engine binary (used for both sides).")
    ap.add_argument("--white", help="Path to engine binary for White (overrides --engine).")
    ap.add_argument("--black", help="Path to engine binary for Black (overrides --engine).")
    ap.add_argument("--movetime", type=int, default=None, help="Movetime in ms per move.")
    ap.add_argument("--depth", type=int, default=None, help="Fixed search depth per move.")
    ap.add_argument("--max-moves", type=int, default=200, help="Max plies before aborting a game.")
    ap.add_argument("--games", type=int, default=1, help="Number of games to play.")
    ap.add_argument("--out-prefix", default="selfplay", help="Output file prefix.")
    ap.add_argument("--out-dir", default="games",
                     help="Directory all generated files (pgn/csv/jsonl/review/outliers/"
                          "live-board) are written under. Created if missing. Default: games/")
    ap.add_argument("--full-log", action="store_true", default=True,
                     help="Write every info line per move to <out-prefix>_infos.jsonl (default: on).")
    ap.add_argument("--no-full-log", dest="full_log", action="store_false",
                     help="Disable per-depth JSONL logging.")
    ap.add_argument("--stderr-log", default=None,
                     help="Directory to write each engine instance's stderr to (default: discarded).")
    ap.add_argument("--live-board", default="live_board.svg",
                     help="Path to write the live SVG board to after every move, for a "
                          "browser viewer (e.g. live_view.html) to poll. Pass '' to disable.")
    ap.add_argument("--blunder-threshold", type=int, default=250,
                     help="Centipawn self-inflicted drop to flag in the end-of-run scan.")
    ap.add_argument("--overrun-threshold-ms", type=int, default=500,
                     help="How far a move's wall-clock time may exceed --movetime "
                          "before it's flagged as a deadline-enforcement bug.")
    ap.add_argument("--plain", action="store_true", default=False,
                     help="ASCII table borders instead of Unicode box-drawing.")
    ap.add_argument("--stockfish", default=None,
                     help="Path to a real Stockfish binary for live move verification. "
                          "Auto-detects ../Stockfish/stockfish or 'stockfish' on PATH if omitted; "
                          "silently skipped if none is found.")
    ap.add_argument("--no-stockfish-check", dest="stockfish_check", action="store_false", default=True,
                     help="Disable live Stockfish verification even if a binary is available.")
    ap.add_argument("--stockfish-time-ms", type=int, default=1000,
                     help="Time budget per Stockfish query (two queries per checked move).")
    ap.add_argument("--stockfish-side", choices=["white", "black", "both"], default="both",
                     help="Which side's moves to verify against Stockfish.")
    ap.add_argument("--stockfish-sample-every", type=int, default=1,
                     help="Only verify every Nth ply for the selected side(s) (1 = every move).")
    ap.add_argument("--sf-cp-loss-threshold", type=int, default=150,
                     help="cp lost above which a move is a reported concrete error.")
    args = ap.parse_args()

    white_path = args.white or args.engine
    black_path = args.black or args.engine
    if not white_path or not black_path:
        sys.exit("Must specify --engine, or both --white and --black.")
    if args.movetime is None and args.depth is None:
        args.movetime = 5000

    symmetric = (Path(white_path).resolve() == Path(black_path).resolve())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def in_out_dir(path_str):
        """Resolve a user-given path under out_dir, unless it's already
        absolute (so an explicit full path always wins)."""
        p = Path(path_str)
        return p if p.is_absolute() else out_dir / p

    stderr_dir = None
    if args.stderr_log:
        stderr_dir = in_out_dir(args.stderr_log)
        stderr_dir.mkdir(parents=True, exist_ok=True)

    live_board_path = Path(args.live_board) if args.live_board else None
    if live_board_path is not None and not live_board_path.is_absolute():
        # Anchor to this script's own directory (self_test/), not out_dir
        # (self_test/games/) and not the caller's cwd, so the viewer's
        # relative reference always resolves the same way regardless of
        # where the script is invoked from.
        live_board_path = Path(__file__).resolve().parent / live_board_path

    sf_engine = None
    sf_args = None
    sf_stats = {"white": new_side_stats(), "black": new_side_stats()}
    sf_outliers = []
    if args.stockfish_check:
        sf_path = find_stockfish(args.stockfish)
        if sf_path is None:
            print("(Live Stockfish verification skipped: no Stockfish binary found -- "
                  "pass --stockfish PATH, place it at ../Stockfish/stockfish, or put it "
                  "on PATH. Use --no-stockfish-check to silence this.)")
        else:
            print(f"(Live Stockfish verification: using {sf_path}, "
                  f"{args.stockfish_time_ms}ms/position, side={args.stockfish_side})")
            sf_engine = UciEngine(sf_path, name="stockfish", options={"Threads": 1})
            sides = {"white", "black"} if args.stockfish_side == "both" else {args.stockfish_side}
            sf_args = {
                "time_ms": args.stockfish_time_ms,
                "sides": sides,
                "sample_every": max(1, args.stockfish_sample_every),
                "cp_loss_threshold": args.sf_cp_loss_threshold,
            }

    # Auto-numbered so a repeat run with the same --out-prefix (e.g. running
    # the same test command twice) gets its own files instead of clobbering
    # the previous run's pgn/csv/review/outliers. All files live under
    # out_dir.
    run_prefix = str(out_dir / args.out_prefix)
    csv_path = f"{run_prefix}_moves.csv"
    infos_path = f"{run_prefix}_infos.jsonl"
    review_path = f"{run_prefix}_review.txt"

    all_rows = []
    results = []
    pgn_paths = []
    fancy = not args.plain

    full_log_fh = open(infos_path, "w") if args.full_log else None
    report_fh = open(review_path, "w")
    report_fh.write(f"Self-play review -- run prefix: {run_prefix}\n{'=' * 60}\n\n")

    for g in range(1, args.games + 1):
        game, result = play_game(
            white_path, black_path, args.movetime, args.depth,
            args.max_moves, all_rows, g, args.games, full_log_fh, stderr_dir, fancy,
            sf_engine=sf_engine, sf_args=sf_args,
            sf_stats=sf_stats, sf_outliers=sf_outliers,
            live_board_path=live_board_path, report_fh=report_fh,
        )
        results.append(result)
        # Each game gets its own numbered PGN file (game1.pgn, game2.pgn, ...)
        # instead of all games in a run sharing (and overwriting into) one file.
        game_pgn_path = f"{run_prefix}_{g}.pgn"
        with open(game_pgn_path, "w") as pgn_file:
            print(game, file=pgn_file, end="\n\n")
        pgn_paths.append(game_pgn_path)

    if full_log_fh is not None:
        full_log_fh.close()
    if sf_engine is not None:
        sf_engine.quit()

    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        depths = [r["depth"] for r in all_rows if r["depth"] is not None]
        npss = [r["nps"] for r in all_rows if r["nps"] is not None]
        if depths:
            print(f"\nSummary: {len(all_rows)} moves | "
                  f"avg depth={sum(depths)/len(depths):.1f} "
                  f"(min={min(depths)}, max={max(depths)}) | "
                  + (f"avg nps={sum(npss)/len(npss):,.0f}" if npss else ""))

        print_blunder_scan(all_rows, args.blunder_threshold, args.movetime, args.overrun_threshold_ms,
                            report_fh=report_fh)

    if sf_args is not None:
        print_stockfish_summary(sf_stats, sf_outliers, sf_args["cp_loss_threshold"], report_fh=report_fh)
        if sf_outliers:
            outliers_path = f"{run_prefix}_stockfish_outliers.json"
            with open(outliers_path, "w") as f:
                json.dump(sf_outliers, f, indent=2)
            print(f"Saved confirmed-error details to {outliers_path}")

    print_engine_review(symmetric, white_path, black_path, sf_stats, results, fancy, report_fh=report_fh)

    report_fh.close()

    if len(pgn_paths) == 1:
        print(f"\nSaved PGN to {pgn_paths[0]}")
    else:
        print(f"\nSaved {len(pgn_paths)} PGNs: {pgn_paths[0]} .. {pgn_paths[-1]}")
    print(f"Saved per-move log to {csv_path}")
    if args.full_log:
        print(f"Saved per-depth info log to {infos_path}")
    if stderr_dir:
        print(f"Saved engine stderr logs to {stderr_dir}/")
    if live_board_path:
        print(f"Live board written to {live_board_path}")
    print(f"Saved reviews + summary to {review_path}")


if __name__ == "__main__":
    main()
