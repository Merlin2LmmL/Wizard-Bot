#!/usr/bin/env python3
"""
Root-cause diagnostic for confirmed cp-loss outliers from self_play.py's
live Stockfish verification, using only the UCI protocol (no engine source
needed).

IMPORTANT: an earlier version of this script tried to isolate a candidate
move with UCI's `go ... searchmoves <move>`. Testing against the real
engine showed it does NOT honor that restriction -- asked to search only
`c8d8`, it returned `f4e2` as bestmove, which is only possible if
`searchmoves` was silently ignored (the codebase recap's description of
`handle_go` never mentions it either). So that approach is removed here --
it was silently producing a normal full search every time and mislabeling
the result as "the engine's isolated opinion of this move", which is not
what it was.

Instead, this uses a technique that only needs plain `go` (nothing an
engine could silently ignore): it FORCES the candidate move itself (with
python-chess, off to the side, no engine involvement) and then asks the
engine for its own normal full search of the resulting position. That
tells us the engine's own opinion of "if I had played this move, how good
would my position be" -- flip the sign and compare to what Stockfish
thinks that same resulting position is worth.

  - If the engine's own eval of the resulting position roughly agrees with
    Stockfish (it correctly judges the position AFTER the candidate move),
    but it never chose that move at the root -- the search's own evaluator
    isn't the problem. The failure is that the ROOT search never properly
    explores/keeps that branch. Look at move ordering (is the candidate
    tried too late?) and negamax.rs's LMR / futility / late-move-pruning
    margins -- something is cutting this branch before it gets the search
    time to reveal it's actually fine.

  - If the engine's own eval of the resulting position ALSO disagrees with
    Stockfish (same misjudgment it made at the root), the bug isn't a root
    artifact -- the engine's normal search machinery, even when it does
    look at this exact line, comes to the wrong conclusion. That points
    further down the tree: check extensions, quiescence horizon, or a
    line-specific eval/search issue in that particular subtree.

A secondary, best-effort searchmoves probe is still attempted (some
engines do support it) but its result is only trusted if the returned
bestmove is actually inside the requested restriction -- otherwise it's
printed as informational-only and excluded from the verdict.

Usage:
    python3 investigate_outliers.py --engine ../target/release/uci_binary \\
        --outliers-json selfplay_stockfish_outliers.json

    # outliers.json from an older self_play.py that didn't store actual_move:
    python3 investigate_outliers.py --engine ../target/release/uci_binary \\
        --outliers-json selfplay_stockfish_outliers.json \\
        --moves-csv selfplay_moves.csv --side black

    # Narrow to black's outliers only, use more time per probe:
    python3 investigate_outliers.py --engine ../target/release/uci_binary \\
        --outliers-json selfplay_stockfish_outliers.json \\
        --side black --movetime 5000

    # Just one FEN, ad hoc (skip the outliers file):
    python3 investigate_outliers.py --engine ../target/release/uci_binary \\
        --fen "2r1r1k1/2p2p1p/p5p1/BpR1P3/3qBnb1/8/PPQ2PPP/4R1K1 b - - 12 29" \\
        --actual-move f4e2 --candidate-move c8d8
"""
import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import chess
except ImportError:
    sys.exit("Missing dependency. Install with:\n  pip install python-chess --break-system-packages")

# Reuse the UciEngine wrapper instead of duplicating it -- keep both files
# in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_play import UciEngine, score_to_cp  # noqa: E402

SEARCHMOVES_WARNED = False  # print the "not honored" note only once


def probe_plain(engine, fen, movetime_ms):
    """Plain `go` at an arbitrary FEN -- nothing an engine could partially
    ignore. Returns the engine's own bestmove + score (from the side-to-move
    in `fen`'s perspective)."""
    try:
        bestmove, info, _ = engine.go(fen=fen, movetime=movetime_ms,
                                       timeout=movetime_ms / 1000 + 20)
    except TimeoutError as e:
        return {"error": str(e)}
    return {
        "move": bestmove,
        "score_type": info.get("score_type"),
        "score_val": info.get("score_val"),
        "cp": score_to_cp(info.get("score_type"), info.get("score_val")),
        "depth": info.get("depth"),
        "nodes": info.get("nodes"),
        "pv": info.get("pv"),
    }


def probe_resulting_position(engine, fen, move_uci, movetime_ms):
    """Force `move_uci` onto the board ourselves (no engine involvement),
    then ask the engine to normally search the resulting position. The
    returned score is from the OPPONENT's perspective (they're now to
    move); flip its sign to get the original mover's-perspective value of
    having played move_uci -- this is the engine's own opinion of that
    move, using nothing but a plain, unignorable `go`."""
    board = chess.Board(fen)
    try:
        board.push_uci(move_uci)
    except (ValueError, AssertionError) as e:
        return {"error": f"illegal move {move_uci} at this fen: {e}"}
    result = probe_plain(engine, board.fen(), movetime_ms)
    if "error" in result:
        return result
    result["cp_from_mover"] = -result["cp"]
    result["resulting_fen"] = board.fen()
    return result


def probe_searchmoves(engine, fen, moves, movetime_ms):
    """Best-effort searchmoves probe. Caller must check whether the
    returned bestmove actually falls inside `moves` before trusting it --
    some engines silently ignore this UCI field entirely."""
    try:
        bestmove, info, _ = engine.go(fen=fen, movetime=movetime_ms,
                                       timeout=movetime_ms / 1000 + 20,
                                       searchmoves=moves)
    except TimeoutError as e:
        return {"error": str(e)}
    return {
        "move": bestmove,
        "cp": score_to_cp(info.get("score_type"), info.get("score_val")),
        "honored": bestmove in moves,
    }


def load_actual_moves_index(csv_path):
    """Map (game, ply) -> uci move, from a self_play.py _moves.csv file.
    Used to backfill 'actual_move' for outliers.json files produced by an
    older self_play.py that didn't yet store it directly."""
    index = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["game"]), int(row["ply"]))
            except (KeyError, ValueError):
                continue
            index[key] = row.get("uci")
    return index


def diagnose_one(engine, item, movetime_ms, tolerance_cp=40, try_searchmoves=True):
    global SEARCHMOVES_WARNED
    fen = item["fen_before"]
    actual = item.get("actual_move")
    if not actual:
        print(f"\n=== game {item.get('game','?')} ply {item.get('ply','?')} "
              f"{item.get('side','?')} {item.get('san','?')} -- SKIPPED ===")
        print("    No 'actual_move' for this outlier (older outliers.json format) "
              "and no --moves-csv match found for it. Pass --moves-csv "
              "<out-prefix>_moves.csv from the same run to backfill it, or "
              "re-run self_play.py to regenerate outliers.json with the current version.")
        return None
    candidate = item["sf_best_move"]
    sf_best_cp = item["sf_best_cp"]

    print(f"\n=== game {item.get('game','?')} ply {item.get('ply','?')} "
          f"{item.get('side','?')} {item.get('san','?')} "
          f"(cp_lost={item.get('sf_cp_lost','?')}) ===")
    print(f"    fen: {fen}")
    print(f"    Stockfish says: {candidate} holds {sf_best_cp}cp  |  "
          f"engine played: {actual} (dropped to {item.get('sf_actual_cp','?')}cp)")

    if candidate == actual:
        print("    (Stockfish's move == the move played -- nothing to isolate here, skipping.)")
        return None

    # Primary, reliable method: force the candidate ourselves, ask the
    # engine's own normal search what it thinks of the resulting position.
    eng_after_candidate = probe_resulting_position(engine, fen, candidate, movetime_ms)
    eng_after_actual = probe_resulting_position(engine, fen, actual, movetime_ms)

    if "error" in eng_after_candidate:
        print(f"    engine's own eval after candidate: ERROR -- {eng_after_candidate['error']}")
        return None
    if "error" in eng_after_actual:
        print(f"    engine's own eval after actual: ERROR -- {eng_after_actual['error']}")
        return None

    print(f"    engine's own eval after playing candidate ({candidate}): "
          f"{eng_after_candidate['cp_from_mover']:>6}cp  (opponent's best reply per engine: "
          f"{eng_after_candidate.get('pv','')})")
    print(f"    engine's own eval after playing actual    ({actual}): "
          f"{eng_after_actual['cp_from_mover']:>6}cp  (opponent's best reply per engine: "
          f"{eng_after_actual.get('pv','')})  [sanity check -- should be near what it originally scored the game move]")

    cand_cp = eng_after_candidate["cp_from_mover"]
    if abs(cand_cp - sf_best_cp) <= tolerance_cp:
        verdict = ("ROOT-LEVEL ORDERING/PRUNING: the engine's OWN full search of the position "
                   "after the candidate move agrees with Stockfish that it's fine "
                   f"({cand_cp}cp vs. Stockfish's {sf_best_cp}cp) -- its evaluator and search "
                   "are not the problem here. It just never explores/keeps that branch at the "
                   "root during the real game. Look at ordering.rs (is the candidate tried "
                   "late in the move list?) and negamax.rs's LMR/futility/late-move-pruning "
                   "margins, which could be cutting the branch before its true value surfaces.")
    else:
        verdict = (f"DEEPER SEARCH/EVAL BUG: even the engine's own normal search of the position "
                   f"AFTER the candidate move misjudges it ({cand_cp}cp vs. Stockfish's "
                   f"{sf_best_cp}cp, off by {abs(cand_cp - sf_best_cp)}cp). This isn't a root "
                   "artifact -- the engine's search machinery gets the wrong answer even when "
                   "it's squarely looking at this exact line. Compare the opponent's-reply PV "
                   "printed above against what Stockfish expects there; the mismatch is likely "
                   "further down that specific subtree (an extension, quiescence horizon, or "
                   "eval issue reachable in normal play, not unique to the root).")

    print(f"    -> {verdict}")

    if try_searchmoves:
        sm_candidate = probe_searchmoves(engine, fen, [candidate], movetime_ms)
        if "error" not in sm_candidate:
            if sm_candidate["honored"]:
                print(f"    (searchmoves probe honored: candidate alone scores {sm_candidate['cp']}cp)")
            else:
                if not SEARCHMOVES_WARNED:
                    print("    NOTE: this engine does not appear to honor UCI 'go searchmoves' "
                          "(returned a bestmove outside the requested restriction) -- "
                          "searchmoves-based probes are informational only from here on.")
                    SEARCHMOVES_WARNED = True

    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="Path to the engine binary under investigation.")
    ap.add_argument("--outliers-json", help="Path to <out-prefix>_stockfish_outliers.json from self_play.py.")
    ap.add_argument("--moves-csv", default=None,
                     help="Path to <out-prefix>_moves.csv from the SAME run as --outliers-json. "
                          "Only needed to backfill 'actual_move' if the outliers.json predates "
                          "that field (older self_play.py version) -- current self_play.py writes "
                          "it directly and this isn't needed.")
    ap.add_argument("--fen", help="Investigate a single ad hoc FEN instead of an outliers file.")
    ap.add_argument("--actual-move", help="UCI move actually played (required with --fen).")
    ap.add_argument("--candidate-move", help="UCI move to compare against (required with --fen).")
    ap.add_argument("--side", choices=["white", "black", "both"], default="both",
                     help="Only diagnose outliers from this side (only applies with --outliers-json).")
    ap.add_argument("--top", type=int, default=None,
                     help="Only diagnose the top N outliers by cp_lost (only applies with --outliers-json).")
    ap.add_argument("--movetime", type=int, default=5000,
                     help="Time budget per probe (each probe is one normal full search of the "
                          "resulting position, so this can roughly match a real game's movetime; "
                          "default is generous to reduce depth-related noise).")
    ap.add_argument("--no-searchmoves-probe", dest="try_searchmoves", action="store_false", default=True,
                     help="Skip the best-effort searchmoves probe entirely (it's informational-only "
                          "anyway once honoring is confirmed false; skip to save time).")
    args = ap.parse_args()

    if not args.outliers_json and not args.fen:
        sys.exit("Must specify --outliers-json or --fen (+ --actual-move --candidate-move).")

    engine = UciEngine(args.engine, name="engine")
    engine.new_game()

    items = []
    if args.fen:
        if not args.actual_move or not args.candidate_move:
            sys.exit("--fen requires --actual-move and --candidate-move.")
        items = [{
            "fen_before": args.fen, "actual_move": args.actual_move,
            "sf_best_move": args.candidate_move, "sf_best_cp": None, "sf_actual_cp": None,
            "sf_cp_lost": None,
        }]
    else:
        with open(args.outliers_json) as f:
            items = json.load(f)
        if any("actual_move" not in i for i in items) and args.moves_csv:
            move_index = load_actual_moves_index(args.moves_csv)
            for i in items:
                if "actual_move" not in i:
                    key = (i.get("game"), i.get("ply"))
                    if key in move_index:
                        i["actual_move"] = move_index[key]
        if args.side != "both":
            items = [i for i in items if i.get("side") == args.side]
        items.sort(key=lambda i: -(i.get("sf_cp_lost") or 0))
        if args.top:
            items = items[: args.top]

    if not items:
        print("No outliers to investigate (check --side / --outliers-json contents).")
        engine.quit()
        return

    verdicts = []
    for item in items:
        v = diagnose_one(engine, item, args.movetime, try_searchmoves=args.try_searchmoves)
        if v:
            verdicts.append(v)

    engine.quit()

    if verdicts:
        ordering_n = sum(1 for v in verdicts if v.startswith("ROOT-LEVEL"))
        deeper_n = sum(1 for v in verdicts if v.startswith("DEEPER"))
        print(f"\n--- Summary across {len(verdicts)} diagnosed outliers ---")
        print(f"  Root-level ordering/pruning (engine's own eval agrees w/ Stockfish once it looks): {ordering_n}")
        print(f"  Deeper search/eval bug (engine's own eval disagrees even when it looks): {deeper_n}")
        if ordering_n >= deeper_n and ordering_n > 0:
            print("  -> Majority pattern points at move ordering / pruning constants "
                  "(ordering.rs, negamax.rs's LMR/futility/LMP margins) as the next place to look.")
        elif deeper_n > 0:
            print("  -> Majority pattern points at a deeper per-line search/eval issue -- "
                  "compare the opponent's-reply PVs printed above against Stockfish's own "
                  "expectation at those specific resulting positions.")


if __name__ == "__main__":
    main()
