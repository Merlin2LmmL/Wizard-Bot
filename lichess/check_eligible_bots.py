#!/usr/bin/env python3
"""
check_eligible_bots.py — diagnostic: how many online bots actually pass your
autoqueue filters right now, and how big the raw candidate pool is.

Usage:
  export LICHESS_TOKEN=xxxxx      # optional, works unauthenticated too
  python3 check_eligible_bots.py --config config.json
"""
import argparse
import json
import os
import requests

def speed_bucket(limit_secs, increment_secs):
    total = limit_secs + 40 * increment_secs
    if total < 30:
        return "ultraBullet"
    if total < 180:
        return "bullet"
    if total < 480:
        return "blitz"
    if total < 1500:
        return "rapid"
    return "classical"


def rating_allows(rr, opponent_rating, my_rating):
    mode = rr.get("mode", "off")
    if mode == "off":
        return True
    if opponent_rating is None:
        return False
    if mode == "static":
        if rr.get("min_elo") is not None and opponent_rating < rr["min_elo"]:
            return False
        if rr.get("max_elo") is not None and opponent_rating > rr["max_elo"]:
            return False
        return True
    if mode == "dynamic":
        if my_rating is None:
            return True
        diff = opponent_rating - my_rating
        lo = rr.get("min_diff")
        if lo is None and rr.get("max_diff") is not None:
            lo = -rr["max_diff"]
        hi = rr.get("max_diff")
        if lo is not None and diff < lo:
            return False
        if hi is not None and diff > hi:
            return False
        return True
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--nb", type=int, default=None,
                     help="override candidates_per_poll, e.g. --nb 500 to see the true pool size")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    aq = cfg["autoqueue"]
    token = os.environ.get("LICHESS_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    nb = args.nb or aq.get("candidates_per_poll", 50)
    resp = requests.get(f"https://lichess.org/api/bot/online?nb={nb}", headers=headers)
    resp.raise_for_status()
    bots = [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    speed = speed_bucket(aq["speed_clock"]["limit"], aq["speed_clock"]["increment"])
    rr = aq.get("rating_range", {"mode": "off"})
    min_games = aq.get("min_games", 0)

    print(f"Speed bucket for your clock ({aq['speed_clock']}): {speed}")
    print(f"Fetched {len(bots)} online bots (nb={nb})")

    eligible = []
    reasons = {"no_perf": 0, "few_games": 0, "rating_out_of_range": 0}
    for bot in bots:
        username = bot.get("username") or bot.get("id")
        perf = (bot.get("perfs") or {}).get(speed) or {}
        games = perf.get("games", 0)
        rating = perf.get("rating")
        if rating is None:
            reasons["no_perf"] += 1
            continue
        if games < min_games:
            reasons["few_games"] += 1
            continue
        if not rating_allows(rr, rating, None):
            reasons["rating_out_of_range"] += 1
            continue
        eligible.append((username, rating, games))

    print(f"\nEligible right now: {len(eligible)}")
    for u, r, g in sorted(eligible, key=lambda x: -x[1]):
        print(f"  {u:<25} rating={r:<5} games={g}")

    print("\nFiltered out because:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    if args.nb is None:
        print(f"\nNote: this only looked at the top {nb} 'recently online' bots "
              f"(candidates_per_poll in your config). Re-run with --nb 500 or so "
              f"to see how much bigger the true eligible pool is once you widen the fetch.")


if __name__ == "__main__":
    main()
