#!/usr/bin/env python3
"""
lichess_bot.py — minimal driver that plugs a raw UCI binary (e.g. wizardbot_engine's
uci_binary) into the Lichess Bot API.

Features
  - Event stream: accepts/declines incoming challenges based on config rules.
  - Plays up to N games concurrently, each in its own thread, each with its own
    engine subprocess.
  - Auto-queue: optionally goes looking for other online bots and challenges them,
    respecting a concurrency cap and a per-bot cooldown list.
  - Cooldown list: when Lichess replies "please wait until <timestamp>" (daily cap,
    already-challenged, etc.), the bot is parked in cooldowns.json until then.
  - Global rate-limit safety: any HTTP 429 triggers an exponential backoff during
    which the bot issues no new outgoing requests (existing games keep playing).

Usage
  export LICHESS_TOKEN=xxxxx
  python3 lichess_bot.py --engine /path/to/uci_binary --config config.example.json

Only needs `requests` (pip install requests).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

LICHESS_BASE = "https://lichess.org"
COOLDOWN_FILE = "cooldowns.json"
LOG_DIR = "logs"
LOG = logging.getLogger("lichess_bot")
# Bumped whenever the autoqueue/engine-lifecycle logic changes, and printed at
# startup — makes it trivial to confirm you're actually running the file you
# think you are, instead of debugging symptoms from an old copy.
SCRIPT_VERSION = "2026-09-14.3-pending-challenge-fix"


class ConsoleFormatter(logging.Formatter):
    """Keeps the console quiet: plain messages for INFO (no timestamp, no
    'INFO' tag taking up space), but WARNING/ERROR still get a visible tag.
    Exceptions logged via LOG.exception(...) still get their traceback
    printed — dropping it here would silently hide the real cause of any
    "Error handling an event" style messages."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno == logging.INFO:
            return message
        line = f"{record.levelname}: {message}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class RatingRange:
    """Restricts which opponents we'll play, by rating.

    mode "off"     — no restriction (default).
    mode "static"  — opponent rating must fall within [min_elo, max_elo].
    mode "dynamic" — opponent rating must fall within
                     [my_rating + min_diff, my_rating + max_diff], recomputed
                     against our own current rating for that speed each time.
                     min_diff defaults to -max_diff (a symmetric band) if not
                     given, e.g. max_diff=150 alone means "opponents within
                     150 points either side of me".
    """
    mode: str = "off"
    min_elo: Optional[int] = None
    max_elo: Optional[int] = None
    min_diff: Optional[int] = None
    max_diff: Optional[int] = None

    def allows(self, opponent_rating: Optional[int], my_rating: Optional[int]) -> bool:
        if self.mode == "off":
            return True
        if opponent_rating is None:
            return False  # can't verify a range restriction against an unknown rating
        if self.mode == "static":
            if self.min_elo is not None and opponent_rating < self.min_elo:
                return False
            if self.max_elo is not None and opponent_rating > self.max_elo:
                return False
            return True
        if self.mode == "dynamic":
            if my_rating is None:
                return True  # nothing to compare against yet; don't block on it
            diff = opponent_rating - my_rating
            lo = self.min_diff if self.min_diff is not None else \
                 (-self.max_diff if self.max_diff is not None else None)
            hi = self.max_diff
            if lo is not None and diff < lo:
                return False
            if hi is not None and diff > hi:
                return False
            return True
        return True


def _parse_rating_range(raw: Optional[dict]) -> RatingRange:
    return RatingRange(**raw) if raw else RatingRange()


def _speed_bucket(limit_secs: int, increment_secs: int) -> str:
    """Mirrors Lichess's own speed classification so we can look up the
    matching perf rating (perfs.bullet.rating, perfs.blitz.rating, ...)."""
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


@dataclass
class AcceptRules:
    """What incoming challenges we'll say yes to."""
    variants: list = field(default_factory=lambda: ["standard"])
    speeds: list = field(default_factory=lambda: ["bullet", "blitz", "rapid"])
    rated: Optional[bool] = None          # None = accept both rated and casual
    only_bots: bool = False               # require challenger to be a BOT account
    min_initial_secs: int = 15
    max_initial_secs: int = 1800
    max_increment_secs: int = 60
    decline_reason: str = "generic"       # lichess decline reason code
    rating_range: RatingRange = field(default_factory=RatingRange)

def _parse_time_control(entry) -> dict:
    """Accepts either a {"limit": secs, "increment": secs} dict, or the usual
    chess shorthand as a string, e.g. "5+3" (5 minutes, 3s increment per
    move) or "3+0". The shorthand's left side is always MINUTES, right side
    is increment in SECONDS — matching how people actually write time
    controls, even though clock_limit on the wire is seconds."""
    if isinstance(entry, dict):
        if "limit" not in entry:
            raise ValueError(f"Time control dict missing 'limit': {entry!r}")
        return {"limit": int(entry["limit"]), "increment": int(entry.get("increment", 0))}
    if isinstance(entry, str):
        m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", entry)
        if not m:
            raise ValueError(
                f"Invalid time control {entry!r}: expected 'MINUTES+INCREMENT', e.g. '5+3'"
            )
        minutes, increment = int(m.group(1)), int(m.group(2))
        return {"limit": minutes * 60, "increment": increment}
    raise ValueError(f"Invalid time control entry: {entry!r}")


def _format_time_control(tc: dict) -> str:
    return f"{tc['limit'] // 60}+{tc['increment']}"


@dataclass
class AutoQueueRules:
    enabled: bool = False
    poll_interval_secs: int = 90
    max_pending_challenges: int = 3
    variant: str = "standard"
    # One or more time controls to rotate through when auto-queueing, so a
    # bot that doesn't fit your rating band at one speed can still match at
    # another. Each entry is either {"limit": secs, "increment": secs} or the
    # shorthand "M+S" (e.g. "5+3", "3+1", "1+1").
    time_controls: list = field(default_factory=lambda: [{"limit": 300, "increment": 3}])
    rated: bool = True
    candidates_per_poll: int = 50         # how many online bots to fetch each pass
    min_seconds_between_challenges: int = 20
    rating_range: RatingRange = field(default_factory=RatingRange)
    min_games: int = 0                    # skip candidates with fewer games in this
                                           # speed/variant than this — a bot's rating
                                           # there is otherwise just an untested default
                                           # (e.g. a variants specialist parked at 1500
                                           # in standard blitz because it barely plays it)

@dataclass
class BotConfig:
    token: str = ""
    engine_path: str = ""
    engine_args: list = field(default_factory=list)
    max_concurrent_games: int = 4
    accept: AcceptRules = field(default_factory=AcceptRules)
    autoqueue: AutoQueueRules = field(default_factory=AutoQueueRules)
    base_backoff_secs: float = 5.0
    max_backoff_secs: float = 600.0
    cooldown_file: str = COOLDOWN_FILE

    @staticmethod
    def load(path: Optional[str]) -> "BotConfig":
        cfg = BotConfig()
        if path and os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            cfg.token = raw.get("token", cfg.token)
            cfg.engine_path = raw.get("engine_path", cfg.engine_path)
            cfg.engine_args = raw.get("engine_args", cfg.engine_args)
            cfg.max_concurrent_games = raw.get("max_concurrent_games", cfg.max_concurrent_games)
            cfg.base_backoff_secs = raw.get("base_backoff_secs", cfg.base_backoff_secs)
            cfg.max_backoff_secs = raw.get("max_backoff_secs", cfg.max_backoff_secs)
            cfg.cooldown_file = raw.get("cooldown_file", cfg.cooldown_file)
            if "accept" in raw:
                accept_raw = dict(raw["accept"])
                rr = _parse_rating_range(accept_raw.pop("rating_range", None))
                cfg.accept = AcceptRules(**{**cfg.accept.__dict__, **accept_raw})
                cfg.accept.rating_range = rr
            if "autoqueue" in raw:
                aq_raw = dict(raw["autoqueue"])
                rr = _parse_rating_range(aq_raw.pop("rating_range", None))
                old_speed_clock = aq_raw.pop("speed_clock", None)
                if old_speed_clock is not None and "time_controls" not in aq_raw:
                    # Back-compat: old configs had a single speed_clock dict.
                    aq_raw["time_controls"] = [old_speed_clock]
                if "time_controls" in aq_raw:
                    aq_raw["time_controls"] = [_parse_time_control(tc) for tc in aq_raw["time_controls"]]
                cfg.autoqueue = AutoQueueRules(**{**cfg.autoqueue.__dict__, **aq_raw})
                cfg.autoqueue.rating_range = rr
        return cfg


# --------------------------------------------------------------------------- #
# Rate limiter / cooldown bookkeeping
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Global outgoing-request guard. Trips on HTTP 429, backs off exponentially,
    and blocks all *new* outgoing requests (challenges, autoqueue, etc.) until the
    backoff window passes. Doesn't touch already-open game move/event streams."""

    def __init__(self, base_backoff: float, max_backoff: float):
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        self._backoff = base_backoff
        self._base = base_backoff
        self._max = max_backoff

    def wait_if_blocked(self):
        while True:
            with self._lock:
                remaining = self._blocked_until - time.time()
            if remaining <= 0:
                return
            LOG.warning("Rate limiter active, sleeping %.1fs", remaining)
            time.sleep(min(remaining, 5.0))

    def note_response(self, resp: requests.Response):
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            with self._lock:
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = self._backoff
                else:
                    delay = self._backoff
                self._blocked_until = time.time() + delay
                self._backoff = min(self._backoff * 2, self._max)
            LOG.error("Got 429 — backing off %.1fs (next backoff %.1fs)", delay, self._backoff)
        elif resp.ok:
            with self._lock:
                self._backoff = self._base  # reset on any success


class CooldownStore:
    """Persisted {username: iso_timestamp} of bots we should not challenge yet."""

    _TS_RE = re.compile(r"please wait until ([0-9T:\.\-Z]+)")

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_cooling_down(self, username: str) -> bool:
        with self._lock:
            until = self._data.get(username.lower())
        if not until:
            return False
        try:
            dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError:
            return False
        return datetime.now(timezone.utc) < dt

    def set_cooldown(self, username: str, until_iso: str):
        with self._lock:
            self._data[username.lower()] = until_iso
            self._save()
        LOG.info("Cooldown: won't challenge %s again until %s", username, until_iso)

    def record_from_error_text(self, username: str, text: str) -> bool:
        """Look for '...please wait until <ISO timestamp>...' in an API error body
        and, if found, store it. Returns True if a cooldown was recorded."""
        m = self._TS_RE.search(text)
        if m:
            self.set_cooldown(username, m.group(1))
            return True
        return False


# --------------------------------------------------------------------------- #
# Lichess API wrapper
# --------------------------------------------------------------------------- #

class LichessAPI:
    def __init__(self, token: str, limiter: RateLimiter):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.limiter = limiter

    def _request(self, method: str, path: str, max_attempts: int = 5,
                 **kwargs) -> requests.Response:
        """Send a request, retrying on transient network errors (DNS blips,
        dropped connections, timeouts) with short exponential backoff. This is
        separate from RateLimiter, which only reacts to HTTP 429 — this handles
        the case where the request never even got a response."""
        backoff = 1.0
        for attempt in range(1, max_attempts + 1):
            self.limiter.wait_if_blocked()
            try:
                resp = self.session.request(method, LICHESS_BASE + path, timeout=30, **kwargs)
            except requests.exceptions.RequestException as e:
                if attempt == max_attempts:
                    LOG.error("Giving up on %s %s after %d attempts: %s",
                              method, path, attempt, e)
                    raise
                LOG.warning("Network error on %s %s (attempt %d/%d): %s — retrying in %.1fs",
                            method, path, attempt, max_attempts, e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            self.limiter.note_response(resp)
            return resp

    def stream_lines(self, path: str):
        """Yield decoded NDJSON lines from a streaming endpoint forever (with
        auto-reconnect on drop)."""
        while True:
            self.limiter.wait_if_blocked()
            try:
                with self.session.get(LICHESS_BASE + path, stream=True, timeout=None) as resp:
                    self.limiter.note_response(resp)
                    if resp.status_code == 429:
                        continue
                    for line in resp.iter_lines():
                        if line:
                            yield line.decode("utf-8")
            except requests.RequestException as e:
                LOG.warning("Stream %s dropped (%s), reconnecting in 5s", path, e)
                time.sleep(5)

    # -- account / discovery -------------------------------------------------
    def my_profile(self) -> dict:
        return self._request("GET", "/api/account").json()

    def online_bots(self, nb: int = 50) -> list[dict]:
        resp = self._request("GET", f"/api/bot/online?nb={nb}")
        bots = []
        for line in resp.text.splitlines():
            if line.strip():
                try:
                    bots.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return bots

    # -- challenges -----------------------------------------------------------
    def accept_challenge(self, challenge_id: str) -> requests.Response:
        return self._request("POST", f"/api/challenge/{challenge_id}/accept")

    def decline_challenge(self, challenge_id: str, reason: str) -> requests.Response:
        return self._request("POST", f"/api/challenge/{challenge_id}/decline",
                              data={"reason": reason})

    def create_challenge(self, username: str, rated: bool, clock_limit: int,
                          clock_increment: int, variant: str = "standard") -> requests.Response:
        return self._request("POST", f"/api/challenge/{username}", data={
            "rated": str(rated).lower(),
            "clock.limit": clock_limit,
            "clock.increment": clock_increment,
            "variant": variant,
            "color": "random",
        })

    # -- games ------------------------------------------------------------
    def make_move(self, game_id: str, uci_move: str) -> requests.Response:
        return self._request("POST", f"/api/bot/game/{game_id}/move/{uci_move}")


# --------------------------------------------------------------------------- #
# UCI engine wrapper
# --------------------------------------------------------------------------- #

class UciEngine:
    def __init__(self, path: str, args: list[str]):
        self.proc = subprocess.Popen(
            [path, *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
            # Ctrl+C in the terminal sends SIGINT to the whole foreground
            # process group, which by default includes this child. Most UCI
            # engines don't handle SIGINT and just die — then the *next*
            # best_move() call gets no response at all (stdout hits EOF).
            # start_new_session detaches the engine into its own session so
            # only an explicit quit() (below) ever stops it.
            start_new_session=(os.name == "posix"),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        if os.name == "posix":
            # Verify the detach actually took effect — if this ever prints
            # the warning, the engine WILL die on the next Ctrl+C, same as
            # before the fix, so this is worth checking after any change to
            # how the process is launched (containers, process managers,
            # wrapper scripts, etc. can all interfere with setsid()).
            try:
                same_session = os.getsid(self.proc.pid) == os.getsid(0)
            except OSError:
                same_session = None
            if same_session:
                LOG.warning("Engine process did not get its own session — it will still be "
                            "killed by Ctrl+C. Check for a wrapper/supervisor around this "
                            "script that resets the child's process group.")
            else:
                LOG.debug("Engine pid %d detached into its own session (sid=%s) — insulated "
                          "from terminal signals.", self.proc.pid, os.getsid(self.proc.pid))
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            if token in line:
                return
        LOG.warning("Engine did not respond with '%s' in time", token)

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    @staticmethod
    def _parse_info(line: str) -> Optional[dict]:
        """Parse a UCI 'info ...' line into {depth, seldepth, nodes, nps, time,
        score: {'cp': n} or {'mate': n}, pv: [...]}. Returns None if the line
        has no score (e.g. currmove/refutation chatter we don't care about)."""
        tokens = line.split()
        if not tokens or tokens[0] != "info":
            return None
        info: dict = {}
        i = 1
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            if tok in ("depth", "seldepth", "multipv", "nodes", "nps", "time", "hashfull", "tbhits"):
                if i + 1 < n and tokens[i + 1].lstrip("-").isdigit():
                    info[tok] = int(tokens[i + 1])
                i += 2
            elif tok == "score":
                if i + 2 < n and tokens[i + 1] in ("cp", "mate"):
                    try:
                        info["score"] = {tokens[i + 1]: int(tokens[i + 2])}
                    except ValueError:
                        pass
                    i += 3
                else:
                    i += 1
            elif tok == "pv":
                info["pv"] = tokens[i + 1:]
                break
            else:
                i += 1
        return info if "score" in info else None

    def best_move(self, moves: list[str], wtime: int, btime: int,
                   winc: int, binc: int) -> tuple[Optional[str], Optional[dict], float]:
        """Returns (bestmove, last_info_with_score, think_time_seconds)."""
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        started = time.time()
        self._send(f"go wtime {wtime} btime {btime} winc {winc} binc {binc}")
        deadline = started + max((wtime + btime) / 1000.0, 30) + 10
        last_info: Optional[dict] = None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            if line.startswith("info"):
                info = self._parse_info(line)
                if info:
                    last_info = info
            elif line.startswith("bestmove"):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else None
                return move, last_info, time.time() - started
        return None, last_info, time.time() - started

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------------- #
# Formatting helpers for the per-game move log
# --------------------------------------------------------------------------- #

def _fmt_count(n: Optional[int]) -> str:
    """Abbreviate a node/nps count (e.g. 79,160,120 -> '79.2M') so it fits in
    a fixed-width column."""
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)

def _fmt_eval(info: Optional[dict], white_to_move: bool) -> str:
    """UCI score is from the mover's perspective; flip sign to get White's
    perspective so the number in the log means the same thing every move."""
    if not info or "score" not in info:
        return "-"
    score = info["score"]
    sign = 1 if white_to_move else -1
    if "mate" in score:
        m = score["mate"] * sign
        return f"#{m:+d}"
    cp = score["cp"] * sign
    return f"{cp / 100:+.2f}"


# --------------------------------------------------------------------------- #
# Game session
# --------------------------------------------------------------------------- #

class GameSession(threading.Thread):
    def __init__(self, api: LichessAPI, cfg: BotConfig, game_id: str, my_id: str,
                 on_finish):
        super().__init__(daemon=True)
        self.api = api
        self.cfg = cfg
        self.game_id = game_id
        self.my_id = my_id
        self.on_finish = on_finish
        self.engine = UciEngine(cfg.engine_path, cfg.engine_args)
        self.my_color: Optional[str] = None
        self.glog = self._make_game_logger(game_id)

    @staticmethod
    def _make_game_logger(game_id: str) -> logging.Logger:
        os.makedirs(LOG_DIR, exist_ok=True)
        glog = logging.getLogger(f"game.{game_id}")
        glog.setLevel(logging.INFO)
        glog.propagate = True  # also shows up on the console via the root handler
        if not glog.handlers:
            fh = logging.FileHandler(os.path.join(LOG_DIR, f"{game_id}.log"))
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            glog.addHandler(fh)
        return glog

    def run(self):
        try:
            self.engine.new_game()
            for line in self.api.stream_lines(f"/api/bot/game/stream/{self.game_id}"):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    etype = event.get("type")
                    if etype == "gameFull":
                        self.my_color = "white" if event["white"].get("id") == self.my_id else "black"
                        self._log_header(event)
                        self._maybe_move(event["state"])
                    elif etype == "gameState":
                        status = event.get("status")
                        if status not in ("started", "created"):
                            self._log_result(status, event.get("winner"))
                            break
                        self._maybe_move(event)
                except Exception:
                    # A single bad event/move (e.g. a network blip that outlasted
                    # _request's retries) shouldn't forfeit the game — log it and
                    # keep listening. The next gameState event will give us
                    # another chance to move (or the game will end naturally).
                    LOG.exception("Error handling an event in game %s, continuing",
                                  self.game_id)
        except Exception:
            LOG.exception("Game %s crashed", self.game_id)
        finally:
            self.engine.quit()
            self.on_finish(self.game_id)

    def _log_header(self, game_full: dict):
        white = game_full.get("white", {}) or {}
        black = game_full.get("black", {}) or {}
        clock = game_full.get("clock") or {}
        rated = "rated" if game_full.get("rated") else "casual"
        variant = game_full.get("variant", {}).get("name", "Standard")
        white_name = f"{white.get('name', '?')} ({white.get('rating', '?')})"
        black_name = f"{black.get('name', '?')} ({black.get('rating', '?')})"
        clock_str = (f"{clock['initial'] // 60}+{clock['increment']}"
                     if clock else "correspondence")
        self.glog.info("=" * 78)
        self.glog.info("[%s] %s vs %s  |  %s %s  |  %s  |  playing %s",
                        self.game_id, white_name, black_name, rated, variant,
                        clock_str, self.my_color)
        self.glog.info("=" * 78)

    def _log_result(self, status: str, winner: Optional[str]):
        if winner:
            result = f"{winner} wins ({status})"
        else:
            result = status
        self.glog.info("[%s] Game over: %s", self.game_id, result)

    def _maybe_move(self, state: dict):
        moves = state.get("moves", "").split() if state.get("moves") else []
        turn_is_white = (len(moves) % 2 == 0)
        my_turn = (turn_is_white and self.my_color == "white") or \
                  (not turn_is_white and self.my_color == "black")
        if not my_turn:
            return
        move, info, think_secs = self.engine.best_move(
            moves,
            state.get("wtime", 60000), state.get("btime", 60000),
            state.get("winc", 0), state.get("binc", 0),
        )
        move_no = len(moves) // 2 + 1
        side = "White" if turn_is_white else "Black"
        self._log_move(move_no, side, move, info, think_secs)
        if move and move != "(none)":
            resp = self.api.make_move(self.game_id, move)
            if not resp.ok:
                LOG.warning("Move %s rejected for game %s: %s",
                            move, self.game_id, resp.text)

    def _log_move(self, move_no: int, side: str, move: Optional[str],
                   info: Optional[dict], think_secs: float):
        if not move:
            self.glog.warning("[%s] #%-3d %-5s  engine returned no move (%.1fs)",
                               self.game_id, move_no, side, think_secs)
            return
        depth = info.get("depth") if info else None
        seldepth = info.get("seldepth") if info else None
        depth_str = f"{depth}/{seldepth}" if depth is not None and seldepth else \
                     (str(depth) if depth is not None else "-")
        nodes_str = _fmt_count(info.get("nodes") if info else None)
        nps_str = _fmt_count(info.get("nps") if info else None)
        eval_str = _fmt_eval(info, white_to_move=(side == "White"))
        # Compact, fixed-width line (fits an 80-col terminal); no timestamp,
        # log level, or PV — those add width without helping at a glance.
        self.glog.info(
            "[%s] #%-3d %-5s %-6s d%-5s n%-6s nps%-6s eval%-6s %5.1fs",
            self.game_id, move_no, side, move, depth_str, nodes_str, nps_str,
            eval_str, think_secs,
        )


# --------------------------------------------------------------------------- #
# Main bot orchestrator
# --------------------------------------------------------------------------- #

class LichessBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.limiter = RateLimiter(cfg.base_backoff_secs, cfg.max_backoff_secs)
        self.api = LichessAPI(cfg.token, self.limiter)
        self.cooldowns = CooldownStore(cfg.cooldown_file)
        self.games_lock = threading.Lock()
        self.active_games: dict[str, GameSession] = {}
        self.pending_challenges = 0
        self.pending_challenge_targets: dict[str, str] = {}  # challenge_id -> username, autoqueue only
        self.pending_lock = threading.Lock()
        self.my_id = ""
        self.my_perfs: dict = {}           # speed -> {"rating": int, ...}, from /api/account
        self._last_challenge_ts = 0.0
        self.shutting_down = threading.Event()

    # -- helpers --------------------------------------------------------
    def _game_count(self) -> int:
        with self.games_lock:
            return len(self.active_games)

    def _my_rating(self, speed: str) -> Optional[int]:
        return (self.my_perfs.get(speed) or {}).get("rating")

    def request_shutdown(self):
        """Called from the Ctrl+C / SIGTERM handler. First call drains: no new
        games are accepted or queued, and we exit as soon as the current ones
        finish. A second call forces an immediate exit."""
        if self.shutting_down.is_set():
            LOG.warning("Second interrupt — forcing immediate exit (active games left mid-move).")
            os._exit(1)
        self.shutting_down.set()
        n = self._game_count()
        if n == 0:
            LOG.info("Shutdown requested, no active games — exiting now.")
            os._exit(0)
        LOG.info("Shutdown requested: finishing %d active game(s), no new games will be "
                  "accepted or queued. Press Ctrl+C again to quit immediately instead.", n)

    def _on_game_finish(self, game_id: str):
        with self.games_lock:
            self.active_games.pop(game_id, None)
        remaining = self._game_count()
        LOG.info("Game %s finished. Active games: %d", game_id, remaining)
        if self.shutting_down.is_set() and remaining == 0:
            LOG.info("All games finished — exiting.")
            os._exit(0)

    def _accept_rules_pass(self, challenge: dict) -> bool:
        rules = self.cfg.accept
        variant = challenge.get("variant", {}).get("key", "standard")
        speed = challenge.get("speed", "")
        rated = challenge.get("rated", False)
        clock = challenge.get("timeControl", {})
        challenger = challenge.get("challenger", {}) or {}

        if variant not in rules.variants:
            return False
        if speed not in rules.speeds:
            return False
        if rules.rated is not None and rated != rules.rated:
            return False
        if rules.only_bots and challenger.get("title") != "BOT":
            return False
        if not rules.rating_range.allows(challenger.get("rating"), self._my_rating(speed)):
            return False
        limit = clock.get("limit")
        increment = clock.get("increment", 0)
        if limit is not None:
            if not (rules.min_initial_secs <= limit <= rules.max_initial_secs):
                return False
            if increment is not None and increment > rules.max_increment_secs:
                return False
        return True

    # -- event stream handling -------------------------------------------
    def _handle_event(self, event: dict):
        etype = event.get("type")
        if etype == "challenge":
            challenge = event["challenge"]
            challenger_id = (challenge.get("challenger") or {}).get("id", "")
            if challenger_id and challenger_id.lower() == self.my_id.lower():
                # We are the challenger — this is one of our own outgoing
                # challenges (e.g. one autoqueue just sent) showing up on our
                # own event stream, not a challenge *to* us. (Lichess's
                # "direction" field, which would normally flag this, isn't
                # actually included on this event type — only on
                # challengeDeclined/challengeCanceled — so we work it out
                # from the challenger id instead.)
                return
            self._handle_incoming_challenge(challenge)
        elif etype == "gameStart":
            game = event.get("game", {})
            self._start_game(game.get("gameId"))
            # If this game came from one of our own outgoing autoqueue
            # challenges (rather than an incoming one we accepted), that
            # challenge is now resolved — free up its pending-challenge slot.
            # Without this, every accepted autoqueue challenge permanently
            # eats a slot: pending_challenges only ever got decremented on
            # challengeCanceled/challengeDeclined, never on acceptance, so
            # it would ratchet up to max_pending_challenges and autoqueue
            # would stop firing for good, even once games finished and
            # active_games dropped back down.
            opponent = (game.get("opponent") or {}).get("id", "")
            if opponent:
                with self.pending_lock:
                    matched_cid = next(
                        (cid for cid, username in self.pending_challenge_targets.items()
                         if username.lower() == opponent.lower()),
                        None,
                    )
                    if matched_cid:
                        self.pending_challenge_targets.pop(matched_cid, None)
                        self.pending_challenges = max(0, self.pending_challenges - 1)
        elif etype == "challengeCanceled" or etype == "challengeDeclined":
            cid = event.get("challenge", {}).get("id")
            reason = (event.get("challenge", {}).get("declineReason")
                      or event.get("challenge", {}).get("declineReasonKey"))
            with self.pending_lock:
                self.pending_challenges = max(0, self.pending_challenges - 1)
                username = self.pending_challenge_targets.pop(cid, None) if cid else None
            if etype == "challengeDeclined" and username:
                # A bot declining our outgoing challenge doesn't come back as
                # an HTTP error — it's this async event — so without this the
                # cooldown list never gets updated and we'd re-challenge the
                # same bot again on the very next autoqueue pass.
                until = datetime.now(timezone.utc).timestamp() + 3600
                until_iso = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()
                self.cooldowns.set_cooldown(username, until_iso)
                LOG.info("%s declined our challenge (%s) — cooling down 1h",
                         username, reason or "no reason given")

    def _handle_incoming_challenge(self, challenge: dict):
        cid = challenge["id"]
        challenger = challenge.get("challenger", {}).get("name", "?")
        if self.shutting_down.is_set():
            LOG.info("Declining %s from %s: shutting down", cid, challenger)
            self.api.decline_challenge(cid, "later")
            return
        if self._game_count() >= self.cfg.max_concurrent_games:
            LOG.info("Declining %s from %s: at capacity", cid, challenger)
            self.api.decline_challenge(cid, "later")
            return
        if self._accept_rules_pass(challenge):
            LOG.info("Accepting challenge %s from %s", cid, challenger)
            self.api.accept_challenge(cid)
        else:
            LOG.info("Declining challenge %s from %s: rules mismatch", cid, challenger)
            self.api.decline_challenge(cid, self.cfg.accept.decline_reason)

    def _start_game(self, game_id: str):
        with self.games_lock:
            if game_id in self.active_games:
                return
            session = GameSession(self.api, self.cfg, game_id, self.my_id,
                                   self._on_game_finish)
            self.active_games[game_id] = session
        session.start()
        LOG.info("Started game %s. Active games: %d", game_id, self._game_count())

    # -- autoqueue ---------------------------------------------------------
    def _autoqueue_loop(self):
        aq = self.cfg.autoqueue
        if not aq.enabled:
            return
        while not self.shutting_down.is_set():
            time.sleep(aq.poll_interval_secs)
            try:
                self._autoqueue_pass()
            except Exception:
                LOG.exception("Autoqueue pass failed")

    def _autoqueue_pass(self):
        aq = self.cfg.autoqueue
        if self.shutting_down.is_set():
            return
        n = self._game_count()
        if n >= self.cfg.max_concurrent_games:
            LOG.debug("Autoqueue: skipping, at capacity (%d/%d active games)",
                       n, self.cfg.max_concurrent_games)
            return
        with self.pending_lock:
            pending = self.pending_challenges
            if pending >= aq.max_pending_challenges:
                LOG.debug("Autoqueue: skipping, pending challenges at cap (%d/%d) — "
                           "targets: %s", pending, aq.max_pending_challenges,
                           list(self.pending_challenge_targets.values()))
                return
        now = time.time()
        wait_left = aq.min_seconds_between_challenges - (now - self._last_challenge_ts)
        if wait_left > 0:
            LOG.debug("Autoqueue: skipping, %.1fs left before next challenge is allowed",
                       wait_left)
            return

        # Group configured time controls by speed bucket so we only look up
        # each perf/my-rating once even if several time controls share a
        # bucket (e.g. "3+0" and "3+1" are both blitz).
        tcs_by_speed: dict[str, list[dict]] = {}
        for tc in aq.time_controls:
            speed = _speed_bucket(tc["limit"], tc["increment"])
            tcs_by_speed.setdefault(speed, []).append(tc)

        bots = self.api.online_bots(nb=aq.candidates_per_poll)
        with self.pending_lock:
            already_pending = {u.lower() for u in self.pending_challenge_targets.values()}

        # (username, time_control) pairs — a bot that only fits one of
        # several configured time controls still gets one shot at that one.
        eligible: list[tuple[str, dict]] = []
        skipped_pending = skipped_cooldown = skipped_rating_or_games = 0
        for bot in bots:
            username = bot.get("username") or bot.get("id")
            if not username or username.lower() == self.my_id.lower():
                continue
            if username.lower() in already_pending:
                # We already have an outstanding challenge to this bot that
                # hasn't been accepted/declined/canceled yet — don't pile on
                # a second one while we wait to hear back.
                skipped_pending += 1
                continue
            if self.cooldowns.is_cooling_down(username):
                skipped_cooldown += 1
                continue
            perfs = bot.get("perfs", {})
            matching_tcs = []
            for speed, tcs in tcs_by_speed.items():
                perf = perfs.get(speed) or {}
                if perf.get("games", 0) < aq.min_games:
                    # Too few games in this speed/variant for its rating to
                    # mean anything — often a bot that specializes elsewhere
                    # and is just sitting at the untested default here.
                    continue
                if not aq.rating_range.allows(perf.get("rating"), self._my_rating(speed)):
                    continue
                matching_tcs.extend(tcs)
            if matching_tcs:
                # One entry per bot (not per matching time control) so a bot
                # that happens to qualify at three speeds isn't three times
                # as likely to get picked as one that qualifies at one.
                eligible.append((username, random.choice(matching_tcs)))
            else:
                skipped_rating_or_games += 1

        if not eligible:
            LOG.debug("Autoqueue: 0/%d online bots eligible this pass "
                       "(%d already pending, %d cooling down, %d no rating/game-count match)",
                       len(bots), skipped_pending, skipped_cooldown, skipped_rating_or_games)
            return
        # online_bots() comes back in a fairly fixed order (recently-online
        # bots first), so always taking the first match tends to hammer
        # whichever popular bot happens to sit at the top. Pick randomly
        # among everyone who qualifies instead.
        username, time_control = random.choice(eligible)
        self._send_autoqueue_challenge(username, time_control)

    def _send_autoqueue_challenge(self, username: str, time_control: dict):
        aq = self.cfg.autoqueue
        LOG.info("Auto-queueing challenge to %s (%s)", username, _format_time_control(time_control))
        resp = self.api.create_challenge(
            username, rated=aq.rated,
            clock_limit=time_control["limit"],
            clock_increment=time_control["increment"],
            variant=aq.variant,
        )
        self._last_challenge_ts = time.time()
        if resp.status_code == 429:
            return  # RateLimiter already recorded the backoff
        if resp.ok:
            challenge_id = None
            try:
                challenge_id = resp.json().get("challenge", {}).get("id")
            except (ValueError, AttributeError):
                pass
            with self.pending_lock:
                self.pending_challenges += 1
                if challenge_id:
                    self.pending_challenge_targets[challenge_id] = username
        elif resp.status_code == 400:
            recorded = self.cooldowns.record_from_error_text(username, resp.text)
            if not recorded:
                # No explicit timestamp given; still avoid hammering this bot —
                # park it for an hour as a generic backoff.
                until = datetime.now(timezone.utc).timestamp() + 3600
                until_iso = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()
                self.cooldowns.set_cooldown(username, until_iso)
            LOG.info("Challenge to %s rejected: %s", username, resp.text.strip())
        else:
            LOG.warning("Unexpected status %s challenging %s: %s",
                        resp.status_code, username, resp.text)

    # -- main loop -----------------------------------------------------------
    def _login_with_retry(self) -> dict:
        """Block until we can reach Lichess, instead of crashing the whole
        process if the network/DNS isn't up yet at startup. Backs off up to
        60s between attempts."""
        backoff = 5.0
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.api.my_profile()
            except requests.exceptions.RequestException as e:
                LOG.error("Can't reach Lichess yet (attempt %d): %s "
                          "— check this machine's network/DNS. Retrying in %.0fs",
                          attempt, e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 60.0)

    def run(self):
        LOG.info("lichess_bot script version: %s", SCRIPT_VERSION)
        profile = self._login_with_retry()
        self.my_id = profile.get("id") or profile.get("username", "")
        self.my_perfs = profile.get("perfs", {})
        LOG.info("Logged in as %s. Max concurrent games: %d",
                  self.my_id, self.cfg.max_concurrent_games)

        signal.signal(signal.SIGINT, lambda signum, frame: self.request_shutdown())
        signal.signal(signal.SIGTERM, lambda signum, frame: self.request_shutdown())
        LOG.info("Press Ctrl+C to stop: no new games will start, and the process "
                 "exits once current games finish. Press it twice to quit immediately.")

        threading.Thread(target=self._autoqueue_loop, daemon=True).start()

        for line in self.api.stream_lines("/api/stream/event"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._handle_event(event)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run a Rust UCI engine as a Lichess bot")
    parser.add_argument("--config", help="Path to JSON config file", default=None)
    parser.add_argument("--token", help="Lichess API token (overrides config/env)")
    parser.add_argument("--engine", help="Path to the UCI binary (overrides config)")
    parser.add_argument("--max-games", type=int, help="Max concurrent games (overrides config)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleFormatter())
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        handlers=[console_handler],
    )

    cfg = BotConfig.load(args.config)
    cfg.token = args.token or cfg.token or os.environ.get("LICHESS_TOKEN", "")
    cfg.engine_path = args.engine or cfg.engine_path
    if args.max_games:
        cfg.max_concurrent_games = args.max_games

    if not cfg.token:
        parser.error("No Lichess token given (--token, config file, or LICHESS_TOKEN env var)")
    if not cfg.engine_path:
        parser.error("No engine path given (--engine or config file)")

    bot = LichessBot(cfg)
    bot.run()


if __name__ == "__main__":
    main()
