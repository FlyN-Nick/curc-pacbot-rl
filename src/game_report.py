"""
Plays the same seeded evaluation games with each checkpoint and reports how they end.

Every checkpoint plays games with seeds 1..GAMES at each speed, so it gets the same games
(the frightened ghosts' moves are the only randomness; a seed fixes them), and playing a
checkpoint twice gives the same results. Games end at the first lost life, as in training's
evaluation.

For each game it records how it ended (board cleared, life lost, or the step cap) and, for a
lost life, what the situation was just before it:
  catcher         the dangerous ghost nearest Pacman (red, pink, cyan, orange)
  ghosts_near     dangerous ghosts within 5 squares (Manhattan) of Pacman: 2+ means cornered
                  between ghosts rather than caught by one chaser
  fright_ended    steps since the ghosts were last frightened (power pellet wore off)
  since_pellet    steps since Pacman last ate a pellet (wandering if large)
  pellets_left    pellets left; with the quadrant of the board most of them are in
  region          where Pacman died (top/bottom, left/right half of the maze)
  distinct_100    distinct squares Pacman visited in its last 100 steps (low: going in circles)

Run from src/ with PYTHONPATH=.:../model_package:
  python game_report.py GAMES OUT_PREFIX CKPT [CKPT...]   (env GAME_REPORT_WORKERS, default 4)
Writes OUT_PREFIX.games.csv (one row per game) and OUT_PREFIX.summary.txt, and prints the summary.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SPEEDS = range(4, 15)
MAX_STEPS = 100_000
COLORS = ("red", "pink", "cyan", "orange")
SUPER_PELLETS = ((3, 1), (3, 26), (23, 1), (23, 26))
NEAR = 5

# The competition server's wire format (pacbot_rs GameState::to_bytes)
_MODE, _LIVES, _GHOSTS, _PACMAN, _PELLETS = 3, 11, 13, 29, 35


def _loc(b: bytes, i: int) -> tuple[int, int]:
    return b[i] & 0x3F, b[i + 1] & 0x3F


def decode(b: bytes) -> dict:
    ghosts = []
    for g in range(4):
        i = _GHOSTS + 4 * g
        aux, aux2 = b[i + 2], b[i + 3]
        ghosts.append({
            "loc": _loc(b, i),
            "fright": aux & 0x3F,
            "spawning": bool(aux & 0x80),
            "eaten": bool(aux2 & 0x80),
            "trapped": aux2 & 0x3F,
        })
    rows = [int.from_bytes(b[_PELLETS + 4 * r:_PELLETS + 4 * r + 4], "big") for r in range(31)]
    pellets = [(r, c) for r in range(31) for c in range(28) if rows[r] >> c & 1]
    return {"mode": ("paused", "scatter", "chase")[b[_MODE]] if b[_MODE] < 3 else "?",
            "lives": b[_LIVES], "pacman": _loc(b, _PACMAN), "ghosts": ghosts, "pellets": pellets}


def _region(loc: tuple[int, int]) -> str:
    return ("top" if loc[0] < 16 else "bottom") + "-" + ("left" if loc[1] < 14 else "right")


def _dangerous(g: dict) -> bool:
    # In the maze (not waiting in the ghost house), not frightened, not returning eaten
    return g["fright"] == 0 and not g["eaten"] and g["loc"] != (32, 32) and g["trapped"] == 0


def _dist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


_MODELS: dict = {}


def play(job) -> dict:
    path, ticks_per_step, seed = job
    import torch

    import eval as ev
    import utils

    torch.set_num_threads(1)
    if path not in _MODELS:
        _MODELS[path] = ev.load_checkpoint(path, torch.device("cpu"))
    q_net, iter_num, _, _, env_version = _MODELS[path]
    q_net.eval()
    env = utils.make_gym(utils.speed_configuration(ticks_per_step), env_version)
    env.seed = seed
    pellets_start = env.remaining_pellets()
    last_pellet_step = 0
    last_fright_step = None
    recent = deque(maxlen=100)
    before = decode(env.game_state_bytes())
    done = False
    steps = 0
    while not done and steps < MAX_STEPS:
        before = decode(env.game_state_bytes())
        recent.append(before["pacman"])
        if any(g["fright"] > 0 for g in before["ghosts"]):
            last_fright_step = steps
        pellets_before = env.remaining_pellets()
        _, done = utils.step_env_once(env, utils_policy(q_net), torch.device("cpu"))
        steps += 1
        if env.remaining_pellets() < pellets_before:
            last_pellet_step = steps

    cleared = done and env.lives() == 3
    end = "cleared" if cleared else ("died" if done else "step_cap")
    row = {
        "run": Path(path).parent.name, "file": Path(path).name, "iter": iter_num,
        "speed": ticks_per_step, "seed": seed, "end": end, "score": env.score(), "steps": steps,
        "pellets_left": 0 if cleared else env.remaining_pellets(),
        "pellets_eaten_frac": round(1 - (0 if cleared else env.remaining_pellets()) / pellets_start, 3),
    }
    if end != "cleared":
        pac = before["pacman"]
        danger = [(_dist(g["loc"], pac), COLORS[i]) for i, g in enumerate(before["ghosts"]) if _dangerous(g)]
        quadrants = Counter(_region(p) for p in before["pellets"])
        top_quadrant, top_count = quadrants.most_common(1)[0] if quadrants else ("-", 0)
        row.update({
            "mode": before["mode"],
            "catcher": min(danger)[1] if danger and end == "died" else "",
            "catcher_dist": min(danger)[0] if danger else "",
            "ghosts_near": sum(d <= NEAR for d, _ in danger),
            "fright_ended": "" if last_fright_step is None else steps - last_fright_step,
            "since_pellet": steps - last_pellet_step,
            "region": _region(pac),
            "pellets_quadrant": top_quadrant,
            "pellets_quadrant_frac": round(top_count / max(1, len(before["pellets"])), 2),
            "super_left": sum(p in before["pellets"] for p in SUPER_PELLETS),
            "distinct_100": len(set(recent)),
        })
    return row


def utils_policy(q_net):
    from policies import MaxQPolicy
    return MaxQPolicy(q_net)


FIELDS = ["run", "file", "iter", "speed", "seed", "end", "score", "steps", "pellets_left",
          "pellets_eaten_frac", "mode", "catcher", "catcher_dist", "ghosts_near", "fright_ended",
          "since_pellet", "region", "pellets_quadrant", "pellets_quadrant_frac", "super_left",
          "distinct_100"]


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:3.0f}%" if d else "  - "


def summarize(rows: list[dict]) -> str:
    out = []
    by_ckpt = defaultdict(list)
    for r in rows:
        by_ckpt[(r["run"], r["file"])].append(r)
    for (run, file), games in by_ckpt.items():
        out.append(f"\n=== {run}/{file} ({len(games)} games) ===")
        out.append("speed  mean   clear  died  cap | caught by r/p/c/o     | 2+ near  <15 after fright  "
                   "no pellet 50+  circling(<15 sq)  pellets left (med)  <=20 left")
        groups = [(str(k), [g for g in games if g["speed"] == k]) for k in SPEEDS]
        groups += [("4-8", [g for g in games if g["speed"] <= 8]), ("9-11", [g for g in games if 9 <= g["speed"] <= 11]),
                   ("12-14", [g for g in games if g["speed"] >= 12]), ("all", games)]
        for label, gs in groups:
            if not gs:
                continue
            n = len(gs)
            died = [g for g in gs if g["end"] == "died"]
            d = len(died)
            caught = Counter(g["catcher"] for g in died)
            fright = [g for g in died if g["fright_ended"] != "" and g["fright_ended"] < 15]
            out.append(
                f"{label:>5} {statistics.mean(g['score'] for g in gs):6.0f}  "
                f"{_pct(sum(g['end'] == 'cleared' for g in gs), n)} {_pct(d, n)} "
                f"{_pct(sum(g['end'] == 'step_cap' for g in gs), n)} | "
                + "/".join(_pct(caught[c], d).strip() for c in COLORS).ljust(21) + " | "
                f"{_pct(sum(g['ghosts_near'] >= 2 for g in died), d):>7}  "
                f"{_pct(len(fright), d):>16}  "
                f"{_pct(sum(g['since_pellet'] >= 50 for g in died), d):>13}  "
                f"{_pct(sum(g['distinct_100'] < 15 for g in died), d):>16}  "
                f"{statistics.median(g['pellets_left'] for g in died) if died else 0:>18.0f}  "
                f"{_pct(sum(g['pellets_left'] <= 20 for g in died), d):>9}"
            )
        died = [g for g in games if g["end"] == "died"]
        if died:
            out.append("deaths by region: " + ", ".join(f"{k} {_pct(v, len(died)).strip()}" for k, v in
                                                      Counter(g["region"] for g in died).most_common()))
            out.append("deaths by mode: " + ", ".join(f"{k} {_pct(v, len(died)).strip()}" for k, v in
                                                    Counter(g["mode"] for g in died).most_common()))
            left = [g for g in died if g["pellets_left"] <= 40]
            if left:
                out.append(f"deaths with <=40 pellets left: {len(left)}; their pellets mostly in one quadrant "
                           f"(>=60%): {_pct(sum(g['pellets_quadrant_frac'] >= 0.6 for g in left), len(left)).strip()}, "
                           f"power pellets still up: {_pct(sum(g['super_left'] > 0 for g in left), len(left)).strip()}")
    return "\n".join(out)


if __name__ == "__main__":
    games, prefix, paths = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    jobs = [(p, k, seed) for p in paths for k in SPEEDS for seed in range(1, games + 1)]
    workers = int(os.environ.get("GAME_REPORT_WORKERS", "4"))
    rows = []
    with ProcessPoolExecutor(workers) as pool, open(f"{prefix}.games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for row in pool.map(play, jobs, chunksize=4):
            rows.append(row)
            w.writerow(row)
            f.flush()
    text = summarize(rows)
    Path(f"{prefix}.summary.txt").write_text(text + "\n")
    print(text)
