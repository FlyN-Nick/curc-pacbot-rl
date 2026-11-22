"""
Compares search depths (search.LookaheadPolicy) for a single-model checkpoint, over the
training speeds. Games are uncapped: they run until the first lost life, or the board is
cleared. Run from src/:

  python search_sweep.py CHECKPOINT OUT.csv --depths 0 2 3 4 --games 4 [--speeds 4 6 8 ...]

Each (depth, speed, game) is one task on a pool of --workers CPU processes. The CSV has one row
per game, plus the time each move's search took.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

import torch

SPEEDS = list(range(4, 15))


def _game(job):
    path, depth, speed = job
    torch.set_num_threads(1)
    import eval as ev
    from search import LookaheadPolicy, play_game
    from utils import make_gym, single_model, speed_configuration

    q_net, _, _, config, env_version = ev.load_checkpoint(path, torch.device("cpu"))
    if not single_model():
        raise SystemExit("search needs a single-model checkpoint")
    policy = LookaheadPolicy(q_net, depth, config["reward_scale"], config["discount_factor"])
    env = make_gym(speed_configuration(speed), env_version)
    start = time.perf_counter()
    result = play_game(env, policy)
    result.update(depth=depth, speed=speed,
                  ms_per_move=1000 * (time.perf_counter() - start) / max(1, result["steps"]),
                  states_per_move=policy.states_searched / max(1, result["steps"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 2, 3])
    parser.add_argument("--speeds", type=int, nargs="+", default=SPEEDS)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    # Deepest searches first, so the slowest games don't start last
    jobs = [(args.checkpoint, d, s) for d in sorted(args.depths, reverse=True)
            for s in args.speeds for _ in range(args.games)]
    fields = ["depth", "speed", "score", "steps", "cleared", "pellets_left", "pellets_eaten",
              "ms_per_move", "states_per_move"]
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in pool.map(_game, jobs):
            writer.writerow({k: result[k] for k in fields})
            f.flush()

    import collections
    by_depth = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(args.out) as f:
        for row in csv.DictReader(f):
            by_depth[int(row["depth"])][int(row["speed"])].append(row)
    print(f"{'depth':>5} {'mean':>7} {'speed 8':>8} {'pellets left':>12} {'cleared':>8} {'ms/move':>8}")
    for depth in sorted(by_depth):
        speeds = by_depth[depth]
        mean = sum(sum(float(r["score"]) for r in rows) / len(rows) for rows in speeds.values()) / len(speeds)
        rows = [r for rs in speeds.values() for r in rs]
        s8 = speeds.get(8)
        print(f"{depth:>5} {mean:>7.0f} "
              f"{(sum(float(r['score']) for r in s8) / len(s8)) if s8 else float('nan'):>8.0f} "
              f"{sum(float(r['pellets_left']) for r in rows) / len(rows):>12.1f} "
              f"{sum(r['cleared'] == 'True' for r in rows) / len(rows):>8.2f} "
              f"{sum(float(r['ms_per_move']) for r in rows) / len(rows):>8.1f}")


if __name__ == "__main__":
    main()
