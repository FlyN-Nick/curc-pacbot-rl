"""
How many Gumbel search simulations fit in a move on this computer, as at competition: one
game, so every simulation is its own network call on one state, one after another.

Plays real game states (reached by random legal moves at random speeds), searches each one
with each simulation count and device, and prints the time per move (median, 95th percentile,
worst) next to the time the game gives a move: 24 game ticks per second, so 4-14 ticks per move
is ~167-583 ms (the robot still has to move, so the search should use well under that).

The network's weights don't change its speed, so by default each model in --models is freshly
initialized (PVNetV2 and PVNetV2Wide for AlphaZero, QNetV2Wide for the wide DQN, which plays
with search through QNetworkEvaluator); pass --checkpoint to time a trained one instead.

Run from src/ (after building pacbot_rs with maturin):
  python benchmark_search.py [--devices cpu,mps] [--models PVNetV2,PVNetV2Wide,QNetV2Wide]
                             [--sims 8,16,32,64,128] [--states 30] [--checkpoint CKPT]
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import numpy as np
import torch

import models
import utils
from alphazero import NetworkEvaluator, QNetworkEvaluator
from pacbot_rs import GumbelConfig, gumbel_search


def game_states(n: int, seed: int) -> list:
    rng = random.Random(seed)
    states = []
    while len(states) < n:
        env = utils.make_gym(utils.speed_configuration(rng.randint(4, 14)), utils.ENV_VERSION)
        env.seed = rng.randrange(2**63)
        for _ in range(rng.randint(0, 300)):
            mask = env.action_mask()
            _, done = env.step(rng.choice([a for a in range(utils.NUM_ACTIONS) if mask[a]]))
            if done:
                break
        if not env.is_done():
            states.append(env)
    return states


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    available = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (
        ["cuda"] if torch.cuda.is_available() else [])
    parser.add_argument("--devices", default=",".join(available))
    parser.add_argument("--sims", default="8,16,32,64,128")
    parser.add_argument("--states", type=int, default=30)
    parser.add_argument("--models", default="PVNetV2,PVNetV2Wide,QNetV2Wide",
                        help="model classes to time, freshly initialized (ignored with --checkpoint)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cpu-threads", type=int, default=None, help="torch CPU threads (default: torch's)")
    args = parser.parse_args()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)

    utils.set_single_model(True)
    obs_config = utils.replay_buffer_obs_config()
    states = game_states(args.states, seed=0)
    print(f"{len(states)} game states; torch {torch.__version__}, {torch.get_num_threads()} CPU threads")
    print(f"time per move at 24 ticks/s: speed 4 = 167 ms, 8 = 333 ms, 14 = 583 ms\n")
    print(f"{'model':12s} {'device':6s} {'params':>7s} {'sims':>5s} {'median':>8s} {'p95':>8s} {'worst':>8s}"
          f"   per network call")

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        runs = [(ck["config"]["model"], ck["state_dict"])]
    else:
        runs = [(model_name, None) for model_name in args.models.split(",")]
    for (model_name, state_dict), name in ((r, d) for r in runs for d in args.devices.split(",")):
        device = torch.device(name)
        net = getattr(models, model_name)(utils.OBS_SHAPE, utils.NUM_ACTIONS)
        if state_dict is not None:
            net.load_state_dict(state_dict)
        params = sum(p.numel() for p in net.parameters())
        net = net.to(device).eval()
        if device.type != "mps":
            net = net.to(memory_format=torch.channels_last)
        if model_name.startswith("Q"):  # a DQN, searching with its Q values (temperature 0.3 as in eval)
            evaluator = QNetworkEvaluator(net, device, temperature=0.3)
        else:
            evaluator = NetworkEvaluator(net, device, max_batch=1)  # a CUDA graph on CUDA; plain calls elsewhere

        # Warm up (first calls compile kernels / pick algorithms)
        for env in states[:3]:
            gumbel_search([env], evaluator, GumbelConfig(num_simulations=8, gumbel_scale=0.0), obs_config, 0)

        # One network call alone
        obs = np.stack([states[0].obs_numpy(obs_config)])
        legal = np.ones((1, utils.NUM_ACTIONS), bool)
        call_times = []
        for _ in range(50):
            t = time.perf_counter()
            evaluator(obs, legal)
            sync(device)
            call_times.append((time.perf_counter() - t) * 1000)
        per_call = statistics.median(call_times)

        for sims in (int(s) for s in args.sims.split(",")):
            config = GumbelConfig(num_simulations=sims, gumbel_scale=0.0)
            times = []
            for i, env in enumerate(states):
                t = time.perf_counter()
                gumbel_search([env], evaluator, config, obs_config, i)
                times.append((time.perf_counter() - t) * 1000)
            times.sort()
            p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
            print(f"{model_name:12s} {name:6s} {params / 1e6:6.2f}M {sims:5d} {statistics.median(times):7.1f}ms {p95:7.1f}ms {times[-1]:7.1f}ms"
                  f"   {per_call:.2f} ms", flush=True)
        print()


if __name__ == "__main__":
    main()
