"""
Fixed-seed evaluation of LightZero checkpoints (algorithms/train_lightzero.py): seeds 1..N at
each speed 4-14, all games at once, played with the algorithm's own search, as
eval_az_seeds.py plays our AlphaZero and DQN checkpoints, and written in the same CSV format.

Usage: eval_lightzero_seeds.py SEEDS OUT.csv ALGORITHM:CKPT:SIMS ...
    (e.g. muzero:checkpoints/mz1/run-.../ckpt/ckpt_best.pth.tar:128)
The network settings must match the run's (--num_res_blocks, --num_channels; defaults 2 and 64).
With PACBOT_FULL_GAME=1, the games are full games, as in eval_az_seeds.py.
"""

import argparse
import csv
import os
import tempfile
import time

import numpy as np
import torch

import utils
from lightzero_pacbot.configs import make_config
from lightzero_pacbot.pacbot_env import PacbotLightZeroEnv


def make_policy(algorithm: str, sims: int, num_res_blocks: int, num_channels: int):
    """
    A LightZero policy with freshly initialized weights. Collect mode is set up too: Gumbel and
    Stochastic MuZero's eval forward searches with the collect model.
    """
    from ding.config import compile_config
    from ding.policy import create_policy
    cuda = torch.cuda.is_available()
    main, create = make_config(algorithm, "eval", num_simulations=sims, cuda=cuda,
                               num_res_blocks=num_res_blocks, num_channels=num_channels)
    main.policy.device = "cuda" if cuda else "cpu"
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # compile_config writes the config out
        try:
            cfg = compile_config(main, seed=0, env=None, auto=True, create_cfg=create, save_cfg=False)
        finally:
            os.chdir(cwd)
    return create_policy(cfg.policy, enable_field=["learn", "collect", "eval"])


def load_policy(algorithm: str, path: str, sims: int, num_res_blocks: int, num_channels: int):
    policy = make_policy(algorithm, sims, num_res_blocks, num_channels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.learn_mode.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    return policy.eval_mode


def play(policy, speeds, seeds, max_steps: int):
    games = []
    for speed in speeds:
        for seed in seeds:
            env = PacbotLightZeroEnv({"eval_games": [(speed, seed)], "max_episode_steps": max_steps,
                                      "full_game": utils.full_game(), "max_speed": utils.max_speed()})
            games.append({"speed": speed, "seed": seed, "env": env, "obs": env.reset(), "done": False, "info": None})
    while True:
        live = [g for g in games if not g["done"]]
        if not live:
            break
        data = torch.from_numpy(np.stack([g["obs"]["observation"] for g in live]))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        masks = [g["obs"]["action_mask"] for g in live]
        out = policy.forward(data.to(device), action_mask=masks, to_play=[-1] * len(live),
                             ready_env_id=np.arange(len(live)))
        for i, g in enumerate(live):
            step = g["env"].step(out[i]["action"])
            g["obs"] = step.obs
            if step.done:
                g["done"], g["info"] = True, step.info
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seeds", help="N (seeds 1..N) or A-B")
    parser.add_argument("out")
    parser.add_argument("specs", nargs="+", help="ALGORITHM:CKPT:SIMS")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_channels", type=int, default=64)
    args = parser.parse_args()
    lo, _, hi = args.seeds.partition("-")
    seeds = range(int(lo), int(hi) + 1) if hi else range(1, int(lo) + 1)
    max_steps = 40_000 if utils.full_game() else 10_000
    speeds = utils.training_speeds()  # 4-14, or up to PACBOT_MAX_SPEED, as eval_az_seeds.py
    with open(args.out, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["checkpoint", "sims", "temperature", "speed", "seed", "score", "steps", "cleared",
                      "pellets_left", "boards", "lives"])
        for spec in args.specs:
            algorithm, path, sims = spec.split(":")
            policy = load_policy(algorithm, path, int(sims), args.num_res_blocks, args.num_channels)
            t = time.perf_counter()
            games = play(policy, speeds, seeds, max_steps)
            name = f"{algorithm}:" + path.split("/checkpoints/")[-1]
            for g in games:
                env, info = g["env"]._env, g["info"]
                boards = info["boards"]
                cleared = boards > 0 if utils.full_game() else env.is_done() and env.lives() == 3
                out.writerow([name, sims, "", g["speed"], g["seed"], info["score"], info["steps"], int(cleared),
                              0 if cleared and not utils.full_game() else env.remaining_pellets(), boards, info["lives"]])
            f.flush()
            sc = np.array([g["info"]["score"] for g in games])
            by = lambda lo, hi: np.mean([g["info"]["score"] for g in games if lo <= g["speed"] <= hi])
            cleared = np.mean([g["env"]._env.lives() == 3 and g["env"]._env.is_done() for g in games])
            print(f"{name} sims {sims}: mean {sc.mean():.0f} (se {sc.std() / np.sqrt(len(sc)):.0f}) | 4-8 {by(4, 8):.0f} "
                  f"9-11 {by(9, 11):.0f} 12-14 {by(12, 14):.0f} | cleared {cleared:.0%} | {time.perf_counter() - t:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
