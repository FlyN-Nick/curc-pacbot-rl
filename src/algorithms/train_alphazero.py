"""
AlphaZero-style training with Gumbel search (see alphazero.py and pacbot_rs/src/gumbel.rs).

num_games games are played at once. Each move, every game is searched with num_simulations
simulations through the real game (batched: one network call per simulation wave), the action
the search chooses is played, and the move becomes a training example: the observation, the
search's improved policy (the policy target) and, td_steps moves later, the value target. Each
move of all the games is followed by updates_per_step updates of the network on batches from
the newest buffer_capacity examples. Games start like DQN training's: at a random speed
(4-14 ticks per move), from the normal start or (random_start_proportion of them) a random one,
and end at the first lost life.

Every eval_every moves the network plays fixed-seed games at each speed alone (greedy policy),
and every search_eval_every moves with search too; the best greedy mean is kept as
model-best.ckpt.pt (and with search, model-best-search.ckpt.pt).

SIGTERM or SIGINT saves model-latest.ckpt.pt and exits (128 + signal); --resume continues
from a checkpoint (the training examples are not saved; the store refills).

Run from src/ (single-model observations):
  python -m algorithms.train_alphazero [--no-wandb] [--checkpoint-dir DIR] [--resume CKPT] [--HYPERPARAM VALUE ...]
"""

from __future__ import annotations

import math
import os
import random
import signal
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm

import models
import utils
from alphazero import NetworkEvaluator, Targets, TrainingStore, evaluate_games, losses
from pacbot_rs import GumbelConfig, gumbel_search
from utils import ENV_VERSION, NUM_ACTIONS, OBS_SHAPE, TRAINING_SPEEDS, make_gym, random_start_configuration, reset_env

HYPERPARAMS = {
    "model": "PVNetV2",
    "num_games": 256,
    "num_simulations": 32,
    "num_moves": 20_000_000,  # moves played in all (num_moves / num_games steps)
    "td_steps": 10,
    "discount": 0.99,
    "reward_scale": 0.02,
    "buffer_capacity": 500_000,
    "min_buffer": 20_000,
    "batch_size": 512,
    "updates_per_step": 2,
    "learning_rate": 3e-4,
    "lr_final_fraction": 0.1,
    "weight_decay": 1e-4,
    "value_loss_weight": 0.25,
    "grad_clip_norm": 10.0,
    "random_start_proportion": 0.25,
    "c_visit": 50.0,
    "c_scale": 0.1,
    "search_threads": 1,  # 1 was fastest (see GumbelConfig.threads)
    "eval_every": 200_000,  # moves
    "eval_seeds": 2,  # games per speed
    "search_eval_every": 1_000_000,
    "search_eval_seeds": 1,
    "checkpoint_every": 500_000,
}

_exit_signal = 0


def _request_exit(signum, _frame) -> None:
    global _exit_signal
    _exit_signal = signum


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/alphazero")
    parser.add_argument("--resume", metavar="CHECKPOINT", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    for name, default in HYPERPARAMS.items():
        parser.add_argument(f"--{name}", type=type(default), default=default, help="Default: %(default)s")
    args = parser.parse_args()
    if not hasattr(models, args.model):
        parser.error(f"unknown --model {args.model!r}")

    utils.set_single_model(True)
    obs_config = utils.replay_buffer_obs_config()
    device = utils.select_device(args.device)
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    resumed = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    config = {name: getattr(args, name) for name in HYPERPARAMS}
    if resumed is not None:
        # The run's own settings, except those that only say how long or how fast to run
        # (and defaults for settings added since)
        config = {**config, **resumed["config"],
                  **{k: v for k, v in config.items() if k in ("num_moves", "search_threads")}}
    run = wandb.init(
        project="pacbot-ind-study", group="alphazero", config={**config, "device": str(device), "seed": seed},
        mode="disabled" if args.no_wandb else "online",
        id=resumed.get("wandb_id") if resumed else None, resume="allow" if resumed else None,
    )
    cfg = wandb.config
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    net = getattr(models, cfg.model)(OBS_SHAPE, NUM_ACTIONS).to(device)
    if device.type == "cuda":
        # As in DQN training: cuDNN picks the fastest convolutions for each (fixed) batch
        # shape, and channels-last suits its kernels
        torch.backends.cudnn.benchmark = True
        net = net.to(memory_format=torch.channels_last)
    print(f"{cfg.model} has {sum(p.numel() for p in net.parameters())} parameters")
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    step, best, best_search = 0, -math.inf, -math.inf
    if resumed is not None:
        net.load_state_dict(resumed["state_dict"])
        optimizer.load_state_dict(resumed["optimizer"])
        step, best, best_search = resumed["step"], resumed["best"], resumed["best_search"]
        print(f"Resuming from step {step} ({step * cfg.num_games} moves)")

    num_steps = cfg.num_moves // cfg.num_games
    search = GumbelConfig(num_simulations=cfg.num_simulations, discount=cfg.discount,
                          reward_scale=cfg.reward_scale, c_visit=cfg.c_visit, c_scale=cfg.c_scale,
                          gumbel_scale=1.0, threads=cfg.search_threads)
    eval_search = GumbelConfig(num_simulations=cfg.num_simulations, discount=cfg.discount,
                               reward_scale=cfg.reward_scale, c_visit=cfg.c_visit, c_scale=cfg.c_scale,
                               gumbel_scale=0.0, threads=cfg.search_threads)
    store = TrainingStore(cfg.buffer_capacity, device)
    targets = Targets(cfg.num_games, cfg.td_steps, cfg.discount)
    evaluator = NetworkEvaluator(net, device, max_batch=cfg.num_games)  # one CUDA graph

    def reset_config():
        if rng.random() < cfg.random_start_proportion:
            return random_start_configuration()
        return obs_config  # the normal start at a random speed

    games = [make_gym(reset_config(), ENV_VERSION) for _ in range(cfg.num_games)]
    game_moves = [0] * cfg.num_games

    def learning_rate(at_step: int) -> float:
        t = min(1.0, at_step / max(1, num_steps))
        return cfg.learning_rate * (cfg.lr_final_fraction + (1 - cfg.lr_final_fraction) * 0.5 * (1 + math.cos(math.pi * t)))

    def save(name: str) -> None:
        state = {"state_dict": net.state_dict(), "optimizer": optimizer.state_dict(), "config": dict(cfg),
                 "step": step, "moves": step * cfg.num_games, "best": best, "best_search": best_search,
                 "wandb_id": run.id if run else None, "env_version": ENV_VERSION, "single_model": True}
        tmp = checkpoint_dir / f".{name}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, checkpoint_dir / name)

    def evaluate(with_search: bool) -> dict:
        seeds = range(1, (cfg.search_eval_seeds if with_search else cfg.eval_seeds) + 1)
        start = time.perf_counter()
        results = evaluate_games(net, device, ENV_VERSION, obs_config, TRAINING_SPEEDS, seeds,
                                 search=eval_search if with_search else None)
        prefix = "search_eval" if with_search else "eval"
        by_speed = {s: [r["score"] for r in results if r["speed"] == s] for s in TRAINING_SPEEDS}
        metrics = {
            f"{prefix}_mean_score": float(np.mean([r["score"] for r in results])),
            f"{prefix}_cleared": float(np.mean([r["cleared"] for r in results])),
            f"{prefix}_pellets_left": float(np.mean([r["pellets_left"] for r in results])),
            f"{prefix}_seconds": time.perf_counter() - start,
            **{f"{prefix}_speed{s}": float(np.mean(v)) for s, v in by_speed.items()},
        }
        return metrics

    signal.signal(signal.SIGTERM, _request_exit)
    signal.signal(signal.SIGINT, _request_exit)

    episode_scores, episode_moves = [], []
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "grad_norm": []}
    timer = {"search": 0.0, "play": 0.0, "train": 0.0}
    progress = tqdm(range(step, num_steps), initial=step, total=num_steps, smoothing=0.05)
    for step in progress:
        if _exit_signal:
            save("model-latest.ckpt.pt")
            print(f"Stopped by signal {_exit_signal} at step {step}; saved model-latest.ckpt.pt")
            wandb.finish()
            raise SystemExit(128 + _exit_signal)

        # Search and play one move in every game
        t0 = time.perf_counter()
        net.eval()
        actions, policy, value, network_value, _ = gumbel_search(games, evaluator, search, obs_config, rng.getrandbits(63))
        t1 = time.perf_counter()
        examples = []
        for i, env in enumerate(games):
            obs = env.obs_numpy(obs_config)
            legal = np.array(env.action_mask())
            reward, done = env.step(int(actions[i]))
            game_moves[i] += 1
            examples += targets.add(i, obs, legal, policy[i], float(value[i]), reward * cfg.reward_scale)
            if done:
                examples += targets.end(i)
                episode_scores.append(env.score())
                episode_moves.append(game_moves[i])
                game_moves[i] = 0
                reset_env(env, reset_config())
        store.add(examples)
        t2 = time.perf_counter()

        # Train
        if len(store) >= cfg.min_buffer:
            net.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        for _ in range(cfg.updates_per_step if len(store) >= cfg.min_buffer else 0):
            policy_loss, value_loss, entropy = losses(net, store.sample(cfg.batch_size))
            loss = policy_loss + cfg.value_loss_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            for k, v in (("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy), ("grad_norm", grad_norm)):
                stats[k].append(v.detach())
        t3 = time.perf_counter()
        timer["search"] += t1 - t0
        timer["play"] += t2 - t1
        timer["train"] += t3 - t2

        moves = (step + 1) * cfg.num_games
        if (step + 1) % 20 == 0:
            metrics = {"moves": moves, "learning_rate": learning_rate(step), "stored": len(store),
                       "network_value": float(np.mean(network_value)), "search_value": float(np.mean(value)),
                       **{f"seconds_{k}": v for k, v in timer.items()}}
            for k, v in stats.items():
                if v:
                    metrics[k] = torch.stack(v).float().mean().item()
                v.clear()
            timer = dict.fromkeys(timer, 0.0)
            if episode_scores:
                metrics["train_episode_score"] = float(np.mean(episode_scores))
                metrics["train_episode_moves"] = float(np.mean(episode_moves))
                metrics["train_episodes"] = len(episode_scores)
                episode_scores.clear()
                episode_moves.clear()
            if moves % cfg.eval_every < cfg.num_games * 20:
                metrics.update(evaluate(with_search=False))
                if metrics["eval_mean_score"] > best:
                    best = metrics["eval_mean_score"]
                    save("model-best.ckpt.pt")
            if moves % cfg.search_eval_every < cfg.num_games * 20:
                metrics.update(evaluate(with_search=True))
                if metrics["search_eval_mean_score"] > best_search:
                    best_search = metrics["search_eval_mean_score"]
                    save("model-best-search.ckpt.pt")
            if moves % cfg.checkpoint_every < cfg.num_games * 20:
                save("model-latest.ckpt.pt")
                save(f"model-moves{moves:09}.ckpt.pt")
            wandb.log(metrics, step=moves)
            progress.set_postfix(score=metrics.get("train_episode_score"), stored=len(store))

    step = num_steps
    save("model-latest.ckpt.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
