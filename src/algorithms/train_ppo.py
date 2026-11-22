"""
PPO (proximal policy optimization) with a shared policy and value network (PVNetV2), in the same
games, observations and evaluations as AlphaZero training (train_alphazero.py), so the two
compare directly and PPO checkpoints play wherever AlphaZero's do (eval_az_seeds.py, the bot
client, with or without search).

num_games games are played at once. Each rollout, every game plays rollout_steps moves by
sampling from the network's policy over the legal moves; then the network is trained for
`epochs` passes over the rollout, in `minibatches` minibatches, with PPO's clipped policy loss
(clip_range), a value loss (value_loss_weight; optionally clipped, value_clip_range) and an
entropy bonus (entropy_coef). Advantages come from GAE (discount, gae_lambda), on rewards
scaled by reward_scale, so the value is in the units AlphaZero's search uses. Games start like
AlphaZero training's: at a random speed (speed_sampling, up to max_speed), from the normal
start or (random_start_proportion of them) a random one, and end at the first lost life or
cleared board, or with full_game, the last life.

Every eval_every moves the network plays fixed-seed games at each speed alone (greedy policy),
and every search_eval_every moves with search too (search_eval_simulations, AlphaZero's search
constants c_visit and c_scale); the best greedy mean is kept as model-best.ckpt.pt (and with
search, model-best-search.ckpt.pt).

SIGTERM or SIGINT saves model-latest.ckpt.pt and exits (128 + signal); --resume continues
from a checkpoint (the rollout in progress starts over).

Run from src/ (single-model observations):
  python -m algorithms.train_ppo [--no-wandb] [--checkpoint-dir DIR] [--resume CKPT] [--HYPERPARAM VALUE ...]
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
import torch.nn.functional as F
import wandb
from tqdm import tqdm

import models
import utils
from alphazero import evaluate_games
from pacbot_rs import GumbelConfig
from replay_storage import ObsCodec
from speed_sampling import SpeedSampler
from utils import ENV_VERSION, NUM_ACTIONS, OBS_SHAPE, make_gym, reset_env, training_configuration, training_speeds

HYPERPARAMS = {
    "model": "PVNetV2",
    "num_games": 256,
    "rollout_steps": 128,  # moves per game per rollout (num_games * rollout_steps examples)
    "num_moves": 100_000_000,  # moves played in all
    "epochs": 4,
    "minibatches": 8,
    "discount": 0.99,
    "gae_lambda": 0.95,
    "reward_scale": 0.02,
    "clip_range": 0.2,
    "value_clip_range": 0.0,  # 0: the value loss isn't clipped
    "value_loss_weight": 0.5,
    "entropy_coef": 0.01,
    "learning_rate": 3e-4,
    "lr_final_fraction": 0.1,
    "weight_decay": 0.0,
    "grad_clip_norm": 0.5,
    "normalize_advantages": True,
    "target_kl": 0.0,  # 0: no early stop; else stop a rollout's epochs once the approximate KL passes it
    "random_start_proportion": 0.25,
    "full_game": False,  # three lives and board after board, as in competition (utils.set_full_game)
    "speed_sampling": "uniform",  # or "equal_moves" (speed_sampling.py)
    "max_speed": 14,
    "c_visit": 50.0,  # AlphaZero's search constants, for evaluating with search
    "c_scale": 0.1,
    "search_eval_simulations": 32,
    "eval_every": 2_000_000,  # moves
    "eval_seeds": 2,  # games per speed
    "search_eval_every": 10_000_000,
    "search_eval_seeds": 1,
    "checkpoint_every": 5_000_000,
}

_exit_signal = 0


def _request_exit(signum, _frame) -> None:
    global _exit_signal
    _exit_signal = signum


def gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
        discount: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized advantage estimates and value targets for a rollout: rewards, values and dones are
    (steps, games); dones[t] says the game ended with move t (no bootstrapping past it, and the next
    row is a new game); last_value is the value of each game's state after the rollout.
    """
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for t in range(rewards.shape[0] - 1, -1, -1):
        alive = 1.0 - dones[t]
        delta = rewards[t] + discount * next_value * alive - values[t]
        running = delta + discount * gae_lambda * alive * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def ppo_losses(net: torch.nn.Module, obs: torch.Tensor, legal: torch.Tensor, actions: torch.Tensor,
               old_log_prob: torch.Tensor, old_value: torch.Tensor, advantages: torch.Tensor,
               returns: torch.Tensor, clip_range: float, value_clip_range: float) -> dict[str, torch.Tensor]:
    """PPO's clipped policy loss, the value loss, the entropy and the approximate KL, for a minibatch."""
    logits, value = net(obs)
    log_policy = F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=1)
    log_prob = log_policy.gather(1, actions[:, None]).squeeze(1)
    ratio = torch.exp(log_prob - old_log_prob)
    policy_loss = -torch.min(ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages).mean()
    if value_clip_range > 0:
        clipped = old_value + (value - old_value).clamp(-value_clip_range, value_clip_range)
        value_loss = 0.5 * torch.max((value - returns) ** 2, (clipped - returns) ** 2).mean()
    else:
        value_loss = 0.5 * F.mse_loss(value, returns)
    probs = log_policy.exp()
    entropy = -(probs * log_policy.masked_fill(~legal, 0.0)).sum(dim=1).mean()
    with torch.no_grad():
        log_ratio = log_prob - old_log_prob
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_range).float().mean()
    return {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
            "approx_kl": approx_kl, "clip_fraction": clip_fraction}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo")
    parser.add_argument("--resume", metavar="CHECKPOINT", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    for name, default in HYPERPARAMS.items():
        kind = utils.parse_bool if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{name}", type=kind, default=default, help="Default: %(default)s")
    args = parser.parse_args()
    if not hasattr(models, args.model) or not issubclass(getattr(models, args.model), models.PVNetV2):
        parser.error(f"--model must be a policy and value network (PVNetV2 or PVNetV2Wide), got {args.model!r}")
    if args.speed_sampling not in utils.SPEED_SAMPLING_MODES:
        parser.error(f"--speed_sampling must be one of {utils.SPEED_SAMPLING_MODES}")
    if args.max_speed < utils.MIN_SPEED:
        parser.error(f"--max_speed must be at least {utils.MIN_SPEED}")
    if (args.num_games * args.rollout_steps) % args.minibatches:
        parser.error("num_games * rollout_steps must be a multiple of minibatches")

    utils.set_single_model(True)
    obs_config = utils.replay_buffer_obs_config()
    device = utils.select_device(args.device)
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    resumed = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    config = {name: getattr(args, name) for name in HYPERPARAMS}
    if resumed is not None:
        # The run's own settings, except how long to run (and defaults for settings added since)
        config = {**config, **resumed["config"], **{k: v for k, v in config.items() if k == "num_moves"}}
    run = wandb.init(
        project="pacbot-ind-study", group="ppo", config={**config, "device": str(device), "seed": seed},
        mode="disabled" if args.no_wandb else "online",
        id=resumed.get("wandb_id") if resumed else None, resume="allow" if resumed else None,
    )
    cfg = wandb.config
    utils.set_speed_sampling(cfg.speed_sampling)
    utils.set_max_speed(cfg.max_speed)
    utils.set_full_game(cfg.full_game)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    net = getattr(models, cfg.model)(OBS_SHAPE, NUM_ACTIONS).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        net = net.to(memory_format=torch.channels_last)
    print(f"{cfg.model} has {sum(p.numel() for p in net.parameters())} parameters")
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, eps=1e-5)
    rollout_moves = cfg.num_games * cfg.rollout_steps
    num_rollouts = max(1, cfg.num_moves // rollout_moves)
    rollout, best, best_search = 0, -math.inf, -math.inf
    if resumed is not None:
        net.load_state_dict(resumed["state_dict"])
        optimizer.load_state_dict(resumed["optimizer"])
        rollout, best, best_search = resumed["rollout"], resumed["best"], resumed["best_search"]
        print(f"Resuming from rollout {rollout} ({rollout * rollout_moves} moves)")

    eval_search = GumbelConfig(num_simulations=cfg.search_eval_simulations, discount=cfg.discount,
                               reward_scale=cfg.reward_scale, c_visit=cfg.c_visit, c_scale=cfg.c_scale,
                               gumbel_scale=0.0)
    speed_sampler = SpeedSampler(rng=rng)

    def reset_config(speed: int):
        return training_configuration(speed, random_start=rng.random() < cfg.random_start_proportion)

    game_speeds = [speed_sampler.draw() for _ in range(cfg.num_games)]
    games = [make_gym(reset_config(speed), ENV_VERSION) for speed in game_speeds]
    game_moves = [0] * cfg.num_games

    # One rollout, on the device: observations encoded compactly (ObsCodec; plain on MPS)
    T, B = cfg.rollout_steps, cfg.num_games
    codec = ObsCodec(device)
    plain_obs = device.type == "mps"
    obs_stores = ((torch.zeros((T * B, *OBS_SHAPE), dtype=torch.float32, device=device),) if plain_obs
                  else ObsCodec.stores(T * B, device))
    legal_store = torch.zeros((T * B, NUM_ACTIONS), dtype=torch.bool, device=device)
    action_store = torch.zeros(T * B, dtype=torch.long, device=device)
    log_prob_store = torch.zeros(T * B, device=device)
    value_store = torch.zeros((T, B), device=device)
    reward_store = torch.zeros((T, B), device=device)
    done_store = torch.zeros((T, B), device=device)

    def decode(idx: torch.Tensor) -> torch.Tensor:
        if plain_obs:
            return obs_stores[0][idx]
        return codec.decode(*(store[idx] for store in obs_stores))

    def learning_rate(at_rollout: int) -> float:
        t = min(1.0, at_rollout / max(1, num_rollouts))
        return cfg.learning_rate * (cfg.lr_final_fraction + (1 - cfg.lr_final_fraction) * 0.5 * (1 + math.cos(math.pi * t)))

    def save(name: str, completed: int) -> None:
        """Saves the network and optimizer after `completed` rollouts."""
        state = {"state_dict": net.state_dict(), "optimizer": optimizer.state_dict(), "config": dict(cfg),
                 "rollout": completed, "moves": completed * rollout_moves, "best": best, "best_search": best_search,
                 "wandb_id": run.id if run else None, "env_version": ENV_VERSION, "single_model": True,
                 "full_game": bool(cfg.full_game), "algorithm": "ppo"}
        tmp = checkpoint_dir / f".{name}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, checkpoint_dir / name)

    def evaluate(with_search: bool) -> dict:
        seeds = range(1, (cfg.search_eval_seeds if with_search else cfg.eval_seeds) + 1)
        start = time.perf_counter()
        results = evaluate_games(net, device, ENV_VERSION, obs_config, training_speeds(), seeds,
                                 search=eval_search if with_search else None)
        prefix = "search_eval" if with_search else "eval"
        by_speed = {s: [r["score"] for r in results if r["speed"] == s] for s in training_speeds()}
        return {
            f"{prefix}_mean_score": float(np.mean([r["score"] for r in results])),
            f"{prefix}_cleared": float(np.mean([r["cleared"] for r in results])),
            f"{prefix}_seconds": time.perf_counter() - start,
            **{f"{prefix}_speed{s}": float(np.mean(v)) for s, v in by_speed.items()},
        }

    signal.signal(signal.SIGTERM, _request_exit)
    signal.signal(signal.SIGINT, _request_exit)

    episode_scores, episode_moves = [], []
    progress = tqdm(range(rollout, num_rollouts), initial=rollout, total=num_rollouts, smoothing=0.05)
    for rollout in progress:
        if _exit_signal:
            save("model-latest.ckpt.pt", rollout)
            print(f"Stopped by signal {_exit_signal} at rollout {rollout}; saved model-latest.ckpt.pt")
            wandb.finish()
            raise SystemExit(128 + _exit_signal)

        # Play rollout_steps moves in every game, sampling from the policy
        t0 = time.perf_counter()
        net.eval()
        with torch.inference_mode():
            for t in range(T):
                obs = torch.from_numpy(np.stack([env.obs_numpy(obs_config) for env in games])).to(device)
                legal = torch.tensor([env.action_mask() for env in games], device=device)
                logits, value = net(obs)
                log_policy = F.log_softmax(logits.float().masked_fill(~legal, -torch.inf), dim=1)
                actions = torch.multinomial(log_policy.exp(), 1, generator=generator).squeeze(1)
                rows = slice(t * B, (t + 1) * B)
                for store, part in zip(obs_stores, (obs,) if plain_obs else codec.encode(obs)):
                    store[rows] = part
                legal_store[rows] = legal
                action_store[rows] = actions
                log_prob_store[rows] = log_policy.gather(1, actions[:, None]).squeeze(1)
                value_store[t] = value.float()
                rewards, dones = [], []
                for i, (env, action) in enumerate(zip(games, actions.tolist())):
                    reward, done = env.step(action)
                    game_moves[i] += 1
                    rewards.append(reward * cfg.reward_scale)
                    dones.append(float(done))
                    if done:
                        episode_scores.append(env.score())
                        episode_moves.append(game_moves[i])
                        speed_sampler.record(game_speeds[i], game_moves[i])
                        game_speeds[i], game_moves[i] = speed_sampler.draw(), 0
                        reset_env(env, reset_config(game_speeds[i]))
                reward_store[t] = torch.tensor(rewards, device=device)
                done_store[t] = torch.tensor(dones, device=device)
            last_obs = torch.from_numpy(np.stack([env.obs_numpy(obs_config) for env in games])).to(device)
            last_value = net(last_obs)[1].float()
            advantages, returns = gae(reward_store, value_store, done_store, last_value, cfg.discount, cfg.gae_lambda)
        advantages, returns = advantages.reshape(-1), returns.reshape(-1)
        old_values = value_store.reshape(-1)
        t1 = time.perf_counter()

        # Train: epochs passes over the rollout in minibatches
        net.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(rollout)
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
        minibatch = T * B // cfg.minibatches
        stopped_early = False
        for _ in range(cfg.epochs):
            order = torch.randperm(T * B, device=device, generator=generator)
            for start in range(0, T * B, minibatch):
                idx = order[start:start + minibatch]
                adv = advantages[idx]
                if cfg.normalize_advantages:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                out = ppo_losses(net, decode(idx), legal_store[idx], action_store[idx], log_prob_store[idx],
                                 old_values[idx], adv, returns[idx], cfg.clip_range, cfg.value_clip_range)
                loss = out["policy_loss"] + cfg.value_loss_weight * out["value_loss"] - cfg.entropy_coef * out["entropy"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                out["grad_norm"] = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                for k in stats:
                    stats[k].append(out[k].detach())
            if cfg.target_kl > 0 and torch.stack(stats["approx_kl"][-cfg.minibatches:]).mean().item() > cfg.target_kl:
                stopped_early = True
                break
        t2 = time.perf_counter()

        moves = (rollout + 1) * rollout_moves
        metrics = {"moves": moves, "learning_rate": learning_rate(rollout), "seconds_play": t1 - t0,
                   "seconds_train": t2 - t1, "stopped_early": float(stopped_early),
                   "value_mean": float(old_values.mean()), "return_mean": float(returns.mean()),
                   "explained_variance": float(1 - (returns - old_values).var() / (returns.var() + 1e-8)),
                   **{k: torch.stack(v).float().mean().item() for k, v in stats.items()},
                   **{f"speed_weight_{s}": w for s, w in speed_sampler.weights().items()}}
        if not plain_obs and codec.mismatches():
            metrics["obs_decode_mismatches"] = codec.mismatches()
        if episode_scores:
            metrics["train_episode_score"] = float(np.mean(episode_scores))
            metrics["train_episode_moves"] = float(np.mean(episode_moves))
            metrics["train_episodes"] = len(episode_scores)
            episode_scores.clear()
            episode_moves.clear()
        if moves % cfg.eval_every < rollout_moves:
            metrics.update(evaluate(with_search=False))
            if metrics["eval_mean_score"] > best:
                best = metrics["eval_mean_score"]
                save("model-best.ckpt.pt", rollout + 1)
        if moves % cfg.search_eval_every < rollout_moves:
            metrics.update(evaluate(with_search=True))
            if metrics["search_eval_mean_score"] > best_search:
                best_search = metrics["search_eval_mean_score"]
                save("model-best-search.ckpt.pt", rollout + 1)
        if moves % cfg.checkpoint_every < rollout_moves:
            save("model-latest.ckpt.pt", rollout + 1)
            save(f"model-moves{moves:09}.ckpt.pt", rollout + 1)
        wandb.log(metrics, step=moves)
        progress.set_postfix(score=metrics.get("train_episode_score"), kl=metrics["approx_kl"])

    save("model-latest.ckpt.pt", num_rollouts)
    wandb.finish()


if __name__ == "__main__":
    main()
