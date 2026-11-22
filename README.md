# What is this repo?

Reinforcement learning agents that control a robot in the Pacbot competition. Three algorithms are implemented from scratch: **DQN**, **PPO** and **AlphaZero** (Gumbel search). The training and evaluation scripts are in `src/algorithms/` and `src/`. (`src/algorithms/deprecated/` keeps an early PPO, which no longer runs against the current game code, for reference.)

The game environment and other performance-sensitive code, such as the batched Gumbel search, are written in Rust as a Python extension module (`pyo3` and `maturin`), in `pacbot_rs/`.

AlphaZero gives the strongest agents at the moment. On the fixed-seed evaluation (528 games, speeds 4 to 14, 128 search simulations per move), the round-10 AlphaZero network scores about 11,600 on average, against about 10,700 for the best DQN played with the same search.

# Setup

Python 3.11 and newer is supported (3.11–3.14 are checked; Triton currently caps it below 3.15). Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Rust](https://www.rust-lang.org/tools/install), then:

```sh
uv python install 3.11
uv sync --python 3.11
uv run --project .. --directory pacbot_rs maturin develop --release   # again after changing Rust code
```

Training logs to [Weights & Biases](https://wandb.ai/) by default; see [W&B setup](docs/wandb.md), or pass `--no-wandb`.

# Training

## AlphaZero

`src/algorithms/train_alphazero.py` trains a policy and value network (`--model PVNetV2`) with Gumbel AlphaZero search through the real game (`pacbot_rs.gumbel_search`, in `pacbot_rs/src/gumbel.rs`): many games are searched at once, one network call per simulation wave, and the network learns the search's improved policy and 10-step returns bootstrapped from the search's value.

```sh
RAYON_NUM_THREADS=4 uv run --project . --directory src python -m algorithms.train_alphazero \
  --checkpoint-dir checkpoints/az1 --speed_sampling equal_moves --max_speed 18
```

- Games start at a random speed, some from random positions, and end at the first lost life or cleared board.
- `--full_game true` trains on full games instead, as in competition: three lives and board after board (`PacmanGym.full_game`).
  - A lost life costs the death penalty, and the game goes on from the respawn with the pellets kept.
  - A cleared board earns the clear bonus, and the next level starts with faster ghosts (the observation's speed channel shows it).
  - The observation is unchanged, so full-game checkpoints play wherever the others do.
- `--speed_sampling equal_moves --max_speed 18` gives each speed from 4 to 18 game ticks per move about the same number of training moves. The default trains on speeds 4 to 14 with the same number of games at each, which gives the slow speeds most of the moves (see `src/speed_sampling.py`).
- `--num_moves` sets the run's length (20M by default). `--num_simulations` sets the search simulations per training move (32 by default).
- **Checkpoints:**
  - It evaluates on fixed-seed games at every speed, with the network alone and with search, and keeps `model-best.ckpt.pt` and `model-best-search.ckpt.pt`.
  - SIGTERM or Ctrl-C saves `model-latest.ckpt.pt`, and `--resume` continues from it.
- `RAYON_NUM_THREADS` sets the search's CPU threads; on a busy machine, fewer is faster.
- Every hyperparameter is a flag: see `HYPERPARAMS` at the top of the script, or `--help`.

## PPO

`src/algorithms/train_ppo.py` trains the same policy and value network as AlphaZero (`--model PVNetV2`) with PPO, on the same games, so the two compare directly. PPO checkpoints play wherever AlphaZero's do: `eval_az_seeds.py` and the bot client, with or without search.

```sh
uv run --project . --directory src python -m algorithms.train_ppo \
  --checkpoint-dir checkpoints/ppo1 --speed_sampling equal_moves --max_speed 18
```

- **Each rollout:** each of `--num_games` games (256) plays `--rollout_steps` moves (128), sampling from the policy over the legal moves.
- **Then training:** `--epochs` passes (4) over the rollout in `--minibatches` (8), with PPO's clipped policy loss (`--clip_range` 0.2), the value loss (`--value_loss_weight` 0.5, optionally clipped with `--value_clip_range`) and an entropy bonus (`--entropy_coef` 0.01).
- **Advantages:** GAE (`--discount` 0.99, `--gae_lambda` 0.95), on rewards scaled by `--reward_scale` (0.02), so the value is in the search's units.
- **Shared options:** `--full_game`, `--speed_sampling`, `--max_speed`, the evaluations and checkpoints (`model-best.ckpt.pt`, `model-best-search.ckpt.pt`, `model-latest.ckpt.pt`; `--resume`) work as in AlphaZero training.
- **More:** `--target_kl` stops a rollout's epochs early once the policy has moved that far. Every hyperparameter is a flag (`HYPERPARAMS` at the top of the script, or `--help`).

## DQN

```sh
uv run --project . --directory src python -m algorithms.train_dqn --single-model
```

`--single-model` trains one network that plays the whole game. Without it, a fixed super pellet model takes over when ghosts are close or frightened. `--finetune path/to/checkpoint` starts from a checkpoint, and `--checkpoint-dir` sets where checkpoints go (`./checkpoints/` by default).

[DQN training](docs/dqn-training.md) covers its other options: actor processes, the replay buffer, several trainings on one machine, bfloat16 and the hyperparameters. On a machine several people train on, start runs with [`pacbot-train`](docs/shared-machine.md).

## MuZero, EfficientZero, Gumbel MuZero and Stochastic MuZero

These are trained with [LightZero](https://github.com/opendilab/LightZero) on our game, in their own environment; see [the MuZero family](docs/muzero.md).

## Environment versions

Checkpoints record the version of the environment rules they were trained with as `env_version` (see `LATEST_ENV_VERSION` in `pacbot_rs/src/env.rs`); checkpoints without it are version 1. The evaluation scripts, `--eval` and the bot client use each checkpoint's version, so older checkpoints keep the rules they were trained with. When changing rules that affect what the agent sees or which model plays, add a new version instead of changing an old one.

# Evaluating

`src/eval_az_seeds.py` plays the same fixed-seed games (seeds 1 to N at every speed) with each checkpoint, with search, and writes one CSV row per game. It needs an NVIDIA GPU. It plays AlphaZero checkpoints with their network, and DQN checkpoints with their Q values driving the same search.

```sh
cd src
PYTHONPATH=.:../model_package ../.venv/bin/python eval_az_seeds.py 48 results.csv \
  checkpoints/az1/model-latest.ckpt.pt:128 checkpoints/dqn1/q_net-best.ckpt.pt:128:0.3
```

- Each checkpoint is written as `PATH:SIMULATIONS[:TEMPERATURE]`. The temperature applies to DQN priors and defaults to 0.1; 0.3 worked best.
- `PACBOT_FULL_GAME=1` plays every game as a full game, adding `boards` (boards cleared) and `lives` (lives left) columns.
- `src/game_report.py` plays the same seeded games without search and reports how each one ends, and what was going on just before a lost life.

To watch a checkpoint play in the terminal:

```sh
uv run --project . --directory src python -m algorithms.train_dqn --eval best_checkpoints/urlriljg/q_net-iter0619000.ckpt.pt
```
