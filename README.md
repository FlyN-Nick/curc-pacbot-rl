# What is this repo?

This repository contains reinforcement learning implementations for creating agents to control a robot in the Harvard PacBot competition.

I implemented three RL algorithms from scratch: **DQN**, **PPO**, and **AlphaZero**. The main training/evaluation scripts are in `src/algorithms/`.

The rest of the `src/` directory contains supporting Python code.

The game environment, as well as some other performance-sentitive code such as the vectorized Monte Carlo tree search implementation for AlphaZero, are implemented in Rust as a Python extension module (using `pyo3` and `maturin`). This code is located in `pacbot_rs/`.


# Usage

Python 3.11 and newer is supported (Python 3.11–3.14 is currently checked). The project currently has a temporary Python `<3.15` cap because of its Triton dependency. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Rust](https://www.rust-lang.org/tools/install), then install a Python interpreter with uv:

```sh
uv python install 3.11
```

DQN has generally worked better than PPO and AlphaZero for us, so the main script you should run is `src/algorithms/train_dqn.py`.

## Setup

Install the locked dependencies. uv creates a project-local `.venv` automatically:

```sh
uv sync --python 3.11
```

Initially, and after making changes to Rust code in `pacbot_rs/`, build the extension and install it into the project environment:
```sh
uv run --project . --directory pacbot_rs maturin develop --release
```

## Training

To start a training run:
```bash
uv run --project . --directory src python -m algorithms.train_dqn
```
 - To fine-tune (start from an existing checkpoint instead of initializing the model randomly), pass `--finetune path/to/checkpoint`.
 - By default, model checkpoints will be saved to `./checkpoints/`. You can change this by passing `--checkpoint-dir path/to/dir`.
 - By default, this will use [Weights & Biases](https://wandb.ai/) to log checkpoints and various metrics to aid debugging and tracking training progress (see [Weights & Biases setup](#weights--biases-setup) below). To disable this, pass `--no-wandb`.
 - By default, the device is picked in this order: `mps` (Apple Silicon GPU) if available, then `cuda` (NVIDIA GPU), then `cpu`. To manually set the PyTorch device, pass `--device <device>` (for example `--device cuda` or `--device cpu`).
 - Games are played in separate CPU processes so the training process only does the network updates on the device. Actor processes (`--num-actors`, each using `--actor-threads` CPU threads) play the parallel environments continuously, a few steps ahead, with weights a few updates old, while every environment still contributes the same transitions per iteration. Evaluation workers (`--eval-workers`) play the periodic evaluation games in the background, skipping an evaluation when they are all busy (logged as `skipped_evals`). Both default to numbers that fit the machine's CPU cores (printed at startup); `--num-actors 0 --eval-workers 0` runs everything in the training process, as before.
 - Metrics are logged one iteration late, so reading them back doesn't make the CPU wait for the GPU. Pass `--timing` to print how long each part of every iteration takes (CPU time; GPU work runs asynchronously).
 - The replay buffer lives on the training device in compact form (about 30% of float32 size, decoded exactly; see `src/replay_storage.py`; on Apple Silicon (MPS) it holds plain float32 instead, about 1.7 GB for the default 30,000 transitions, because decoding each batch there is slow), and on CUDA each update, batch sampling included, is replayed as a CUDA graph (`--no-cuda-graph` runs it eagerly; see `src/dqn_update.py`).

To run several trainings at once on one machine, split the CPU cores between them, and give each its own `--checkpoint-dir`. The runs share the GPU's update time, so together they go only a little faster than one run alone: on an RTX 5070 Ti with an 8-core/16-thread CPU, about 90 iterations/s for one run and about 95 combined for two or three. Settings that worked there:

| Runs at once | Per run |
| --- | --- |
| 1 | defaults |
| 2 | `--num-actors 2 --actor-threads 2 --eval-workers 1` |
| 3 | `--num-actors 2 --actor-threads 1 --eval-workers 1` |

On Apple Silicon, an M2 Max with 32 GB reached about 20 iterations/s for one run over 5,000 iterations (17 before tuning), with the defaults there: `--num-actors 4 --actor-threads 1 --eval-workers 3`, which fill its 8 performance cores. The update step on the GPU takes about 30 ms, so that, not the actors, is what limits it. `torch.compile` and channels-last didn't speed up the MPS update, and Adam's `fused` kernel saved about 1 ms but changed the training results.

`--bf16` (opt-in, CUDA only) runs the network's convolutions and matrix products in bfloat16: `update` for the gradient pass only, `all` for the target passes too. It mainly helps runs that share a GPU (there, with `all`, one run reached about 100 iterations/s, two about 135 combined and three about 140), at the cost of rounding those results to about 0.4% instead of TF32's 0.05%. Weights, the optimizer, the loss and game play stay in float32. It hasn't been compared with the default over a long run; the setting is recorded in the W&B config and checkpoint metadata.

Checkpoints record the version of the environment rules they were trained with as `env_version` (see `LATEST_ENV_VERSION` in `pacbot_rs/src/env.rs`); checkpoints without it are version 1. The evaluation scripts, `--eval`, and the bot client use each checkpoint's version, so older checkpoints keep the rules they were trained with. When changing rules that affect what the agent sees or which model plays, add a new version instead of changing an old one.

You can also change **hyperparameters** with command-line arguments. To see the full list, pass `--help` or take a look at the `hyperparam_defaults` dictionary near the top of `train_dqn.py`. In particular:
 - If you're running out of memory, consider reducing the `batch_size`.
 - `num_iters` controls the total runtime of the script execution, and also indirectly the value of $\epsilon$ (epsilon), which controls the amount of random exploration ($\epsilon$ starts at `initial_epsilon` and decreases linearly to `final_epsilon` over the first `epsilon_decay_fraction` of `num_iters`, then stays there; the default 1.0 decays over the whole run). Of course, you can always set `num_iters` to a large value (like the default) and just kill the script once it reaches a satisfactory level of performance.
 - `huber_delta` > 0 trains with the Huber loss instead of MSE: quadratic for errors up to `huber_delta` (in scaled reward units, so 1.0 is `1 / reward_scale` = 50 game points), linear beyond, so rare large errors (deaths, ghosts eaten) don't dominate the gradient. The default 0 keeps MSE.
 - `--single-model` trains one network that plays the whole game, super pellets included, instead of handing control to the fixed super pellet model (`checkpoints_known/superpelletmode.safetensors`) whenever ghosts are close or frightened. Checkpoints record it, and the evaluation scripts play them the same way. The robot client (`bot_client/dqn_module.py`) doesn't support single-model checkpoints yet.
 - A game from the normal start is deterministic: the ghosts have no randomness, and the only thing that varies between training games is Pacman's speed (4 to 14 game ticks per action, 8 being normal). So repeating an evaluation game gives the same game. Every `best_eval_steps` iterations (and at the last one), the current network plays one game at each of those 11 speeds, and the one with the best mean score so far is saved as `q_net-best.ckpt.pt` (logged as `candidate_mean_score`, with `candidate_min_score`, `candidate_normal_speed_score` and `best_eval_mean_score`). The single games every `evaluate_steps` iterations, at the normal speed, draw the learning curve; from one checkpoint to the next their score jumps around a lot.

### Weights & Biases setup

Training logs metrics to the `pacbot-ind-study` project and uploads a checkpoint as a model artifact every 1,000 iterations. `wandb` is already installed by `uv sync`. To set it up once per machine:

1. Create an account at <https://wandb.ai> (free for academic use).
2. Copy your API key from <https://wandb.ai/authorize>.
3. Log in from this repo, then paste the key when prompted. It is saved to `~/.netrc`:
   ```sh
   uv run wandb login
   ```
   On a remote or headless machine, you can set `WANDB_API_KEY=<your key>` in the environment instead.

Runs go to your default W&B entity (your username). To log to a shared team instead, set `WANDB_ENTITY=<team name>` before training. Results appear at `https://wandb.ai/<entity>/pacbot-ind-study`. When you fine-tune from a checkpoint that recorded a W&B run id, training resumes that run instead of starting a new one.

To train without an account or network connection, pass `--no-wandb`. Checkpoints are still saved locally to `--checkpoint-dir`.

## Visualizing a trained agent

To show an animated visualization in the terminal of the agent being controlled by a particular checkpoint, use `--eval`:
```bash
uv run --project . --directory src python -m algorithms.train_dqn --eval path/to/checkpoint.pt
```

For example, to visualize the best tracked checkpoint (the path is relative to `src/`):
```bash
uv run --project . --directory src python -m algorithms.train_dqn --eval best_checkpoints/urlriljg/q_net-iter0619000.ckpt.pt
```
