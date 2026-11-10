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
 - Games are played in separate CPU processes so the training process only does the network updates on the device. Actor processes (`--num-actors`, each using `--actor-threads` CPU threads) play the parallel environments while the network trains, and evaluation workers (`--eval-workers`) play the periodic evaluation games in the background, skipping an evaluation when they are all busy. Both default to numbers that fit the machine's CPU cores (printed at startup). Actors play with the weights from one update earlier than collecting in the training process would; `--num-actors 0 --eval-workers 0` runs everything in the training process, as before.

Checkpoints record the version of the environment rules they were trained with as `env_version` (see `LATEST_ENV_VERSION` in `pacbot_rs/src/env.rs`); checkpoints without it are version 1. The evaluation scripts, `--eval`, and the bot client use each checkpoint's version, so older checkpoints keep the rules they were trained with. When changing rules that affect what the agent sees or which model plays, add a new version instead of changing an old one.

You can also change **hyperparameters** with command-line arguments. To see the full list, pass `--help` or take a look at the `hyperparam_defaults` dictionary near the top of `train_dqn.py`. In particular:
 - If you're running out of memory, consider reducing the `batch_size`.
 - `num_iters` controls the total runtime of the script execution, and also indirectly the value of $\epsilon$ (epsilon), which controls the amount of random exploration ($\epsilon$ starts at `initial_epsilon` and linearly decreases to `final_epsilon` after `num_iters`). Of course, you can always set `num_iters` to a large value (like the default) and just kill the script once it reaches a satisfactory level of performance.

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
