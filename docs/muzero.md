# The MuZero family (LightZero)

MuZero learns its own model of the game and searches inside it, where AlphaZero searches the
real game. We train MuZero, EfficientZero, Gumbel MuZero and Stochastic MuZero with
[LightZero](https://github.com/opendilab/LightZero) (OpenDILab's tested implementations), on
our Rust game:

- `src/lightzero_pacbot/pacbot_env.py` wraps the game as a LightZero environment.
  - Training games are AlphaZero's: a speed drawn by the speed sampling (equal moves per speed,
    up to 18, by default), and a random start for a quarter of the games.
  - Evaluation games are the fixed-seed games of `eval_az_seeds.py`.
  - The observation is the same 17 × 28 × 31 network input, and the reward is the game's reward.
- `src/lightzero_pacbot/configs.py` holds the settings.
  - Where the algorithms allow, they match our AlphaZero: discount 0.99 and 10-step value targets.
  - LightZero's residual networks run at the board's full resolution and without image
    augmentation, which suits camera frames rather than a grid.
- `src/algorithms/train_lightzero.py` trains.
- `src/eval_lightzero_seeds.py` scores checkpoints on the fixed-seed games, in
  `eval_az_seeds.py`'s CSV format.

## Setup

LightZero needs numpy below 2, which the main project can't use, so it gets its own locked uv
project, `lightzero/`, with its own environment (`lightzero/.venv`). From the repo root:

```sh
uv sync --project lightzero
uv run --project ../lightzero --directory pacbot_rs maturin develop --release
```

The second command builds our game into that environment; rerun it after changing Rust code.
LightZero 0.2.0 publishes builds for Linux and macOS only, so this doesn't work on Windows
(use WSL). On Linux and Windows the lock installs PyTorch's CUDA 12.8 build, as the main project
does.

## Training

```sh
cd src
uv run --project ../lightzero python -m algorithms.train_lightzero \
  --algorithm muzero --checkpoint-dir checkpoints/mz1
```

- `--algorithm` is `muzero`, `efficientzero`, `gumbel_muzero` or `stochastic_muzero`.
- `--num_moves` sets the game moves to collect (40M by default, as our AlphaZero runs). `--num_simulations` sets the search's simulations per move (32 by default).
- `--full_game` trains on full games (three lives, board after board).
- `--help` lists the rest: collecting games, replay ratio, reanalysis, network size, learning rate.
- Each launch writes LightZero's TensorBoard logs and checkpoints to its own folder, `--checkpoint-dir/run-YYMMDD-HHMMSS`. Checkpoints go in `ckpt/iteration_N.pth.tar`, with `ckpt/ckpt_best.pth.tar` for the best evaluation.
- It evaluates every `--eval_freq` training iterations on one fixed-seed game per speed.
- `--resume` starts from a checkpoint's network. LightZero doesn't save its replay buffer or optimizer.

## Evaluating

```sh
uv run --project ../lightzero python eval_lightzero_seeds.py 48 results.csv \
  muzero:checkpoints/mz1/run-.../ckpt/ckpt_best.pth.tar:128
```

- Each checkpoint is written as `ALGORITHM:PATH:SIMULATIONS`.
- Pass `--num_res_blocks` and `--num_channels` if the run changed them.
- `PACBOT_FULL_GAME=1` plays full games, as with `eval_az_seeds.py`.

`tests/test_lightzero.py` checks the environment and that every algorithm's search runs as
LightZero's evaluator and our scorer call it. It's skipped without LightZero; to run it, from
`src/`: `uv run --project ../lightzero python -m unittest discover -s ../tests -p test_lightzero.py`.
CI runs it on Linux.

## LightZero 0.2.0 workarounds

- Stochastic MuZero's eval forward rejects the `timestep` argument LightZero's own evaluator
  passes. `src/lightzero_pacbot/stochastic_muzero.py` patches the policy to drop it, keeping the
  registered name `stochastic_muzero` that LightZero's training entry checks for.
- Gumbel and Stochastic MuZero's eval forward searches with the collect-mode network, so the
  scorer sets up collect mode as well as eval mode.
