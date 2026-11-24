# Training on a shared machine (`pacbot-train`)

On a machine several people train on, start runs with `pacbot-train` (in `tools/launcher/`; install it once with `sudo bash tools/launcher/install.sh`, which also creates the `pacbot` group's shared state in `/var/lib/pacbot-train`). It keeps at most three runs training at once, counting runs started any other way (DQN, AlphaZero, PPO or LightZero training), and starts waiting runs in the order they were queued:

```bash
pacbot-train start my-run -- --single-model --random_start_proportion 0.25   # queue it and return
pacbot-train list              # everyone's runs: queued, running, paused, done
pacbot-train log my-run        # follow its training log (add a user name for someone else's)
pacbot-train stop my-run       # stop it; starting my-run again resumes from its latest checkpoint
```

Run it from your own checkout: it trains with `../.venv/bin/python -m algorithms.train_dqn` in that checkout's `src/`, with `--checkpoint-dir checkpoints/NAME` and `WANDB_NAME=NAME`, logging to `src/checkpoints/NAME/train.log`. Runs keep going after you log out. `pacbot-train run NAME` does the same but waits until the run finishes, so a script can score the run afterwards.

Options before the run's name choose what it trains and where:

```bash
pacbot-train start --algorithm alphazero az-long -- --num_moves 80000000
pacbot-train start --algorithm alphazero --from checkpoints/az-long/model-latest.ckpt.pt az-longer -- --num_moves 160000000
pacbot-train start --algorithm lightzero mz-gumbel -- --algorithm gumbel_muzero
pacbot-train start --checkpoints /path/to/shared/checkpoints my-run -- --single-model
```

- `--algorithm` is `dqn` (the default), `alphazero`, `ppo` or `lightzero`: the run trains with `src/algorithms/train_ALGORITHM.py`. LightZero runs use the checkout's `lightzero/.venv` ([the MuZero family](muzero.md)); the others use `.venv`.
- `--from CHECKPOINT` starts the run from that checkpoint the first time, e.g. to continue another run in a new folder (`--finetune` for DQN, `--resume` for the others). Once the run has a checkpoint of its own, it resumes from that instead.
- `--checkpoints DIR` keeps the run in `DIR/NAME` instead of the checkout's `src/checkpoints/NAME`.
- `pacbot-train list` shows each run's algorithm.

Stopping and resuming depends on the algorithm. DQN, AlphaZero and PPO save a checkpoint when stopped and resume from it (`model-latest.ckpt.pt` for AlphaZero and PPO). LightZero saves checkpoints every so often, not when stopped, and resumes with the network alone, so a resumed LightZero run loses the progress since its last checkpoint and starts its move count over.

The machine's owner limits how many teammates' runs may train at once with `pacbot-capacity N` (`wsl pacbot-capacity 0` from Windows, e.g. before using the GPU for something else). When it goes down, the newest teammate runs stop and wait; they resume when there's room again.

A DQN run that's stopped this way, with `pacbot-train stop`, with Ctrl-C, or with `kill PID` (SIGTERM to the training process, not its process group) saves a checkpoint, its target network and its replay memory (`replay.pt`, about 2.5 KB per stored transition), then exits with code 128 + the signal number. Resuming with `--finetune` from that checkpoint, in the same `--checkpoint-dir`, restores all of it, so training continues where it stopped; only the games in progress start over.
