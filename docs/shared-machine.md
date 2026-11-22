# Training on a shared machine (`pacbot-train`)

On a machine several people train on, start runs with `pacbot-train` (in `tools/launcher/`; install it once with `sudo bash tools/launcher/install.sh`, which also creates the `pacbot` group's shared state in `/var/lib/pacbot-train`). It keeps at most three runs training at once, counting runs started any other way, and starts waiting runs in the order they were queued:

```bash
pacbot-train start my-run -- --single-model --random_start_proportion 0.25   # queue it and return
pacbot-train list              # everyone's runs: queued, running, paused, done
pacbot-train log my-run        # follow its training log (add a user name for someone else's)
pacbot-train stop my-run       # stop it; starting my-run again resumes from its latest checkpoint
```

Run it from your own checkout: it trains with `../.venv/bin/python -m algorithms.train_dqn` in that checkout's `src/`, with `--checkpoint-dir checkpoints/NAME` and `WANDB_NAME=NAME`, logging to `src/checkpoints/NAME/train.log`. Runs keep going after you log out. `pacbot-train run NAME` does the same but waits until the run finishes (for scripts).

The machine's owner limits how many teammates' runs may train at once with `pacbot-capacity N` (`wsl pacbot-capacity 0` from Windows, e.g. before using the GPU for something else). When it goes down, the newest teammate runs stop and wait; they resume when there's room again.

A run that's stopped this way, with `pacbot-train stop`, with Ctrl-C, or with `kill PID` (SIGTERM to the training process, not its process group) saves a checkpoint, its target network and its replay memory (`replay.pt`, about 2.5 KB per stored transition), then exits with code 128 + the signal number. Resuming with `--finetune` from that checkpoint, in the same `--checkpoint-dir`, restores all of it, so training continues where it stopped; only the games in progress start over.
