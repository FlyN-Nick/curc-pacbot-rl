# Weights & Biases setup

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
