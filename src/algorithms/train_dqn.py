from argparse import ArgumentParser
from contextlib import nullcontext
import copy
import itertools
from pathlib import Path
import shutil
import signal
import time
import numpy as np

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm


from pacbot_rs import PacmanGym


_exit_requested = False

def _request_exit(signum, frame):
    global _exit_requested
    _exit_requested = True
    print("\nExit requested — saving checkpoint before stopping...")


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)

import models
from policies import EpsilonGreedy, MaxQPolicy
from replay_buffer import ReplayBuffer
from timing import time_block
from utils import lerp, reset_env, step_env_until_done, step_env_once, select_device, OBS_SHAPE, NUM_ACTIONS, DETERMINISTIC_START_CONFIGURATION


hyperparam_defaults = {
    "learning_rate": 0.0001,
    "batch_size": 512,
    "num_iters": 2_000_000,
    "replay_buffer_size": 30_000,
    "num_parallel_envs": 32,
    "random_start_proportion": 0.5,
    "experience_steps": 4,
    "target_network_update_steps": 5_000,  # Update the target network every ___ steps.
    "evaluate_steps": 10,  # Evaluate every ___ steps.
    "initial_epsilon": 0.8,
    "final_epsilon": 0.4,
    "discount_factor": 0.99,
    "reward_scale": 1 / 50,
    "grad_clip_norm": 10_000,
    "model": "QNetV2"
}

parser = ArgumentParser()
parser.add_argument("--eval", metavar="CHECKPOINT", default=None)
parser.add_argument("--finetune", metavar="CHECKPOINT", default=None)
parser.add_argument("--no-wandb", action="store_true")
parser.add_argument("--no-eval", action="store_true")
parser.add_argument("--checkpoint-dir", default="checkpoints")
parser.add_argument("--device", default=None)
parser.add_argument("--save-legacy-checkpoint", action="store_true")
for name, default_value in hyperparam_defaults.items():
    parser.add_argument(
        f"--{name}",
        type=type(default_value),
        default=default_value,
        help="Default: %(default)s",
    )
args = parser.parse_args()
if not hasattr(models, args.model):
    parser.error(f"Invalid --model: {args.model!r} (must be a class in models.py)")

device = select_device(args.device)
print(f"Using device: {device}")


reward_scale: float = args.reward_scale
# Prepare wandb init kwargs. We may want to resume an existing run when finetuning from
# a checkpoint that recorded its wandb run id.
wandb_init_kwargs = dict(
    project="pacbot-ind-study",
    tags=["DQN"] + (["finetuning"] if args.finetune else []),
    config={
        "device": str(device),
        **{name: getattr(args, name) for name in hyperparam_defaults.keys()},
    },
    mode="disabled" if args.eval or args.no_wandb else "online",
)

# If finetuning from a checkpoint, try to read a metadata checkpoint to obtain the
# original wandb run id so we can resume the run instead of creating a new one.
if args.finetune and not (args.eval or args.no_wandb):
    try:
        ckpt_path = Path(args.finetune)
        if ckpt_path.exists():
            # Attempt to load as a metadata dict first (recommended). If it's a legacy
            # full-model .pt, torch.load will still return a Module instance and we
            # won't find metadata.
            data = torch.load(ckpt_path, map_location="cpu")
            if isinstance(data, dict) and "wandb_run_id" in data:
                wandb_init_kwargs.update({"id": data.get("wandb_run_id"), "resume": "allow"})
                print(f"Will resume wandb run id {data.get('wandb_run_id')} from checkpoint {ckpt_path}")
    except Exception:
        # Don't block training just because metadata couldn't be read; fall back to
        # starting a new run.
        pass

wandb.init(**wandb_init_kwargs)


# Initialize the Q network.
model_class = getattr(models, wandb.config.model)
q_net = model_class(OBS_SHAPE, NUM_ACTIONS).to(device)
print(f"q_net has {sum(p.numel() for p in q_net.parameters())} parameters")

# variables used when resuming from a checkpoint (finetuning)
resume_iter = 0
resume_epsilon = 0
resume_optimizer_state = None
resume_best_eval_score = -float("inf")
if args.finetune:
    # Load checkpoint. Support two formats:
    # 1) metadata dict: {
    #      "state_dict": ..., "wandb_run_id": "<id>", "iter_num": int, "config": {...}
    #    }
    # 2) legacy: a full torch-saved Module instance.
    ckpt_path = Path(args.finetune)
    if ckpt_path.exists():
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            print(f"Finetuning from checkpoint metadata {args.finetune}")
            q_net.load_state_dict(loaded["state_dict"])  # type: ignore[arg-type]
            # If the checkpoint recorded the training iteration or epsilon, record
            # them so we can resume training exactly where it left off.
            resume_iter = int(loaded.get("iter_num", 0))
            resume_epsilon = loaded.get("epsilon", None)
            resume_optimizer_state = loaded.get("optimizer_state_dict", None)
            resume_best_eval_score = float(loaded.get("best_eval_score", -float("inf")))
            ckpt_wandb_id = loaded.get("wandb_run_id")
            if ckpt_wandb_id:
                print(f"Loaded checkpoint was created by wandb run id: {ckpt_wandb_id}")
        else:
            # Legacy checkpoint saved as full model.
            print(f"Finetuning from parameters from legacy checkpoint {args.finetune}")
            q_net = loaded
    else:
        raise FileNotFoundError(f"--finetune checkpoint not found: {args.finetune}")


@torch.no_grad()
def evaluate_episode(max_steps: int = 1000) -> tuple[int, int, bool, int, int, int, float]:
    """
    Performs a single evaluation episode.

    Returns (score, total_steps, is_board_cleared, pellets_start, pellets_end, purgatory_pellets, ghost_proximities).
    """
    gym = PacmanGym(DETERMINISTIC_START_CONFIGURATION)
    pellets_start = gym.remaining_pellets()

    q_net.eval()
    policy = MaxQPolicy(q_net)

    done, step_num = step_env_until_done(gym, policy, device, max_steps=max_steps)

    is_board_cleared = done and gym.lives() == 3
    pellets_end = 0 if is_board_cleared else gym.remaining_pellets()

    return (gym.score(), step_num, is_board_cleared, pellets_start, pellets_end, gym.purgatory_pellets, gym.ghost_proximities)


def train():
    # Initialize the replay buffer.
    replay_buffer = ReplayBuffer(
        maxlen=wandb.config.replay_buffer_size,
        policy=EpsilonGreedy(
            MaxQPolicy(q_net),
            NUM_ACTIONS,
            # use epsilon recovered from checkpoint if finetuning,
            # otherwise use configured initial epsilon
            resume_epsilon if resume_epsilon is not None else (wandb.config.initial_epsilon if args.finetune else 1.0),
        ),
        num_parallel_envs=wandb.config.num_parallel_envs,
        random_start_proportion=wandb.config.random_start_proportion,
        device=device,
    )
    replay_buffer.fill()
    replay_buffer.policy.epsilon = float(resume_epsilon) if resume_epsilon is not None else wandb.config.initial_epsilon

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(q_net.parameters(), lr=wandb.config.learning_rate)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print("Restored optimizer state from checkpoint")

    # Automatic Mixed Precision stuff.
    use_amp = False  # device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()
    )

    best_eval_score = resume_best_eval_score

    def build_metadata(iter_num: int) -> dict:
        metadata = {
            "state_dict": q_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter_num": iter_num,
            "epsilon": replay_buffer.policy.epsilon,
            "best_eval_score": best_eval_score,
            "config": dict(wandb.config) if hasattr(wandb, "config") else {},
        }
        try:
            if wandb.run is not None and getattr(wandb.run, "id", None):
                metadata["wandb_run_id"] = wandb.run.id
        except Exception:
            pass
        return metadata

    prev_sigint = signal.signal(signal.SIGINT, _request_exit)

    target_q_net = copy.deepcopy(q_net)
    target_q_net.eval()

    for iter_num in tqdm(range(resume_iter, wandb.config.num_iters), smoothing=0.01):
        if _exit_requested:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            metadata_path = checkpoint_dir / "q_net-latest.ckpt.pt"
            atomic_torch_save(build_metadata(iter_num), metadata_path)
            shutil.copyfile(metadata_path, checkpoint_dir / f"q_net-iter{iter_num:07}.ckpt.pt")
            print(f"Checkpoint saved at iter {iter_num}.")
            break
        if iter_num % wandb.config.target_network_update_steps == 0:
            with time_block("Update target network"):
                # Update the target network.
                target_q_net = copy.deepcopy(q_net)
                target_q_net.eval()

        with time_block("Collate batch"):
            # Sample and collate a batch.
            with device:
                batch = replay_buffer.sample_batch(wandb.config.batch_size)
                obs_batch = torch.stack([item.obs for item in batch])
                next_obs_batch = torch.stack(
                    [
                        torch.zeros(OBS_SHAPE) if item.next_obs is None else item.next_obs
                        for item in batch
                    ]
                )
                done_mask = torch.tensor([item.next_obs is None for item in batch])
                next_action_masks = torch.tensor([item.next_action_mask for item in batch])
                action_batch = torch.tensor([item.action for item in batch])
                reward_batch = torch.tensor(
                    [item.reward * wandb.config.reward_scale for item in batch]
                )

        with time_block("Compute target Q values"):
            # Get the target Q values.
            double_dqn = True
            with torch.no_grad():
                with autocast:
                    next_q_values = target_q_net(next_obs_batch)
                    next_q_values[~next_action_masks] = -torch.inf
                    if double_dqn:
                        online_next_q_values = q_net(next_obs_batch)
                        online_next_q_values[~next_action_masks] = -torch.inf
                        next_actions = online_next_q_values.argmax(dim=1)
                    else:
                        next_actions = next_q_values.argmax(dim=1)
                    returns = next_q_values[range(len(batch)), next_actions]
                    discounted_returns = wandb.config.discount_factor * returns
                    discounted_returns[done_mask] = 0.0
                    target_q_values = reward_batch + discounted_returns

        with time_block("Compute loss and update parameters"):
            # Compute the loss and update the parameters.
            with time_block("optimizer.zero_grad()"):
                q_net.train()
                optimizer.zero_grad()

            with time_block("Forward pass"):
                with autocast:
                    all_predicted_q_values = q_net(obs_batch)
                    predicted_q_values = all_predicted_q_values[range(len(batch)), action_batch]
                    loss = F.mse_loss(predicted_q_values, target_q_values)

            with time_block("Backward pass"):
                grad_scaler.scale(loss).backward()
            with time_block("Clip grad norm"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    q_net.parameters(),
                    max_norm=wandb.config.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            with time_block("Step optimizer"):
                grad_scaler.step(optimizer)
                grad_scaler.update()

        with torch.no_grad():
            # Log metrics.
            metrics = {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "exploration_epsilon": replay_buffer.policy.epsilon,
                "avg_predicted_value": (
                    all_predicted_q_values.amax(dim=1).mean().item() / wandb.config.reward_scale
                ),
                "avg_target_q_value": target_q_values.mean() / wandb.config.reward_scale,
            }
            if iter_num % wandb.config.evaluate_steps == 0:
                with time_block("Evaluate the current agent"):
                    # Evaluate the current agent.
                    (
                        eval_episode_score,
                        eval_episode_steps,
                        cleared_board,
                        pellets_start,
                        pellets_end,
                        purgatory_pellets,
                        ghost_proximities,
                    ) = evaluate_episode()
                    metrics.update(
                        eval_episode_score=eval_episode_score,
                        eval_episode_steps=eval_episode_steps,
                        cleared_board=int(cleared_board),
                        eval_pellets_start=pellets_start,
                        eval_pellets_end=pellets_end,
                        purgatory_pellets=purgatory_pellets,
                        ghost_proximities=ghost_proximities,
                    )
                    if eval_episode_score > best_eval_score:
                        best_eval_score = eval_episode_score
                        checkpoint_dir = Path(args.checkpoint_dir)
                        checkpoint_dir.mkdir(exist_ok=True)
                        atomic_torch_save(build_metadata(iter_num), checkpoint_dir / "q_net-best.ckpt.pt")
            wandb.log(metrics)

        if iter_num % 500 == 0:
            with time_block("Save checkpoint"):
                # Save a checkpoint.
                checkpoint_dir = Path(args.checkpoint_dir)
                checkpoint_dir.mkdir(exist_ok=True)
                pt_path = checkpoint_dir / "q_net-latest.eval.pt"
                if args.save_legacy_checkpoint:
                    atomic_torch_save(q_net, pt_path)
                    shutil.copyfile(pt_path, checkpoint_dir / f"q_net-iter{iter_num:07}.pt")

                # saves a metadata checkpoint containing the state_dict and
                # metadata like wandb run id and iteration number in order to resume wandb runs
                metadata_path = checkpoint_dir / "q_net-latest.ckpt.pt"
                iter_ckpt_path = checkpoint_dir / f"q_net-iter{iter_num:07}.ckpt.pt"
                atomic_torch_save(build_metadata(iter_num), metadata_path)
                shutil.copyfile(metadata_path, iter_ckpt_path)

                if iter_num % 1_000 == 0:
                    try:
                        wandb.log_artifact(str(iter_ckpt_path), name=f"q_net-iter{iter_num:07}", type="model")
                    except Exception:
                        # Non-fatal: logging artifacts may fail if wandb is disabled.
                        pass

        # Anneal the exploration policy's epsilon.
        replay_buffer.policy.epsilon = lerp(
            wandb.config.initial_epsilon,
            wandb.config.final_epsilon,
            iter_num / (wandb.config.num_iters - 1),
        )

        # Collect experience.
        with time_block("Collect experience"):
            for _ in range(wandb.config.experience_steps):
                replay_buffer.generate_experience_step()

    signal.signal(signal.SIGINT, prev_sigint)


@torch.no_grad()
def visualize_agent():
    gym = PacmanGym(DETERMINISTIC_START_CONFIGURATION)

    q_net.eval()
    policy = MaxQPolicy(q_net)

    print()
    print(f"Step 0")
    gym.print_game_state()
    print()

    for step_num in itertools.count(1):
        time.sleep(0.1)

        reward, done = step_env_once(gym, policy, device)
        print("reward:", reward)

        print()
        print(f"Step {step_num}")
        gym.print_game_state()
        print()

        if done:
            break


if args.eval:
    # handle loading from both legacy checkpoints and metadata checkpoints
    
    model_class = getattr(models, wandb.config.model)
    q_net = model_class(OBS_SHAPE, NUM_ACTIONS).to(device)
    print(f"q_net has {sum(p.numel() for p in q_net.parameters())} parameters")
    
    ckpt_path = Path(args.eval)
    if ckpt_path.exists():
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            q_net.load_state_dict(loaded["state_dict"])
        else:
            q_net = loaded
    else:
        raise FileNotFoundError(f"--eval checkpoint not found: {args.eval}")
else:
    try:
        train()
    except KeyboardInterrupt:
        pass
    wandb.finish()

if not args.no_eval:
    while True:
        visualize_agent()
        try:
            input("Press enter to view another episode")
        except KeyboardInterrupt:
            print()
            break
