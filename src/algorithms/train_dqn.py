from argparse import ArgumentParser
from contextlib import nullcontext
import copy
import math
import itertools
import json
from pathlib import Path
import os
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
from utils import lerp, reset_env, step_env_until_done, step_env_once, select_device, OBS_SHAPE, NUM_ACTIONS, DETERMINISTIC_START_CONFIGURATION, ENV_VERSION, checkpoint_env_version, make_gym
import parallel_training


# Everything that runs the script is under this guard: worker processes (see
# parallel_training) re-import this module, and must only get its definitions.
if __name__ == "__main__":
    _hyperparams_path = Path(__file__).parent / "hyperparams.json"
    hyperparam_defaults: dict = json.loads(_hyperparams_path.read_text())

    parser = ArgumentParser()
    parser.add_argument("--eval", metavar="CHECKPOINT", default=None)
    parser.add_argument("--finetune", metavar="CHECKPOINT", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-legacy-checkpoint", action="store_true")
    parser.add_argument("--timing", action="store_true", help="Print how long each part of every iteration takes")
    parser.add_argument(
        "--bf16", choices=["none", "update", "all"], default="none",
        help="Run the network's convolutions and matrix products in bfloat16 on a CUDA GPU: "
        "'update' for the gradient pass only, 'all' for the target passes too. Faster, but "
        "rounds those results to ~0.4%% (TF32, the default for convolutions, rounds to "
        "~0.05%%). Weights, optimizer, loss and game play stay in float32. Default: %(default)s",
    )
    cpus = os.cpu_count() or 1
    parser.add_argument(
        # Actors' inference is compute-bound, so hyperthreads add little: about one 2-thread
        # actor per 2 physical cores (4 on an 8-core/16-thread CPU) was fastest
        "--num-actors", type=int, default=max(1, min(4, cpus // 4)),
        help="Processes that collect experience on the CPU while this one trains "
        "(0: collect here, between updates, as before). Default for this machine: %(default)s",
    )
    parser.add_argument(
        "--actor-threads", type=int, default=2,
        help="CPU threads per actor process. Default: %(default)s",
    )
    parser.add_argument(
        "--eval-workers", type=int, default=3 if cpus >= 12 else 1,
        help="Processes that play the evaluation games on the CPU in the background "
        "(0: play them here, blocking training, as before). Default for this machine: %(default)s",
    )
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

    if not args.timing:
        time_block = nullcontext
    device = select_device(args.device)
    print(f"Using device: {device}")
    if args.bf16 != "none":
        if device.type != "cuda":
            parser.error("--bf16 needs a CUDA device")
        print(f"bfloat16: {args.bf16}")
    if not args.eval:
        print(f"Actors: {args.num_actors} ({args.actor_threads} threads each), "
              f"evaluation workers: {args.eval_workers}")


    reward_scale: float = args.reward_scale
    # Prepare wandb init kwargs. We may want to resume an existing run when finetuning from
    # a checkpoint that recorded its wandb run id.
    wandb_init_kwargs = dict(
        project="pacbot-ind-study",
        tags=["DQN"] + (["finetuning"] if args.finetune else []),
        config={
            "device": str(device),
            "bf16": args.bf16,
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
    if device.type == "cuda":
        # Faster convolutions at the same precision: cuDNN times its algorithms once for each
        # (fixed) batch shape and keeps the fastest, and channels-last suits its kernels.
        torch.backends.cudnn.benchmark = True
        q_net = q_net.to(memory_format=torch.channels_last)

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
                if checkpoint_env_version(loaded) != ENV_VERSION:
                    print(f"Checkpoint was trained with environment version {checkpoint_env_version(loaded)}; "
                          f"finetuning continues with version {ENV_VERSION}")
                ckpt_wandb_id = loaded.get("wandb_run_id")
                if ckpt_wandb_id:
                    print(f"Loaded checkpoint was created by wandb run id: {ckpt_wandb_id}")
            else:
                # Legacy checkpoint saved as full model.
                print(f"Finetuning from parameters from legacy checkpoint {args.finetune}")
                q_net = loaded
        else:
            raise FileNotFoundError(f"--finetune checkpoint not found: {args.finetune}")


def evaluate_episode(max_steps: int = 1000) -> tuple[int, int, bool, int, int, int, float]:
    """
    Performs a single evaluation episode.

    Returns (score, total_steps, is_board_cleared, pellets_start, pellets_end, purgatory_pellets, ghost_proximities).
    """
    return parallel_training.evaluate_episode(q_net, device, ENV_VERSION, max_steps=max_steps)


def eval_metrics(result: tuple) -> dict:
    """Metrics to log for an evaluate_episode() result."""
    (
        eval_episode_score,
        eval_episode_steps,
        cleared_board,
        pellets_start,
        pellets_end,
        purgatory_pellets,
        ghost_proximities,
    ) = result
    return dict(
        eval_episode_score=eval_episode_score,
        eval_episode_steps=eval_episode_steps,
        cleared_board=int(cleared_board),
        eval_pellets_start=pellets_start,
        eval_pellets_end=pellets_end,
        purgatory_pellets=purgatory_pellets,
        ghost_proximities=ghost_proximities,
    )


class DeferredMetrics:
    """
    Logs each iteration's metrics one iteration later, so reading the GPU values back doesn't
    make the CPU wait for the GPU every iteration. Also checks the gradient norm then, which
    clip_grad_norm_(error_if_nonfinite=True) would otherwise do with a wait of its own.
    """

    names = ("loss", "grad_norm", "avg_predicted_value", "avg_target_q_value")

    def __init__(self, device: torch.device, reward_scale: float):
        self.device = device
        self.reward_scale = reward_scale
        self.pinned = device.type == "cuda"
        self._pending = None

    def add(self, host_metrics: dict, loss, grad_norm, all_predicted_q_values, target_q_values):
        """Queues this iteration's metrics and logs the previous iteration's."""
        values = torch.stack([
            loss.detach().float(),
            grad_norm.detach().float(),
            all_predicted_q_values.detach().float().amax(dim=1).mean() / self.reward_scale,
            target_q_values.float().mean() / self.reward_scale,
        ])
        if self.pinned:
            host = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
            host.copy_(values, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            host, event = values, None
        self.flush()
        self._pending = (host_metrics, host, event)

    def flush(self):
        """Logs the queued metrics, if any (waiting for the GPU to produce them)."""
        if self._pending is None:
            return
        host_metrics, host, event = self._pending
        self._pending = None
        if event is not None:
            event.synchronize()
        values = dict(zip(self.names, host.tolist()))
        if not math.isfinite(values["grad_norm"]):
            raise RuntimeError(
                f"The total norm of gradients is non-finite ({values['grad_norm']}), "
                "so the parameters can't be trusted. Stopping."
            )
        wandb.log({**values, **host_metrics})


def cpu_copy(obj):
    """A copy of obj with every tensor cloned to the CPU (for snapshots of live state)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def train():
    actors = args.num_actors > 0
    evaluator = None
    replay_buffer = None
    try:
        if args.eval_workers > 0:
            evaluator = parallel_training.BackgroundEvaluator(
                args.eval_workers, wandb.config.model, ENV_VERSION
            )
        # use epsilon recovered from checkpoint if finetuning,
        # otherwise use configured initial epsilon
        fill_epsilon = resume_epsilon if resume_epsilon is not None else (wandb.config.initial_epsilon if args.finetune else 1.0)
        # Initialize the replay buffer.
        if actors:
            replay_buffer = parallel_training.ActorReplayBuffer(
                maxlen=wandb.config.replay_buffer_size,
                q_net=q_net,
                model_name=wandb.config.model,
                epsilon=fill_epsilon,
                num_parallel_envs=wandb.config.num_parallel_envs,
                num_actors=args.num_actors,
                max_steps=wandb.config.experience_steps,
                device=device,
                threads_per_actor=args.actor_threads,
            )
        else:
            replay_buffer = ReplayBuffer(
                maxlen=wandb.config.replay_buffer_size,
                policy=EpsilonGreedy(MaxQPolicy(q_net), NUM_ACTIONS, fill_epsilon),
                num_parallel_envs=wandb.config.num_parallel_envs,
                random_start_proportion=wandb.config.random_start_proportion,
                device=device,
            )
        _train(replay_buffer, evaluator, actors)
    finally:
        for resource in (replay_buffer, evaluator):
            if hasattr(resource, "close"):
                resource.close()


def _train(replay_buffer, evaluator, actors: bool):
    replay_buffer.fill()
    replay_buffer.policy.epsilon = float(resume_epsilon) if resume_epsilon is not None else wandb.config.initial_epsilon

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(q_net.parameters(), lr=wandb.config.learning_rate)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print("Restored optimizer state from checkpoint")

    # Automatic Mixed Precision stuff. float16 (with loss scaling) is off; bfloat16 is opt-in
    # with --bf16 and needs no scaling (it has float32's range).
    use_amp = False  # device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def bf16_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        # No cast cache: copies cast under no_grad (the target passes) can't carry gradients
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False)

    best_eval_score = resume_best_eval_score

    def build_metadata(iter_num: int) -> dict:
        metadata = {
            "state_dict": q_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter_num": iter_num,
            "epsilon": replay_buffer.policy.epsilon,
            "best_eval_score": best_eval_score,
            "config": dict(wandb.config) if hasattr(wandb, "config") else {},
            "env_version": ENV_VERSION,
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

    deferred_metrics = DeferredMetrics(device, wandb.config.reward_scale)
    for iter_num in tqdm(range(resume_iter, wandb.config.num_iters), smoothing=0.01):
        if _exit_requested:
            deferred_metrics.flush()
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

        if actors:
            # The actors play this iteration's experience while this process updates the
            # network, with the weights from before the update.
            replay_buffer.start_collection(
                wandb.config.experience_steps,
                lerp(
                    wandb.config.initial_epsilon,
                    wandb.config.final_epsilon,
                    iter_num / (wandb.config.num_iters - 1),
                ),
            )

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
                # The per-transition values go to the device in one copy, from pinned memory so
                # the CPU doesn't wait for the GPU to finish its queued work first.
                packed = torch.tensor(
                    [(item.next_obs is None, *item.next_action_mask, item.action, item.reward) for item in batch],
                    dtype=torch.float64, device="cpu",
                )
                if device.type == "cuda":
                    packed = packed.pin_memory()
                packed = packed.to(device, non_blocking=True)
                done_mask = packed[:, 0].bool()
                next_action_masks = packed[:, 1:1 + NUM_ACTIONS].bool()
                action_batch = packed[:, 1 + NUM_ACTIONS].long()
                reward_batch = (packed[:, 2 + NUM_ACTIONS] * wandb.config.reward_scale).float()

        with time_block("Compute target Q values"):
            # Get the target Q values.
            double_dqn = True
            with torch.no_grad():
                with bf16_autocast(args.bf16 == "all"):
                    # masked_fill and gather instead of boolean-mask assignment and
                    # range indexing: the same values, without making the CPU wait for the GPU
                    next_q_values = target_q_net(next_obs_batch)
                    next_q_values = next_q_values.masked_fill(~next_action_masks, -torch.inf)
                    if double_dqn:
                        online_next_q_values = q_net(next_obs_batch)
                        online_next_q_values = online_next_q_values.masked_fill(~next_action_masks, -torch.inf)
                        next_actions = online_next_q_values.argmax(dim=1)
                    else:
                        next_actions = next_q_values.argmax(dim=1)
                    returns = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    discounted_returns = wandb.config.discount_factor * returns
                    discounted_returns = discounted_returns.masked_fill(done_mask, 0.0)
                    target_q_values = reward_batch + discounted_returns

        with time_block("Compute loss and update parameters"):
            # Compute the loss and update the parameters.
            with time_block("optimizer.zero_grad()"):
                q_net.train()
                optimizer.zero_grad()

            with time_block("Forward pass"):
                with bf16_autocast(args.bf16 in ("update", "all")):
                    all_predicted_q_values = q_net(obs_batch)
                    predicted_q_values = all_predicted_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    loss = F.mse_loss(predicted_q_values, target_q_values)

            with time_block("Backward pass"):
                grad_scaler.scale(loss).backward()
            with time_block("Clip grad norm"):
                # A non-finite norm is caught by DeferredMetrics (error_if_nonfinite would wait for the GPU)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    q_net.parameters(),
                    max_norm=wandb.config.grad_clip_norm,
                )
            with time_block("Step optimizer"):
                grad_scaler.step(optimizer)
                grad_scaler.update()

        with torch.no_grad():
            # Log metrics. The GPU values are logged by DeferredMetrics a step later.
            metrics = {
                "exploration_epsilon": replay_buffer.policy.epsilon,
            }
            if iter_num % wandb.config.evaluate_steps == 0:
                if evaluator is None:
                    with time_block("Evaluate the current agent"):
                        # Evaluate the current agent.
                        result = evaluate_episode()
                        metrics.update(eval_metrics(result))
                        if result[0] > best_eval_score:
                            best_eval_score = result[0]
                            checkpoint_dir = Path(args.checkpoint_dir)
                            checkpoint_dir.mkdir(exist_ok=True)
                            atomic_torch_save(build_metadata(iter_num), checkpoint_dir / "q_net-best.ckpt.pt")
                else:
                    # Play it in the background; if every worker is busy, skip this one.
                    # The snapshot is what gets saved if this turns out to be the best agent.
                    evaluator.submit(iter_num, q_net, lambda: cpu_copy(build_metadata(iter_num)))
            if evaluator is not None:
                for eval_iter, result, snapshot in evaluator.poll():
                    metrics.update(eval_metrics(result), eval_iter=eval_iter)
                    if result[0] > best_eval_score:
                        best_eval_score = result[0]
                        snapshot["best_eval_score"] = best_eval_score
                        checkpoint_dir = Path(args.checkpoint_dir)
                        checkpoint_dir.mkdir(exist_ok=True)
                        atomic_torch_save(snapshot, checkpoint_dir / "q_net-best.ckpt.pt")
                metrics["skipped_evals"] = evaluator.skipped
            deferred_metrics.add(metrics, loss, grad_norm, all_predicted_q_values, target_q_values)

        if iter_num % 500 == 0:
            # Check this update before saving it
            deferred_metrics.flush()
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
            if actors:
                replay_buffer.finish_collection()
            else:
                for _ in range(wandb.config.experience_steps):
                    replay_buffer.generate_experience_step()

    deferred_metrics.flush()
    signal.signal(signal.SIGINT, prev_sigint)


@torch.no_grad()
def visualize_agent():
    gym = make_gym(DETERMINISTIC_START_CONFIGURATION, visualize_env_version)

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


if __name__ == "__main__":
    # Visualize under the rules the shown model was trained with.
    visualize_env_version = ENV_VERSION
    if args.eval:
        # handle loading from both legacy checkpoints and metadata checkpoints
    
        model_class = getattr(models, wandb.config.model)
        q_net = model_class(OBS_SHAPE, NUM_ACTIONS).to(device)
        print(f"q_net has {sum(p.numel() for p in q_net.parameters())} parameters")
    
        ckpt_path = Path(args.eval)
        if ckpt_path.exists():
            loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
            visualize_env_version = checkpoint_env_version(loaded)
            print(f"Using environment version {visualize_env_version}")
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
