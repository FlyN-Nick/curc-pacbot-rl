from argparse import ArgumentParser
from contextlib import nullcontext
import copy
import math
import itertools
import json
import statistics
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
from replay_storage import ReplayStorage
from dqn_update import EagerUpdate, GraphedUpdate, set_capturable
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
        "--no-cuda-graph", action="store_true",
        help="On CUDA, run each update eagerly instead of replaying it as a CUDA graph",
    )
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
                if "best_eval_games" in loaded:
                    # Older checkpoints kept the best single game, which isn't comparable
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


def candidate_metrics(results: list[tuple]) -> dict:
    """Metrics to log for a candidate checkpoint's evaluation games."""
    scores = [r[0] for r in results]
    return dict(
        candidate_mean_score=statistics.fmean(scores),
        candidate_median_score=statistics.median(scores),
        candidate_cleared_rate=statistics.fmean(int(r[2]) for r in results),
        candidate_pellets_end=statistics.fmean(r[4] for r in results),
        candidate_games=len(results),
    )


def exploration_epsilon(iter_num: int) -> float:
    """Epsilon falls linearly from initial to final over epsilon_decay_fraction of training, then holds."""
    decay_iters = max(1.0, (wandb.config.num_iters - 1) * wandb.config.epsilon_decay_fraction)
    return lerp(wandb.config.initial_epsilon, wandb.config.final_epsilon, min(1.0, iter_num / decay_iters))


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

    def add(self, host_metrics: dict, values: torch.Tensor):
        """Queues this iteration's metrics (values from dqn_update) and logs the previous ones."""
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


def check_replay_storage(replay_buffer) -> None:
    """Stops training if any stored observation didn't decode back exactly (waits for the GPU)."""
    mismatches = replay_buffer.storage.codec.mismatches()
    if mismatches:
        raise RuntimeError(
            f"{mismatches} stored observation values didn't decode back exactly; the "
            "observation format may have changed (see replay_storage.ObsCodec). Stopping."
        )


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
                storage=ReplayStorage(
                    wandb.config.replay_buffer_size, wandb.config.num_parallel_envs, device
                ),
            )
        _train(replay_buffer, evaluator, actors)
    finally:
        for resource in (replay_buffer, evaluator):
            if hasattr(resource, "close"):
                resource.close()


def _train(replay_buffer, evaluator, actors: bool):
    replay_buffer.fill()
    replay_buffer.policy.epsilon = float(resume_epsilon) if resume_epsilon is not None else wandb.config.initial_epsilon

    # Initialize the optimizer. Replaying the update as a CUDA graph needs Adam's capturable
    # form; checkpoints load either way.
    graphed = device.type == "cuda" and not args.no_cuda_graph
    optimizer = torch.optim.Adam(q_net.parameters(), lr=wandb.config.learning_rate, capturable=graphed)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        set_capturable(optimizer, graphed)
        print("Restored optimizer state from checkpoint")

    best_eval_score = resume_best_eval_score

    def build_metadata(iter_num: int) -> dict:
        metadata = {
            "state_dict": q_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter_num": iter_num,
            "epsilon": replay_buffer.policy.epsilon,
            "best_eval_score": best_eval_score,
            "best_eval_games": wandb.config.best_eval_games,
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
    q_net.train()
    update = (GraphedUpdate if graphed else EagerUpdate)(
        q_net, target_q_net, optimizer, replay_buffer.storage, wandb.config.batch_size,
        reward_scale=wandb.config.reward_scale,
        discount_factor=wandb.config.discount_factor,
        grad_clip_norm=wandb.config.grad_clip_norm,
        bf16=args.bf16,
    )
    print(f"Updates: {'CUDA graph' if graphed else 'eager'}")

    deferred_metrics = DeferredMetrics(device, wandb.config.reward_scale)

    def is_candidate_iter(iter_num: int) -> bool:
        return iter_num > 0 and (
            iter_num % wandb.config.best_eval_steps == 0 or iter_num == wandb.config.num_iters - 1
        )

    def consider_candidate(iter_num: int, results: list[tuple], snapshot: dict, metrics: dict) -> None:
        nonlocal best_eval_score
        summary = candidate_metrics(results)
        metrics.update(summary, candidate_iter=iter_num)
        if summary["candidate_mean_score"] > best_eval_score:
            best_eval_score = summary["candidate_mean_score"]
            snapshot["best_eval_score"] = best_eval_score
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            atomic_torch_save(snapshot, checkpoint_dir / "q_net-best.ckpt.pt")
            print(f"New best checkpoint: iter {iter_num}, mean score {best_eval_score:.0f} "
                  f"over {len(results)} games")
        metrics["best_eval_mean_score"] = best_eval_score
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
                # Update the target network (in place: a CUDA graph keeps using its tensors)
                target_q_net.load_state_dict(q_net.state_dict())

        if actors:
            # The actors play this iteration's experience while this process updates the
            # network, with the weights from before the update.
            replay_buffer.start_collection(
                wandb.config.experience_steps,
                exploration_epsilon(iter_num),
            )

        with time_block("Update the network"):
            # Sample a batch (indices drawn on the device; see replay_storage) and update the
            # network on it (see dqn_update)
            update_metrics = update.run(replay_buffer.storage.sample_indices(wandb.config.batch_size))

        with torch.no_grad():
            # Log metrics. The GPU values are logged by DeferredMetrics a step later.
            metrics = {
                "exploration_epsilon": replay_buffer.policy.epsilon,
            }
            if iter_num % wandb.config.evaluate_steps == 0:
                # One game, for the learning curve
                if evaluator is None:
                    with time_block("Evaluate the current agent"):
                        metrics.update(eval_metrics(evaluate_episode()))
                else:
                    # Played in the background; if every worker is busy, skipped
                    evaluator.submit(iter_num, q_net, lambda: None)
            if is_candidate_iter(iter_num):
                # Many games of this checkpoint; the best mean score is kept as q_net-best
                if evaluator is None:
                    with time_block("Evaluate a candidate"):
                        results = [evaluate_episode() for _ in range(wandb.config.best_eval_games)]
                    consider_candidate(iter_num, results, build_metadata(iter_num), metrics)
                else:
                    evaluator.submit_candidate(
                        iter_num, q_net, wandb.config.best_eval_games, cpu_copy(build_metadata(iter_num))
                    )
            if evaluator is not None:
                for eval_iter, result, _ in evaluator.poll():
                    metrics.update(eval_metrics(result), eval_iter=eval_iter)
                for candidate_iter, results, snapshot in evaluator.poll_candidates():
                    consider_candidate(candidate_iter, results, snapshot, metrics)
                metrics["skipped_evals"] = evaluator.skipped
            deferred_metrics.add(metrics, update_metrics)

        if iter_num % 500 == 0:
            # Check this update, and that stored observations decode exactly, before saving
            deferred_metrics.flush()
            check_replay_storage(replay_buffer)
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
        replay_buffer.policy.epsilon = exploration_epsilon(iter_num)

        # Collect experience.
        with time_block("Collect experience"):
            if actors:
                replay_buffer.finish_collection()
            else:
                for _ in range(wandb.config.experience_steps):
                    replay_buffer.generate_experience_step()

    deferred_metrics.flush()
    if evaluator is not None and not _exit_requested:
        # Finish the candidates still being played (the last checkpoint's, at least)
        while evaluator.candidates_pending():
            metrics = {}
            for candidate_iter, results, snapshot in evaluator.poll_candidates(timeout=1.0):
                consider_candidate(candidate_iter, results, snapshot, metrics)
            if metrics:
                wandb.log(metrics)
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
