"""
Measures which training settings are fastest on this machine, to tune platform-specific
defaults (written for Apple Silicon / MPS, works anywhere). Run from src/:

    uv run --project .. python benchmark_platform.py            # ~20-30 minutes
    uv run --project .. python benchmark_platform.py --quick    # a few minutes, rougher

Nothing else heavy should be running. Prints tables and saves everything to
benchmark_results_<host>.json (send that file back).

Sections:
 1. CPU inference (what actors and evaluation workers run): threads, memory layout, batch size
 2. Actor collection alone: actor count x threads per actor
 3. The update step on the training device: eager, channels-last, torch.compile (with a check
    that compiled results match), and the compact replay storage
 4. Short real training runs: single-process vs actors with the defaults and the best setting
 5. The test suite (includes the MPS-only tests when MPS is available)
"""

from argparse import ArgumentParser
import copy
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import models
from utils import NUM_ACTIONS, OBS_SHAPE, select_device

SRC = Path(__file__).resolve().parent
CHECKPOINT = SRC / "best_checkpoints/urlriljg/q_net-iter0619000.ckpt.pt"
results: dict = {}


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def system_info(device: torch.device) -> dict:
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "logical_cpus": os.cpu_count(),
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if sys.platform == "darwin":
        info["cpu"] = sysctl("machdep.cpu.brand_string")
        info["performance_cores"] = sysctl("hw.perflevel0.physicalcpu")
        info["efficiency_cores"] = sysctl("hw.perflevel1.physicalcpu")
        info["memory_gb"] = round(int(sysctl("hw.memsize") or 0) / 2**30, 1)
    else:
        info["cpu"] = platform.processor()
    return info


def trained_qnet(device) -> torch.nn.Module:
    loaded = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS)
    q_net.load_state_dict(loaded["state_dict"])
    return q_net.to(device).eval()


# ── 1. CPU inference ─────────────────────────────────────────────────────────────


def bench_cpu_inference(quick: bool) -> list[dict]:
    section("1. CPU inference (per call; what actors and evaluation workers run)")
    base = trained_qnet("cpu")
    channels_last = copy.deepcopy(base).to(memory_format=torch.channels_last)
    rows = []
    threads_list = (1, 2) if quick else (1, 2, 3, 4)
    prev_threads = torch.get_num_threads()
    try:
        for threads in threads_list:
            torch.set_num_threads(threads)
            for batch in (1, 8):
                x = torch.rand(batch, *OBS_SHAPE)
                with torch.inference_mode():
                    ref = base(x)
                    for name, net in (("contiguous", base), ("channels_last", channels_last)):
                        for _ in range(5):
                            net(x)
                        n = (60 if quick else 300) // batch
                        t = time.perf_counter()
                        for _ in range(n):
                            net(x)
                        ms = 1e3 * (time.perf_counter() - t) / n
                        diff = (net(x) - ref).abs().max().item()
                        rows.append(dict(threads=threads, batch=batch, layout=name, ms=ms, max_diff=diff))
                        print(f"  threads={threads} batch={batch:<2d} {name:13s} {ms:7.3f} ms   max diff vs contiguous {diff:.1e}", flush=True)
    finally:
        torch.set_num_threads(prev_threads)
    return rows


# ── 2. Actor collection ──────────────────────────────────────────────────────────


def bench_actors(device, quick: bool) -> list[dict]:
    import parallel_training

    section("2. Actor collection alone (32 games, 4 steps per iteration)")
    cpus = os.cpu_count() or 4
    specs = [(2, 2), (4, 1), (4, 2)] if quick else [
        (a, t) for a, t in ((2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (6, 1), (6, 2), (8, 1))
        if a * t <= cpus
    ]
    q_net = trained_qnet(device)
    rows = []
    for actors, threads in specs:
        try:
            buffer = parallel_training.ActorReplayBuffer(
                30_000, q_net, "QNetV2", 0.5, 32, actors, 4, device, threads_per_actor=threads)
            try:
                for _ in range(20 if quick else 50):
                    buffer.start_collection(4, 0.5)
                    buffer.finish_collection()
                n = 60 if quick else 300
                t = time.perf_counter()
                for _ in range(n):
                    buffer.start_collection(4, 0.5)
                    buffer.finish_collection()
                ms = 1e3 * (time.perf_counter() - t) / n
            finally:
                buffer.close()
            rows.append(dict(actors=actors, threads=threads, ms_per_iteration=ms, max_it_per_s=1e3 / ms))
            print(f"  {actors} actors x {threads} threads: {ms:6.1f} ms per iteration -> at most {1e3 / ms:5.1f} it/s", flush=True)
        except Exception as e:
            rows.append(dict(actors=actors, threads=threads, error=repr(e)))
            print(f"  {actors} actors x {threads} threads: FAILED {e!r}", flush=True)
    return rows


# ── 3. Update step and replay storage on the training device ────────────────────


def real_observations(games: int, steps: int) -> torch.Tensor:
    from pacbot_rs import PacmanGym, PacmanGymConfiguration as C

    starts = [C({}), C({"random_start": True}), C({"random_ticks": True})]
    rng = random.Random(0)
    out = []
    for game in range(games):
        env = PacmanGym(starts[game % len(starts)])
        for step in range(steps):
            out.append(env.obs_numpy(C({"obs_ignore_super_pellets": game % 2 == 0})))
            if step % 5 == 0:
                action = rng.choice([a for a, ok in enumerate(env.action_mask()) if ok])
            _, done = env.step(action)
            if done:
                break
    return torch.from_numpy(np.stack(out))


def filled_storage(device, obs: torch.Tensor, capacity: int):
    from replay_storage import ReplayStorage

    rng = random.Random(1)
    envs = 32
    storage = ReplayStorage(capacity, envs, device)
    storage.set_initial_obs(obs[:envs].to(device))
    while len(storage) < capacity:
        rows = [rng.randrange(len(obs)) for _ in range(envs)]
        dones = [rng.random() < 0.01 for _ in range(envs)]
        storage.add_step(
            obs[rows].to(device),
            [rng.randrange(NUM_ACTIONS) for _ in range(envs)],
            [rng.choice([0, 10, 50, -2, -20]) for _ in range(envs)],
            dones,
            [[False] * NUM_ACTIONS if d else [True, True, rng.random() < 0.5, True, False] for d in dones],
        )
    return storage


def time_steps(fn, device, n: int) -> float:
    for _ in range(5):
        fn()
    sync(device)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync(device)
    return 1e3 * (time.perf_counter() - t) / n


def bench_update(device, quick: bool) -> dict:
    from dqn_update import dqn_update

    section(f"3. Update step on {device} (batch 512) and replay storage")
    out: dict = {}
    obs = real_observations(games=4 if quick else 12, steps=200)
    storage = filled_storage(device, obs, 4096)
    sync(device)
    mismatches = storage.codec.mismatches()
    out["storage_decode_mismatches"] = mismatches
    print(f"  compact storage: {mismatches} values didn't decode exactly (must be 0)", flush=True)

    idx = storage.sample_indices(512)
    out["storage_gather_ms"] = time_steps(lambda: storage.gather(idx), device, 20 if quick else 100)
    flat = torch.rand(4096 + 32, *OBS_SHAPE, device=device)  # as many rows as the storage
    out["float32_gather_ms"] = time_steps(lambda: (flat[idx.obs_row], flat[idx.next_row]), device, 20 if quick else 100)
    print(f"  batch gather: compact {out['storage_gather_ms']:.2f} ms vs plain float32 {out['float32_gather_ms']:.2f} ms", flush=True)

    kw = dict(reward_scale=0.02, discount_factor=0.99, grad_clip_norm=10000.0)

    def make(layout_cl: bool, compile_mode: str | None):
        torch.manual_seed(0)
        q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS).to(device)
        if layout_cl:
            q_net = q_net.to(memory_format=torch.channels_last)
        target = copy.deepcopy(q_net)
        opt = torch.optim.Adam(q_net.parameters(), lr=1e-4)
        if compile_mode:
            q_call, t_call = torch.compile(q_net, mode=compile_mode), torch.compile(target, mode=compile_mode)
        else:
            q_call, t_call = q_net, target
        return q_net, q_call, t_call, opt

    def step(parts, batch):
        q_net, q_call, t_call, opt = parts
        # dqn_update calls the networks; compiled wrappers share the original parameters
        return dqn_update(q_call, t_call, opt, batch, **kw)

    variants = [("eager", False, None), ("eager channels_last", True, None)]
    variants += [("torch.compile", False, "default")] if quick else [
        ("torch.compile", False, "default"), ("torch.compile max-autotune", False, "max-autotune-no-cudagraphs")]
    batch = storage.gather(idx)
    reference = None
    out["variants"] = []
    for name, cl, mode in variants:
        row = dict(variant=name)
        try:
            parts = make(cl, mode)
            t0 = time.perf_counter()
            first = step(parts, batch)
            sync(device)
            row["first_call_s"] = time.perf_counter() - t0
            loss = first[0].item()
            if reference is None:
                reference = loss
            row["first_loss"] = loss
            row["loss_rel_diff_vs_eager"] = abs(loss - reference) / abs(reference)
            row["ms_per_step"] = time_steps(lambda: step(parts, batch), device, 20 if quick else 100)
            print(f"  {name:28s} {row['ms_per_step']:7.2f} ms/step   (first call {row['first_call_s']:.1f} s, "
                  f"loss rel. diff vs eager {row['loss_rel_diff_vs_eager']:.1e})", flush=True)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  {name:28s} FAILED: {row['error']}", flush=True)
        out["variants"].append(row)
    return out


# ── 4. Short real training runs ──────────────────────────────────────────────────


def bench_training(device, quick: bool, actor_rows: list[dict]) -> list[dict]:
    section("4. Short training runs (fine-tuning the tracked checkpoint, no W&B)")
    iters = 600 if quick else 3000
    cpus = os.cpu_count() or 4
    configs = [("single-process", ["--num-actors", "0", "--eval-workers", "0"]), ("defaults", [])]
    ok = [r for r in actor_rows if "error" not in r]
    if ok:
        best = min(ok, key=lambda r: r["ms_per_iteration"])
        configs.append((f"best actors ({best['actors']}x{best['threads']})",
                        ["--num-actors", str(best["actors"]), "--actor-threads", str(best["threads"])]))
    rows = []
    for name, extra in configs:
        n = 300 if name == "single-process" and not quick else (200 if quick else iters)
        with tempfile.TemporaryDirectory() as ckpt_dir:
            cmd = [sys.executable, "-m", "algorithms.train_dqn", "--no-wandb", "--no-eval",
                   "--finetune", str(CHECKPOINT), "--num_iters", str(619_000 + n),
                   "--checkpoint-dir", ckpt_dir, *extra]
            t = time.perf_counter()
            proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
            wall = time.perf_counter() - t
        text = (proc.stdout + proc.stderr).replace("\r", "\n")
        rates = re.findall(r"(\d+(?:\.\d+)?)it/s", text) or []
        slow = re.findall(r"(\d+(?:\.\d+)?)s/it", text) or []
        rate = float(rates[-1]) if rates else (1 / float(slow[-1]) if slow else None)
        setup = [line for line in text.splitlines() if line.startswith(("Actors:", "Updates:", "Using device"))]
        row = dict(config=name, args=extra, iterations=n, it_per_s=rate, wall_s=wall,
                   exit_code=proc.returncode, setup=setup)
        if proc.returncode != 0:
            row["error_tail"] = text[-1500:]
        rows.append(row)
        shown = f"{rate:6.1f} it/s" if rate else "no rate"
        print(f"  {name:28s} {shown}  (exit {proc.returncode}; {'; '.join(setup)})", flush=True)
    return rows


# ── 5. Tests ─────────────────────────────────────────────────────────────────────


def run_tests() -> dict:
    section("5. Test suite")
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=SRC.parent, env=env, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    summary = [line for line in text.splitlines() if line.startswith(("Ran ", "OK", "FAILED"))]
    print("  " + " | ".join(summary), flush=True)
    failures = [line for line in text.splitlines() if line.startswith(("FAIL:", "ERROR:"))]
    for line in failures:
        print("  " + line, flush=True)
    return dict(exit_code=proc.returncode, summary=summary, failures=failures,
                tail=text[-3000:] if proc.returncode else None)


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="Fewer settings and repetitions")
    parser.add_argument("--device", default=None, help="Training device (default: picked as in training)")
    parser.add_argument("--skip", default="", help="Comma-separated sections to skip, e.g. 4,5")
    parser.add_argument("--out", default=None, help="Results file (default: benchmark_results_<host>.json)")
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    device = select_device(args.device)
    out_path = Path(args.out or f"benchmark_results_{socket.gethostname().split('.')[0]}.json")

    results["system"] = system_info(device)
    results["quick"] = args.quick
    print(json.dumps(results["system"], indent=2))
    steps = [
        ("1", "cpu_inference", lambda: bench_cpu_inference(args.quick)),
        ("2", "actors", lambda: bench_actors(device, args.quick)),
        ("3", "update", lambda: bench_update(device, args.quick)),
        ("4", "training", lambda: bench_training(device, args.quick, results.get("actors") or [])),
        ("5", "tests", run_tests),
    ]
    for number, key, fn in steps:
        if number in skip:
            continue
        try:
            results[key] = fn()
        except Exception:
            results[key] = {"error": traceback.format_exc()}
            print(f"  section {number} failed:\n{results[key]['error']}", flush=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {out_path.resolve()}")


if __name__ == "__main__":
    main()
