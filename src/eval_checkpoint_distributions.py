"""Evaluate arbitrary DQN checkpoints and compare their game score distributions.

Run from the RL repository root, for example:

    uv run --project . python src/eval_checkpoint_distributions.py \
        --checkpoint checkpoints_downloaded/run_urlriljg/q_net-iter0619000.ckpt.pt \
        --games 100 --device cpu

Each --checkpoint may name a checkpoint file or a directory (searched
recursively for q_net-*.ckpt.pt). Results are written using --output-prefix.
Completed checkpoints are skipped on rerun unless --overwrite is specified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
SUMMARY_FIELDS = [
    "label", "checkpoint", "n_games", "mean_score", "mean_ci95_low", "mean_ci95_high",
    "median_score", "median_ci95_low", "median_ci95_high", "std_score", "sem_score",
    "min_score", "p01_score", "p05_score", "p10_score", "p25_score", "p50_score",
    "p75_score", "p90_score", "p95_score", "p99_score", "max_score",
    "mean_pellets_eaten", "mean_steps", "board_cleared_rate", "error",
]
GAME_FIELDS = ["label", "checkpoint", "game", "score", "pellets_eaten", "steps", "board_cleared"]


def expand_checkpoints(inputs: list[str]) -> list[tuple[str, str]]:
    """Expand checkpoint files and directories into unique (label, path) pairs."""
    found: dict[str, str] = {}
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if path.is_dir():
            candidates = sorted(path.rglob("q_net-*.ckpt.pt"))
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"Checkpoint path does not exist: {raw}")
        for checkpoint in candidates:
            if not checkpoint.is_file():
                continue
            full_path = str(checkpoint.resolve())
            name = checkpoint.name.removeprefix("q_net-").removesuffix(".ckpt.pt")
            found[full_path] = f"{checkpoint.parent.name} / {name}"
    if not found:
        raise ValueError("No checkpoint files found (expected q_net-*.ckpt.pt).")
    return [(label, path) for path, label in found.items()]


def evaluate_checkpoint(task: tuple[str, str, int, str, int]) -> dict:
    """Process worker: evaluate a checkpoint and return raw games plus statistics."""
    label, checkpoint, games, device_name, bootstrap_samples = task
    os.chdir(SCRIPT_DIR)
    sys.path.insert(0, str(SCRIPT_DIR))

    import torch
    from eval_best import evaluate_episode, load_checkpoint
    from utils import select_device

    device = select_device(None if device_name == "auto" else device_name)
    q_net, _, _, _ = load_checkpoint(checkpoint, device)
    episodes = [evaluate_episode(q_net, device) for _ in range(games)]
    scores = np.asarray([episode["score"] for episode in episodes], dtype=np.float64)

    # Percentile bootstrap intervals retain the empirical, potentially
    # multimodal score distribution without assuming normally distributed games.
    rng = np.random.default_rng(bootstrap_samples + sum(checkpoint.encode("utf-8")))
    resamples = rng.choice(scores, size=(bootstrap_samples, games), replace=True)
    bootstrap_means = resamples.mean(axis=1)
    bootstrap_medians = np.median(resamples, axis=1)

    mean_score = float(scores.mean())
    std_score = float(scores.std(ddof=1)) if games > 1 else 0.0
    summary = {
        "label": label,
        "checkpoint": checkpoint,
        "n_games": games,
        "mean_score": mean_score,
        "mean_ci95_low": float(np.percentile(bootstrap_means, 2.5)),
        "mean_ci95_high": float(np.percentile(bootstrap_means, 97.5)),
        "median_score": float(np.median(scores)),
        "median_ci95_low": float(np.percentile(bootstrap_medians, 2.5)),
        "median_ci95_high": float(np.percentile(bootstrap_medians, 97.5)),
        "std_score": std_score,
        "sem_score": std_score / math.sqrt(games),
        "min_score": float(scores.min()),
        "p01_score": float(np.percentile(scores, 1)),
        "p05_score": float(np.percentile(scores, 5)),
        "p10_score": float(np.percentile(scores, 10)),
        "p25_score": float(np.percentile(scores, 25)),
        "p50_score": float(np.percentile(scores, 50)),
        "p75_score": float(np.percentile(scores, 75)),
        "p90_score": float(np.percentile(scores, 90)),
        "p95_score": float(np.percentile(scores, 95)),
        "p99_score": float(np.percentile(scores, 99)),
        "max_score": float(scores.max()),
        "mean_pellets_eaten": float(np.mean([e["pellets_eaten"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "board_cleared_rate": float(np.mean([e["board_cleared"] for e in episodes])),
        "error": "",
    }
    raw_games = [
        {
            "label": label,
            "checkpoint": checkpoint,
            "game": index,
            "score": episode["score"],
            "pellets_eaten": episode["pellets_eaten"],
            "steps": episode["steps"],
            "board_cleared": episode["board_cleared"],
        }
        for index, episode in enumerate(episodes, start=1)
    ]
    return {"summary": summary, "games": raw_games}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_existing(summary_path: Path, games_path: Path) -> tuple[list[dict], list[dict]]:
    if not summary_path.exists() or not games_path.exists():
        return [], []
    with summary_path.open(newline="") as file:
        summaries = list(csv.DictReader(file))
    with games_path.open(newline="") as file:
        games = list(csv.DictReader(file))
    complete = {row["checkpoint"] for row in summaries if not row.get("error")}
    summaries = [row for row in summaries if row["checkpoint"] in complete]
    games = [row for row in games if row["checkpoint"] in complete]
    return summaries, games


def save_plots(prefix: Path, summaries: list[dict], games: list[dict]) -> tuple[Path, Path, Path]:
    by_checkpoint: dict[str, list[float]] = {}
    for row in games:
        by_checkpoint.setdefault(row["checkpoint"], []).append(float(row["score"]))
    summaries = [row for row in summaries if row["checkpoint"] in by_checkpoint]
    summaries.sort(key=lambda row: float(row["mean_score"]), reverse=True)
    labels = [row["label"] for row in summaries]
    distributions = [by_checkpoint[row["checkpoint"]] for row in summaries]
    x = np.arange(1, len(summaries) + 1)

    distribution_path = prefix.with_name(prefix.name + "_distributions.png")
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 7))
    violin = ax.violinplot(distributions, positions=x, widths=0.78, showextrema=False)
    palette = plt.get_cmap("tab10")
    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(palette(index % 10))
        body.set_edgecolor("#333333")
        body.set_alpha(0.30)
    ax.boxplot(distributions, positions=x, widths=0.22, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "white", "alpha": 0.75}, medianprops={"color": "black"})
    for position, (scores, summary) in enumerate(zip(distributions, summaries), start=1):
        jitter = np.random.default_rng(position).uniform(-0.12, 0.12, size=len(scores))
        ax.scatter(position + jitter, scores, s=14, alpha=0.45, color=palette((position - 1) % 10),
                   edgecolors="none", zorder=3)
        low = float(summary["mean_ci95_low"])
        high = float(summary["mean_ci95_high"])
        mean = float(summary["mean_score"])
        ax.errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="black",
                    capsize=4, markersize=5, zorder=5)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Game score")
    ax.set_title("Checkpoint score distributions (diamond: mean, bars: bootstrap 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)

    ecdf_path = prefix.with_name(prefix.name + "_ecdf.png")
    fig, ax = plt.subplots(figsize=(11, 7))
    for index, (label, scores) in enumerate(zip(labels, distributions)):
        ordered = np.sort(scores)
        probabilities = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.step(ordered, probabilities, where="post", label=label, color=palette(index % 10), linewidth=2)
    ax.set_xlabel("Game score")
    ax.set_ylabel("Fraction of games at or below score")
    ax.set_title("Empirical cumulative score distributions")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(ecdf_path, dpi=180)
    plt.close(fig)

    means_path = prefix.with_name(prefix.name + "_means_ci.png")
    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.8)))
    y = np.arange(len(summaries))
    means = np.asarray([float(row["mean_score"]) for row in summaries])
    lows = np.asarray([float(row["mean_ci95_low"]) for row in summaries])
    highs = np.asarray([float(row["mean_ci95_high"]) for row in summaries])
    ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", capsize=4, color="#235789")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean game score (bootstrap 95% CI)")
    ax.set_title("Mean score comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(means_path, dpi=180)
    plt.close(fig)
    return distribution_path, ecdf_path, means_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Checkpoint file or directory; repeat for multiple inputs.")
    parser.add_argument("--games", type=int, default=100, help="Games per checkpoint (default: 100).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Inference device; auto prefers MPS, then CUDA, then CPU.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel checkpoint workers (default: 4).")
    parser.add_argument("--bootstrap-samples", type=int, default=20000,
                        help="Bootstrap resamples for confidence intervals (default: 20000).")
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("checkpoint_eval_results/evaluation_100games"),
                        help="Prefix for CSV and PNG outputs (default: checkpoint_eval_results/evaluation_100games).")
    parser.add_argument("--overwrite", action="store_true", help="Rerun checkpoints already in the output CSV.")
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1 or args.bootstrap_samples < 1000:
        parser.error("games and workers must be positive; use at least 1000 bootstrap samples")

    checkpoints = expand_checkpoints(args.checkpoint)
    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    if args.device == "mps" and not __import__("torch").backends.mps.is_available():
        parser.error("MPS was requested but is unavailable.")

    prefix = args.output_prefix.expanduser()
    if not prefix.is_absolute():
        prefix = (Path.cwd() / prefix).resolve()
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    games_path = prefix.with_name(prefix.name + "_games.csv")
    summaries, all_games = load_existing(summary_path, games_path)
    requested_paths = {path for _, path in checkpoints}
    summaries = [row for row in summaries
                 if row["checkpoint"] in requested_paths and int(row["n_games"]) == args.games]
    completed_paths = {row["checkpoint"] for row in summaries}
    all_games = [row for row in all_games if row["checkpoint"] in completed_paths]
    if args.overwrite:
        summaries, all_games = [], []
    done = {row["checkpoint"] for row in summaries}
    pending = [(label, path, args.games, args.device, args.bootstrap_samples)
               for label, path in checkpoints if path not in done]
    write_csv(summary_path, SUMMARY_FIELDS, summaries)
    write_csv(games_path, GAME_FIELDS, all_games)
    print(f"Evaluating {len(pending)} checkpoint(s), {args.games} games each, on {args.device}.", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(evaluate_checkpoint, task): task for task in pending}
        for index, future in enumerate(as_completed(future_map), start=1):
            label, checkpoint, *_ = future_map[future]
            try:
                result = future.result()
                summaries.append(result["summary"])
                all_games.extend(result["games"])
                print(f"Completed {label}: mean {result['summary']['mean_score']:.1f} ", end="", flush=True)
            except Exception as error:
                summaries.append({"label": label, "checkpoint": checkpoint, "n_games": args.games,
                                  "error": repr(error)})
                print(f"Failed {label}: {error}", end="", flush=True)
            write_csv(summary_path, SUMMARY_FIELDS, summaries)
            write_csv(games_path, GAME_FIELDS, all_games)
            print(f"({index}/{len(pending)}, {((time.perf_counter() - started) / 60):.1f} min)", flush=True)

    valid = [row for row in summaries if not row.get("error") and row.get("n_games")]
    if valid:
        plots = save_plots(prefix, valid, all_games)
        print("\nLeaderboard:")
        for rank, row in enumerate(sorted(valid, key=lambda r: float(r["mean_score"]), reverse=True), start=1):
            print(f"{rank:>2}. {row['label']}: {float(row['mean_score']):.1f} "
                  f"(95% bootstrap CI {float(row['mean_ci95_low']):.1f}–{float(row['mean_ci95_high']):.1f}; "
                  f"median {float(row['median_score']):.1f}; P5/P95 "
                  f"{float(row['p05_score']):.1f}/{float(row['p95_score']):.1f})")
        print(f"\nSummary CSV: {summary_path}\nPer-game CSV: {games_path}")
        print("Plots:\n" + "\n".join(str(path) for path in plots))
    print(f"Elapsed: {(time.perf_counter() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
