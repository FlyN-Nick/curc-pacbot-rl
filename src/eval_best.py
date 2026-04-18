"""
Thorough comparison of the best-known checkpoints across all runs.
Checkpoints are hard-coded; no directory scanning.

Usage (from src/):
    python eval_best.py [--n-games N] [--max-workers W] [--device DEVICE]
    python eval_best.py --csv eval_best_results.csv   # re-plot from saved CSV
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pacbot_rs import PacmanGym
import models
from policies import MaxQPolicy
from utils import OBS_SHAPE, NUM_ACTIONS, DETERMINISTIC_START_CONFIGURATION, select_device, step_env_until_done
import argparse
from tqdm import tqdm

# ── Hard-coded best checkpoints ────────────────────────────────────────────────
# Each entry: (label, checkpoint_path_relative_to_src/)
BEST_CHECKPOINTS = [
    # winnie_ec2
    ("winnie_ec2 @163500",  "checkpoints_all/winnie_ec2/checkpoints/q_net-iter0163500.ckpt.pt"),
    ("winnie_ec2 @164000",  "checkpoints_all/winnie_ec2/checkpoints/q_net-iter0164000.ckpt.pt"),
    ("winnie_ec2 @165500",  "checkpoints_all/winnie_ec2/checkpoints/q_net-iter0165500.ckpt.pt"),
    ("winnie_ec2 @702500",  "checkpoints_all/winnie_ec2/checkpoints/q_net-iter0702500.ckpt.pt"),
    ("winnie_ec2 @820500",  "checkpoints_all/winnie_ec2/checkpoints/q_net-iter0820500.ckpt.pt"),
    # nick_ec2
    ("nick_ec2 @218500",    "checkpoints_all/nick_ec2/checkpoints/q_net-iter0218500.ckpt.pt"),
    ("nick_ec2 @399000",    "checkpoints_all/nick_ec2/checkpoints/q_net-iter0399000.ckpt.pt"),
    # run1
    ("run1 @582500",        "checkpoints_all/run1/checkpoints/q_net-iter0582500.ckpt.pt"),
    ("run1 @619000",        "checkpoints_all/run1/checkpoints/q_net-iter0619000.ckpt.pt"),
    ("run1 @927000",        "checkpoints_all/run1/checkpoints/q_net-iter0927000.ckpt.pt"),
    ("run1 @1048000",       "checkpoints_all/run1/checkpoints/q_net-iter1048000.ckpt.pt"),
    ("run1 @1150000",       "checkpoints_all/run1/checkpoints/q_net-iter1150000.ckpt.pt"),
    # nick_local_fourth (checkpoints live directly in the folder, no checkpoints/ subdir)
    ("nick_local_4th @1736500", "checkpoints_all/nick_local_fourth/q_net-iter1736500.ckpt.pt"),
]

# Color per run family
RUN_COLORS = {
    "winnie_ec2":    "#1f77b4",
    "nick_ec2":      "#ff7f0e",
    "run1":          "#2ca02c",
    "nick_local_4th":"#d62728",
}

def label_color(label: str) -> str:
    for prefix, color in RUN_COLORS.items():
        if label.startswith(prefix):
            return color
    return "#7f7f7f"

# ── 1. Load model ──────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str, device: torch.device):
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_class = getattr(models, loaded['config']['model'])
    q_net = model_class(OBS_SHAPE, NUM_ACTIONS).to(device)
    q_net.load_state_dict(loaded['state_dict'])
    q_net.eval()
    return q_net, loaded['iter_num'], loaded['epsilon'], loaded['config']

# ── 2. Evaluate one episode ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_episode(q_net, device: torch.device, max_steps: int = 1000):
    gym = PacmanGym(DETERMINISTIC_START_CONFIGURATION)
    pellets_start = gym.remaining_pellets()
    policy = MaxQPolicy(q_net)
    done, step_num = step_env_until_done(gym, policy, device, max_steps=max_steps)
    is_board_cleared = done and gym.lives() == 3
    pellets_end = 0 if is_board_cleared else gym.remaining_pellets()
    return {
        'score': gym.score(),
        'steps': step_num,
        'board_cleared': is_board_cleared,
        'pellets_eaten': pellets_start - pellets_end,
    }

# ── 3. Worker (runs in a separate process) ─────────────────────────────────────
def worker(label: str, ckpt_path: str, n_games: int, preferred_device: str, progress_queue):
    device = select_device(preferred_device)
    q_net, iter_num, epsilon, config = load_checkpoint(ckpt_path, device)

    scores, pellets, steps_list, cleared = [], [], [], []
    for _ in range(n_games):
        r = evaluate_episode(q_net, device)
        scores.append(r['score'])
        pellets.append(r['pellets_eaten'])
        steps_list.append(r['steps'])
        cleared.append(r['board_cleared'])
        if progress_queue is not None:
            progress_queue.put(1)

    scores_arr = np.array(scores)
    # 95% CI: z-approximation (accurate for n >= 30; default n_games=50)
    ci95 = 1.96 * np.std(scores_arr, ddof=1) / np.sqrt(len(scores_arr)) if len(scores_arr) > 1 else 0.0

    return {
        'label': label,
        'ckpt_path': ckpt_path,
        'iter_num': iter_num,
        'epsilon': epsilon,
        'config': config,
        'n_games': n_games,
        'scores': scores,
        'avg_score': float(np.mean(scores_arr)),
        'median_score': float(np.median(scores_arr)),
        'max_score': float(np.max(scores_arr)),
        'min_score': float(np.min(scores_arr)),
        'score_std': float(np.std(scores_arr)),
        'score_p25': float(np.percentile(scores_arr, 25)),
        'score_p75': float(np.percentile(scores_arr, 75)),
        'score_p95': float(np.percentile(scores_arr, 95)),
        'score_ci95': float(ci95),
        'avg_pellets': float(np.mean(pellets)),
        'avg_steps': float(np.mean(steps_list)),
        'board_cleared_rate': float(np.mean(cleared)),
    }

# ── 4. Run all evaluations in parallel ────────────────────────────────────────
def evaluate_all(checkpoints: list, n_games: int, max_workers: int, device: str):
    device_list = [d.strip() for d in device.split(',')] if device else [None]
    actual_devices = [str(select_device(d)) for d in device_list]
    print(f"Using devices: {', '.join(actual_devices)}")
    print(f"Evaluating {len(checkpoints)} checkpoints × {n_games} games ({max_workers} workers)...")

    results = []
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for i, (label, ckpt_path) in enumerate(checkpoints):
                dev = device_list[i % len(device_list)]
                fut = executor.submit(worker, label, ckpt_path, n_games, dev, progress_queue)
                future_map[fut] = label

            with tqdm(total=len(checkpoints) * n_games, desc="Games played") as pbar:
                pending = list(future_map.keys())
                while pending:
                    done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                    while not progress_queue.empty():
                        try:
                            pbar.update(progress_queue.get_nowait())
                        except Exception:
                            break
                    for fut in done:
                        label = future_map[fut]
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            tqdm.write(f"  [{label}] FAILED: {e}")

    # Restore original order
    label_order = {label: i for i, (label, _) in enumerate(checkpoints)}
    results.sort(key=lambda r: label_order.get(r['label'], 999))
    return results

# ── 5. Print leaderboard ───────────────────────────────────────────────────────
def print_leaderboard(results: list):
    print("\n" + "=" * 100)
    print("LEADERBOARD — ranked by average score")
    print("=" * 100)
    header = f"{'Rank':<5} {'Label':<28} {'Avg Score':<11} {'±95%CI':<9} {'Median':<9} {'P95':<9} {'Max':<9} {'Clear%':<9} {'AvgPellets'}"
    print(header)
    print("-" * len(header))
    ranked = sorted(results, key=lambda r: r['avg_score'], reverse=True)
    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:<5} {r['label']:<28} "
            f"{r['avg_score']:<11.1f} {r['score_ci95']:<9.1f} "
            f"{r['median_score']:<9.1f} {r['score_p95']:<9.1f} "
            f"{r['max_score']:<9.1f} {r['board_cleared_rate']:<9.1%} "
            f"{r['avg_pellets']:.1f}"
        )

    print("\n" + "=" * 100)
    print("LEADERBOARD — ranked by board clear rate")
    print("=" * 100)
    ranked_clear = sorted(results, key=lambda r: r['board_cleared_rate'], reverse=True)
    for rank, r in enumerate(ranked_clear, 1):
        print(
            f"{rank:<5} {r['label']:<28} "
            f"Clear: {r['board_cleared_rate']:<9.1%} "
            f"Avg: {r['avg_score']:<11.1f} "
            f"Max: {r['max_score']:.1f}"
        )

    print("\n" + "=" * 100)
    print("LEADERBOARD — ranked by max score")
    print("=" * 100)
    ranked_max = sorted(results, key=lambda r: r['max_score'], reverse=True)
    for rank, r in enumerate(ranked_max, 1):
        print(
            f"{rank:<5} {r['label']:<28} "
            f"Max: {r['max_score']:<11.1f} "
            f"Avg: {r['avg_score']:.1f}"
        )

# ── 6. Print hyperparameter differences ────────────────────────────────────────
def print_hyperparam_diff(results: list):
    configs = {r['label']: r.get('config', {}) for r in results if 'config' in r}
    if not configs:
        return
    all_keys = sorted({k for cfg in configs.values() for k in cfg})
    differing = [k for k in all_keys if len({str(cfg.get(k)) for cfg in configs.values()}) > 1]

    print("\n=== Differing Hyperparameters ===")
    if not differing:
        print("All checkpoints share identical hyperparameters.")
        return

    col_w = 20
    print(f"{'Hyperparameter':<35}", end="")
    for label in configs:
        print(f"{label[:col_w]:<{col_w}}", end="")
    print()
    print("-" * (35 + col_w * len(configs)))
    for key in differing:
        print(f"{key:<35}", end="")
        for cfg in configs.values():
            print(f"{str(cfg.get(key, 'N/A'))[:col_w]:<{col_w}}", end="")
        print()

# ── 7. Plot ────────────────────────────────────────────────────────────────────
def plot_results(results: list, n_games: int, save_image: bool):
    labels = [r['label'] for r in results]
    colors = [label_color(l) for l in labels]
    x = np.arange(len(labels))
    bar_w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(max(16, len(labels) * 1.1), 12))
    fig.suptitle(f'Best Checkpoint Comparison  ({n_games} games each)', fontsize=15, fontweight='bold')

    # — Subplot 1: avg score + 95% CI error bars —
    ax = axes[0, 0]
    avgs = [r['avg_score'] for r in results]
    cis  = [r['score_ci95'] for r in results]
    bars = ax.bar(x, avgs, width=bar_w, color=colors, yerr=cis, capsize=4, error_kw={'linewidth': 1.2})
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_title('Average Score (±95% CI)')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.4)

    # — Subplot 2: score distribution box plots —
    ax = axes[0, 1]
    box_data = [r['scores'] for r in results]
    bp = ax.boxplot(box_data, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_title('Score Distribution (Box Plot)')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.4)

    # — Subplot 3: board clear rate —
    ax = axes[1, 0]
    clear_rates = [r['board_cleared_rate'] * 100 for r in results]
    ax.bar(x, clear_rates, width=bar_w, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_title('Board Clear Rate (%)')
    ax.set_ylabel('Clear Rate (%)')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.4)
    for xi, val in zip(x, clear_rates):
        ax.text(xi, val + 1, f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

    # — Subplot 4: avg pellets + p95 score —
    ax = axes[1, 1]
    p95s = [r['score_p95'] for r in results]
    bar_w2 = bar_w / 2
    ax.bar(x - bar_w2/2, [r['avg_pellets'] for r in results], width=bar_w2, color=colors, alpha=0.8, label='Avg pellets eaten')
    ax2 = ax.twinx()
    ax2.bar(x + bar_w2/2, p95s, width=bar_w2, color=colors, alpha=0.45, label='95th-pct score')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_title('Avg Pellets Eaten  &  95th-Pct Score')
    ax.set_ylabel('Pellets eaten', color='black')
    ax2.set_ylabel('95th-pct score', color='grey')
    ax.grid(axis='y', alpha=0.3)

    # Legend for run families
    legend_patches = [mpatches.Patch(color=c, label=run) for run, c in RUN_COLORS.items()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(RUN_COLORS), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_image:
        out = 'eval_best_results.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {out}")
    plt.show()

# ── 8. CSV export / import ─────────────────────────────────────────────────────
FIELDS = [
    'label', 'ckpt_path', 'iter_num', 'epsilon', 'n_games',
    'avg_score', 'median_score', 'max_score', 'min_score', 'score_std',
    'score_p25', 'score_p75', 'score_p95', 'score_ci95',
    'avg_pellets', 'avg_steps', 'board_cleared_rate',
    'scores',  # JSON list
]

def export_csv(results: list, filename: str = 'eval_best_results.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in FIELDS}
            row['scores'] = json.dumps(r.get('scores', []))
            writer.writerow(row)
    print(f"Saved results to {filename}")

def load_csv(filename: str):
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed = {
                'label':             row['label'],
                'ckpt_path':         row['ckpt_path'],
                'iter_num':          int(row['iter_num']),
                'epsilon':           float(row['epsilon']),
                'n_games':           int(row['n_games']),
                'avg_score':         float(row['avg_score']),
                'median_score':      float(row['median_score']),
                'max_score':         float(row['max_score']),
                'min_score':         float(row['min_score']),
                'score_std':         float(row['score_std']),
                'score_p25':         float(row['score_p25']),
                'score_p75':         float(row['score_p75']),
                'score_p95':         float(row['score_p95']),
                'score_ci95':        float(row['score_ci95']),
                'avg_pellets':       float(row['avg_pellets']),
                'avg_steps':         float(row['avg_steps']),
                'board_cleared_rate':float(row['board_cleared_rate']),
                'scores':            json.loads(row.get('scores', '[]')),
            }
            results.append(processed)
    n_games = results[0]['n_games'] if results else 0
    return results, n_games

# ── 9. Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thorough comparison of best checkpoints.")
    parser.add_argument("--csv", type=str, help="Load existing results CSV instead of evaluating.")
    parser.add_argument("--n-games", type=int, default=50,
                        help="Games per checkpoint (default: 50 for statistical reliability).")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel worker processes (default: 4).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device(s), comma-separated (e.g. 'cpu', 'mps,cpu').")
    parser.add_argument("--no-save-image", action="store_false", dest="save_image", default=True)
    parser.add_argument("--no-save-csv",   action="store_false", dest="save_csv",   default=True)
    args = parser.parse_args()

    if args.csv:
        print(f"Loading results from {args.csv} ...")
        results, n_games = load_csv(args.csv)
    else:
        # Verify files exist before launching workers
        missing = [(label, path) for label, path in BEST_CHECKPOINTS if not Path(path).exists()]
        if missing:
            print("WARNING: the following checkpoint files were not found:")
            for label, path in missing:
                print(f"  [{label}] {path}")

        to_eval = [(label, path) for label, path in BEST_CHECKPOINTS if Path(path).exists()]
        if not to_eval:
            raise SystemExit("No checkpoint files found. Run from the src/ directory.")

        results = evaluate_all(to_eval, n_games=args.n_games, max_workers=args.max_workers, device=args.device)
        n_games = args.n_games

        if args.save_csv:
            export_csv(results)

    print_leaderboard(results)
    print_hyperparam_diff(results)
    plot_results(results, n_games=n_games, save_image=args.save_image)
