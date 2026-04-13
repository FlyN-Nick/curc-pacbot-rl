import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pacbot_rs import PacmanGym
import models
from policies import MaxQPolicy
from utils import OBS_SHAPE, NUM_ACTIONS, DETERMINISTIC_START_CONFIGURATION, select_device, step_env_until_done
import argparse

device = select_device(None)

# ── 1. Load model ──────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str):
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_class = getattr(models, loaded['config']['model'])
    q_net = model_class(OBS_SHAPE, NUM_ACTIONS)
    q_net.load_state_dict(loaded['state_dict'])
    q_net.eval()
    return q_net, loaded['iter_num'], loaded['epsilon'], loaded['config']

# ── 2. Evaluate one episode ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_episode(q_net, max_steps: int = 1000):
    gym = PacmanGym(DETERMINISTIC_START_CONFIGURATION)
    pellets_start = gym.remaining_pellets()
    policy = MaxQPolicy(q_net)
    done, step_num = step_env_until_done(gym, policy, 'cpu', max_steps=max_steps)
    is_board_cleared = done and gym.lives() == 3
    pellets_end = 0 if is_board_cleared else gym.remaining_pellets()
    return {
        'score': gym.score(),
        'steps': step_num,
        'board_cleared': is_board_cleared,
        'pellets_eaten': pellets_start - pellets_end,
    }

# ── 3. Evaluate one checkpoint (runs in separate process) ─────────────────────
def evaluate_checkpoint_worker(ckpt_path: str, n_games: int = 10):
    """This function runs in a separate process."""
    q_net, iter_num, epsilon, config = load_checkpoint(ckpt_path)
    
    scores, pellets, cleared = [], [], []
    for _ in range(n_games):
        result = evaluate_episode(q_net)
        scores.append(result['score'])
        pellets.append(result['pellets_eaten'])
        cleared.append(result['board_cleared'])
    
    return {
        'ckpt_path': ckpt_path,
        'iter_num': iter_num,
        'epsilon': epsilon,
        'config': config,
        'avg_score': float(np.mean(scores)),
        'max_score': float(np.max(scores)),
        'avg_pellets': float(np.mean(pellets)),
        'board_cleared_rate': float(np.mean(cleared)),
        'score_variance': float(np.var(scores)),
    }

# ── 4. Pick N evenly spaced checkpoints from a run ────────────────────────────
def pick_checkpoints(run_dir: str, n: int = 5, all_checkpoints: bool = False):
    ckpts = sorted(
        [p for p in Path(run_dir).glob("*.ckpt.pt") if 'latest' not in p.name],
        key=lambda p: int(p.stem.split('iter')[1].split('.')[0])
    )
    if not ckpts:
        return []
    
    if all_checkpoints:
        return [str(p) for p in ckpts]
        
    indices = np.linspace(0, len(ckpts) - 1, min(n, len(ckpts)), dtype=int)
    return [str(ckpts[i]) for i in indices]

# ── 5. Evaluate all runs in parallel ──────────────────────────────────────────
def evaluate_all_runs(runs: dict, n_checkpoints: int = 5, n_games: int = 10, max_workers: int = 4, all_checkpoints: bool = False):
    tasks = []
    for run_name, run_dir in runs.items():
        ckpts = pick_checkpoints(run_dir, n=n_checkpoints, all_checkpoints=all_checkpoints)
        if not ckpts:
            print(f"No checkpoints found in {run_dir}, skipping.")
            continue
        for ckpt in ckpts:
            tasks.append((run_name, ckpt))
    
    print(f"Evaluating {len(tasks)} checkpoints across {len(runs)} runs ({max_workers} workers)...")
    
    results_by_run = {run_name: [] for run_name in runs}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(evaluate_checkpoint_worker, ckpt, n_games): (run_name, ckpt)
            for run_name, ckpt in tasks
        }
        for future in as_completed(future_to_task):
            run_name, ckpt = future_to_task[future]
            try:
                result = future.result()
                results_by_run[run_name].append(result)
                print(f"  [{run_name}] iter={result['iter_num']}, avg_score={result['avg_score']:.1f}, max_score={result['max_score']}")
            except Exception as e:
                print(f"  [{run_name}] {ckpt} failed: {e}")
    
    for run_name in results_by_run:
        results_by_run[run_name].sort(key=lambda r: r['iter_num'])
    
    return results_by_run

# ── 6. Plot results ────────────────────────────────────────────────────────────
def plot_results(all_run_results: dict, save_image: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PacBot Training Run Comparison', fontsize=16)

    metrics = [
        ('avg_score', 'Average Score'),
        ('max_score', 'Max Score'),
        ('avg_pellets', 'Average Pellets Eaten'),
        ('score_variance', 'Score Variance'),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        for run_name, results in all_run_results.items():
            if not results:
                continue
            iters = [r['iter_num'] for r in results]
            values = [r[metric] for r in results]
            ax.plot(iters, values, marker='o', label=run_name)

            for r in results:
                if r['epsilon'] < 0.5:
                    ax.axvline(x=r['iter_num'], color='red', linestyle='--', alpha=0.3, label='ε < 0.5')
                    break

        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_image:
        plt.savefig('eval_results.png', dpi=150)
        print("\nSaved graphs to eval_results.png")
    plt.show()

# ── 7. Print hyperparameter comparison table ──────────────────────────────────
def print_hyperparam_table(all_run_results: dict):
    print("\n=== Hyperparameter Comparison ===")
    
    first_results = {
        run_name: results[0] 
        for run_name, results in all_run_results.items() 
        if results
    }
    if not first_results:
        return
    
    keys = list(next(iter(first_results.values()))['config'].keys())
    col_width = 22
    
    print(f"{'Hyperparameter':<35}", end="")
    for run_name in first_results:
        print(f"{run_name:<{col_width}}", end="")
    print()
    print("-" * (35 + col_width * len(first_results)))
    
    for key in keys:
        values = [str(r['config'].get(key, 'N/A')) for r in first_results.values()]
        differs = len(set(values)) > 1
        marker = " <<<" if differs else ""
        print(f"{key:<35}", end="")
        for v in values:
            print(f"{v:<{col_width}}", end="")
        print(marker)

# ── 8. Export to CSV ──────────────────────────────────────────────────────────
def export_csv(all_run_results: dict, filename: str = 'eval_results.csv'):
    fieldnames = ['run_name', 'iter_num', 'epsilon', 'avg_score', 'max_score', 'avg_pellets', 'board_cleared_rate', 'score_variance']
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, results in all_run_results.items():
            for r in results:
                row = {'run_name': run_name}
                for field in fieldnames[1:]:
                    row[field] = r.get(field, '')
                writer.writerow(row)
    print(f"\nSaved results to {filename}")

# ── 9. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PacBot training runs and plot results.")
    parser.add_argument("--n-checkpoints", type=int, default=20, help="Number of checkpoints to evaluate per run (default: 20).")
    parser.add_argument("--n-games", type=int, default=10, help="Number of games to play per checkpoint (default: 10).")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers (default: 4).")
    parser.add_argument("--all-checkpoints", action="store_true", help="Evaluate all available checkpoints (overrides --n-checkpoints).")
    parser.add_argument("--save-image", action="store_true", default=True, help="Save the plot as 'eval_results.png' (default: True).")
    parser.add_argument("--no-save-image", action="store_false", dest="save_image", help="Do not save the plot as an image.")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Save the results as 'eval_results.csv' (default: True).")
    parser.add_argument("--no-save-csv", action="store_false", dest="save_csv", help="Do not save the results as a CSV.")
    
    args = parser.parse_args()

    runs = {
        'run1 (urlriljg)': 'checkpoints_EC2/checkpoints/checkpoints',
        'run2 (jhx7hq3e)': 'checkpoints_EC2/second-checkpoints/checkpoints',
        'winnie':          'checkpoints_EC2/winnie-checkpoints/checkpoints',
    }

    all_run_results = evaluate_all_runs(
        runs,
        n_checkpoints=args.n_checkpoints,
        n_games=args.n_games,
        max_workers=args.max_workers,
        all_checkpoints=args.all_checkpoints
    )
    
    print_hyperparam_table(all_run_results)
    if args.save_csv:
        export_csv(all_run_results)
    plot_results(all_run_results, save_image=args.save_image)
