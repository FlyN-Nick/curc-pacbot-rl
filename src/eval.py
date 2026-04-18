import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pacbot_rs import PacmanGym
import models
from policies import MaxQPolicy
from utils import OBS_SHAPE, NUM_ACTIONS, DETERMINISTIC_START_CONFIGURATION, select_device, step_env_until_done
import argparse
from tqdm import tqdm

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

# ── 3. Evaluate one checkpoint (runs in separate process) ─────────────────────
def evaluate_checkpoint_worker(ckpt_path: str, n_games: int = 10, preferred_device: str = None, progress_queue=None):
    """This function runs in a separate process."""
    device = select_device(preferred_device)
    q_net, iter_num, epsilon, config = load_checkpoint(ckpt_path, device)
    
    scores, pellets, cleared = [], [], []
    for _ in range(n_games):
        result = evaluate_episode(q_net, device)
        scores.append(result['score'])
        pellets.append(result['pellets_eaten'])
        cleared.append(result['board_cleared'])
        if progress_queue is not None:
            progress_queue.put(1)
    
    return {
        'ckpt_path': ckpt_path,
        'iter_num': iter_num,
        'epsilon': epsilon,
        'config': config,
        'avg_score': float(np.mean(scores)),
        'max_score': float(np.max(scores)),
        'min_score': float(np.min(scores)),
        'score_variance': float(np.var(scores)),
        'score_std': float(np.std(scores)),
        'avg_pellets': float(np.mean(pellets)),
        'board_cleared_rate': float(np.mean(cleared)),
    }

# ── 4. Pick N evenly spaced checkpoints from a run ────────────────────────────
def pick_checkpoints(run_dir: str, n: int = 5, all_checkpoints: bool = False):
    ckpts = sorted(
        [p for p in Path(run_dir).glob("*.ckpt.pt") if 'latest' not in p.name and 'best' not in p.name],
        key=lambda p: int(p.stem.split('iter')[1].split('.')[0])
    )
    if not ckpts:
        return []
    
    if all_checkpoints:
        return [str(p) for p in ckpts]
        
    indices = np.linspace(0, len(ckpts) - 1, min(n, len(ckpts)), dtype=int)
    return [str(ckpts[i]) for i in indices]

# ── 5. Evaluate all runs in parallel ──────────────────────────────────────────
def evaluate_all_runs(runs: dict, n_checkpoints: int = 5, n_games: int = 10, max_workers: int = 4, all_checkpoints: bool = False, device: str = None):
    tasks = []
    for run_name, run_dir in runs.items():
        ckpts = pick_checkpoints(run_dir, n=n_checkpoints, all_checkpoints=all_checkpoints)
        if not ckpts:
            print(f"No checkpoints found in {run_dir}, skipping.")
            continue
        for ckpt in ckpts:
            tasks.append((run_name, ckpt))
    
    device_list = [d.strip() for d in device.split(',')] if device else [None]
    actual_devices = [str(select_device(d)) for d in device_list]
    print(f"Using devices: {', '.join(actual_devices)}")
    print(f"Evaluating {len(tasks)} checkpoints across {len(runs)} runs ({max_workers} workers)...")
    
    results_by_run = {run_name: [] for run_name in runs}
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for i, (run_name, ckpt) in enumerate(tasks):
                # Cycle through devices for each task
                worker_device = device_list[i % len(device_list)]
                future = executor.submit(evaluate_checkpoint_worker, ckpt, n_games, worker_device, progress_queue)
                future_to_task[future] = (run_name, ckpt)
                
            with tqdm(total=len(tasks) * n_games, desc="Evaluating games") as pbar:
                pending = list(future_to_task.keys())
                while pending:
                    # Wait for any future to finish, but time out frequently to update progress
                    done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                    
                    # Update progress bar from queue
                    while not progress_queue.empty():
                        try:
                            pbar.update(progress_queue.get_nowait())
                        except:
                            break
                    
                    for future in done:
                        run_name, ckpt = future_to_task[future]
                        try:
                            result = future.result()
                            results_by_run[run_name].append(result)
                        except Exception as e:
                            tqdm.write(f"  [{run_name}] {ckpt} failed: {e}")
    
    for run_name in results_by_run:
        results_by_run[run_name].sort(key=lambda r: r['iter_num'])
    
    return results_by_run

# ── 6. Plot results ────────────────────────────────────────────────────────────
def plot_results(all_run_results: dict, n_games: int, save_image: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PacBot Training Run Comparison ({n_games} games/checkpoint)', fontsize=16)

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
            ax.plot(iters, values, label=run_name)

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
    
    # Get the first result for each run that actually has a config
    first_results = {
        run_name: results[0] 
        for run_name, results in all_run_results.items() 
        if results and 'config' in results[0]
    }
    if not first_results:
        print("Config information not available (likely loaded from CSV).")
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
def print_best_checkpoints(all_run_results: dict, top_n: int = 5):
    all_checkpoints = []
    for run_name, results in all_run_results.items():
        for r in results:
            ckpt_info = r.copy()
            ckpt_info['run_name'] = run_name
            all_checkpoints.append(ckpt_info)
    
    if not all_checkpoints:
        return

    print(f"\n=== Top {top_n} Checkpoints by Average Score ===")
    top_avg = sorted(all_checkpoints, key=lambda x: x['avg_score'], reverse=True)[:top_n]
    print(f"{'Run':<20} {'Iter':<10} {'Avg Score':<12} {'Max Score':<12} {'Clear Rate':<12}")
    for r in top_avg:
        print(f"{r['run_name'][:20]:<20} {r['iter_num']:<10} {r['avg_score']:<12.1f} {r['max_score']:<12.1f} {r.get('board_cleared_rate', 0):<12.2%}")

    print(f"\n=== Top {top_n} Checkpoints by Highest Score ===")
    top_max = sorted(all_checkpoints, key=lambda x: x['max_score'], reverse=True)[:top_n]
    print(f"{'Run':<20} {'Iter':<10} {'Max Score':<12} {'Avg Score':<12} {'Clear Rate':<12}")
    for r in top_max:
        print(f"{r['run_name'][:20]:<20} {r['iter_num']:<10} {r['max_score']:<12.1f} {r['avg_score']:<12.1f} {r.get('board_cleared_rate', 0):<12.2%}")

def export_csv(all_run_results: dict, n_games: int, filename: str = 'eval_results.csv'):
    fieldnames = ['run_name', 'iter_num', 'epsilon', 'avg_score', 'max_score', 'min_score', 'score_std', 'avg_pellets', 'board_cleared_rate', 'score_variance']
    
    with open(filename, mode='w', newline='') as f:
        f.write(f"# Evaluation metadata: n_games_per_checkpoint={n_games}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, results in all_run_results.items():
            for r in results:
                row = {'run_name': run_name}
                for field in fieldnames[1:]:
                    row[field] = r.get(field, '')
                writer.writerow(row)
    print(f"Saved results to {filename}")

def export_hyperparams_csv(all_run_results: dict, filename: str = 'eval_hyperparams.csv'):
    # Get the first result for each run that actually has a config
    first_results = {
        run_name: results[0] 
        for run_name, results in all_run_results.items() 
        if results and 'config' in results[0]
    }
    if not first_results:
        return
    
    keys = list(next(iter(first_results.values()))['config'].keys())
    run_names = list(first_results.keys())
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header: Hyperparameter, Run1, Run2, ...
        writer.writerow(['Hyperparameter'] + run_names)
        
        for key in keys:
            row = [key]
            for run_name in run_names:
                val = first_results[run_name]['config'].get(key, 'N/A')
                row.append(val)
            writer.writerow(row)
    print(f"Saved hyperparameters to {filename}")

# ── 9. Import from CSV ────────────────────────────────────────────────────────
def load_csv(filename: str):
    results_by_run = {}
    n_games = None
    with open(filename, mode='r') as f:
        # Check for metadata line
        pos = f.tell()
        first_line = f.readline()
        if first_line.startswith("# Evaluation metadata:"):
             parts = first_line.split("n_games_per_checkpoint=")
             if len(parts) > 1:
                 try:
                     n_games = int(parts[1].strip())
                 except ValueError:
                     n_games = None
        else:
            f.seek(pos)
            
        reader = csv.DictReader(f)
        for row in reader:
            run_name = row['run_name']
            if run_name not in results_by_run:
                results_by_run[run_name] = []
            
            # Convert types
            processed_row = {
                'iter_num': int(row['iter_num']),
                'epsilon': float(row['epsilon']),
                'avg_score': float(row['avg_score']),
                'max_score': float(row['max_score']),
                'min_score': float(row['min_score']),
                'score_std': float(row['score_std']),
                'avg_pellets': float(row['avg_pellets']),
                'board_cleared_rate': float(row['board_cleared_rate']),
                'score_variance': float(row['score_variance'])
            }
            results_by_run[run_name].append(processed_row)
            
    # Sort results by iter_num for each run
    for run_name in results_by_run:
        results_by_run[run_name].sort(key=lambda r: r['iter_num'])
            
    return results_by_run, n_games

# ── 10. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PacBot training runs and plot results.")
    parser.add_argument("--csv", type=str, help="Path to an existing eval_results.csv file to analyze.")
    parser.add_argument("--n-checkpoints", type=int, default=20, help="Number of checkpoints to evaluate per run (default: 20).")
    parser.add_argument("--n-games", type=int, default=10, help="Number of games to play per checkpoint (default: 10).")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers (default: 4).")
    parser.add_argument("--all-checkpoints", action="store_true", help="Evaluate all available checkpoints (overrides --n-checkpoints).")
    parser.add_argument("--save-image", action="store_true", default=True, help="Save the plot as 'eval_results.png' (default: True).")
    parser.add_argument("--no-save-image", action="store_false", dest="save_image", help="Do not save the plot as an image.")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Save the results as 'eval_results.csv' (default: True).")
    parser.add_argument("--no-save-csv", action="store_false", dest="save_csv", help="Do not save the results as a CSV.")
    parser.add_argument("--device", type=str, default=None, help="Device(s) to use, comma-separated (e.g., 'cpu', 'mps', 'mps,cpu'). Defaults to auto-select.")
    
    args = parser.parse_args()

    if args.csv:
        print(f"Loading results from {args.csv}...")
        all_run_results, loaded_n_games = load_csv(args.csv)
        n_games = loaded_n_games if loaded_n_games is not None else args.n_games
    else:
        runs = {
            'nick_ec2': 'checkpoints_EC2/carl/checkpoints',
            'nick_local': 'checkpoints/',
            'winnie_ec2': 'checkpoints_EC2/winnie-new/checkpoints',
            
        }

        all_run_results = evaluate_all_runs(
            runs,
            n_checkpoints=args.n_checkpoints,
            n_games=args.n_games,
            max_workers=args.max_workers,
            all_checkpoints=args.all_checkpoints,
            device=args.device
        )
        n_games = args.n_games
        
        if args.save_csv:
            export_csv(all_run_results, n_games=n_games)
            export_hyperparams_csv(all_run_results)
    
    print_hyperparam_table(all_run_results)
    print_best_checkpoints(all_run_results)
    plot_results(all_run_results, n_games=n_games, save_image=args.save_image)
