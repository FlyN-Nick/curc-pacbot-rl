"""
MuZero, EfficientZero, Gumbel MuZero and Stochastic MuZero on the Pacbot game, with LightZero
(https://github.com/opendilab/LightZero, installed in its own environment; see docs/muzero.md).

    python -m algorithms.train_lightzero --algorithm muzero --checkpoint-dir checkpoints/mz1

Each launch writes LightZero's logs (TensorBoard, log/) and checkpoints (ckpt/iteration_N.pth.tar,
ckpt/ckpt_best.pth.tar) to its own folder, --checkpoint-dir/run-YYMMDD-HHMMSS. It evaluates every
--eval_freq training iterations on fixed-seed games (by default one per speed 4-14, seed 1).
--resume starts from a checkpoint's network (LightZero doesn't save its replay buffer or
optimizer), and --num_moves counts from there.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightzero_pacbot.configs import ALGORITHMS, make_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--num_moves", type=int, default=40_000_000, help="game moves to collect")
    parser.add_argument("--num_simulations", type=int, default=32)
    parser.add_argument("--collector_env_num", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_ratio", type=float, default=0.25, help="updates per collected move")
    parser.add_argument("--reanalyze_ratio", type=float, default=0.0)
    parser.add_argument("--eval_freq", type=int, default=5_000, help="training iterations between evaluations")
    parser.add_argument("--eval_seeds", type=int, default=1, help="fixed seeds per speed (4-14) in each evaluation")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--speed_sampling", default="equal_moves")
    parser.add_argument("--max_speed", type=int, default=18)
    parser.add_argument("--random_start_proportion", type=float, default=0.25)
    parser.add_argument("--full_game", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", default=None, help="checkpoint (.pth.tar) to start from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eval_games = [(speed, seed) for speed in range(4, 15) for seed in range(1, args.eval_seeds + 1)]
    # LightZero writes under exp_name relative to the working directory
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    exp_name = datetime.datetime.now().strftime("run-%y%m%d-%H%M%S")
    main_config, create_config = make_config(
        args.algorithm, exp_name, num_simulations=args.num_simulations,
        collector_env_num=args.collector_env_num, evaluator_env_num=min(11, len(eval_games)),
        eval_games=eval_games, batch_size=args.batch_size, replay_ratio=args.replay_ratio,
        eval_freq=args.eval_freq, cuda=args.device == "cuda", speed_sampling=args.speed_sampling,
        max_speed=args.max_speed, random_start_proportion=args.random_start_proportion,
        full_game=args.full_game, num_res_blocks=args.num_res_blocks, num_channels=args.num_channels,
        learning_rate=args.learning_rate, reanalyze_ratio=args.reanalyze_ratio)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(checkpoint_dir)
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=args.seed, model_path=str(Path(args.resume).resolve()) if args.resume else None,
                 max_env_step=args.num_moves)


if __name__ == "__main__":
    main()
