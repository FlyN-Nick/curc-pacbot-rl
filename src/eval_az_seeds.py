"""
Fixed-seed evaluation with search: seeds 1..N (or A-B) at each speed, as game_report, all games
at once (speeds 4-14, or up to PACBOT_MAX_SPEED). AlphaZero checkpoints play with their network; DQN checkpoints (any other model) play
with the same search, their Q values as the evaluator (alphazero.QNetworkEvaluator, policy
temperature TEMP, default 0.1).
Usage: eval_az_seeds.py SEEDS OUT.csv CKPT:SIMS[:TEMP] ...
With PACBOT_FULL_GAME=1, every checkpoint plays full games (three lives, board after board, as in
competition) instead of ending at the first lost life or cleared board; "cleared" is then at
least one board, and the boards and lives columns say how many boards it cleared and lives it kept.
Each game's row is written as soon as that game ends (so rows come in the order games finish), and
a progress line is printed every 48 games; the games play exactly as when the rows came at the end.
"""
import csv, sys, time
import numpy as np, torch
import models, utils
from alphazero import QNetworkEvaluator, evaluate_games, search_config
from pacbot_rs import GumbelConfig
utils.set_single_model(True)
obs_config = utils.replay_buffer_obs_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
lo, _, hi = sys.argv[1].partition("-")
seeds = range(int(lo), int(hi) + 1) if hi else range(1, int(lo) + 1)
with open(sys.argv[2], "w", newline="") as f:
    out = csv.writer(f)
    out.writerow(["checkpoint", "sims", "temperature", "speed", "seed", "score", "steps", "cleared", "pellets_left",
                  "boards", "lives"])
    if utils.full_game():
        print("Full games: three lives, board after board", flush=True)
    for spec in sys.argv[3:]:
        path, sims, *temp = spec.split(":")
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["config"]
        net = getattr(models, cfg["model"])(utils.OBS_SHAPE, utils.NUM_ACTIONS).to(device)
        net.load_state_dict(ck["state_dict"]); net = net.to(memory_format=torch.channels_last).eval()
        alphazero = models.is_policy_value(net)  # AlphaZero or PPO; otherwise a DQN
        temperature = float(temp[0]) if temp else 0.1
        if alphazero:
            search = search_config(cfg, int(sims), explore=False)  # Gumbel or PUCT, as the run trained
            evaluator, temperature = None, ""
        else:  # DQN: its own reward scale and discount, AlphaZero's search settings
            search = GumbelConfig(num_simulations=int(sims), discount=cfg["discount_factor"],
                                  reward_scale=cfg["reward_scale"], gumbel_scale=0.0)
            evaluator = QNetworkEvaluator(net, device, temperature)
        t = time.perf_counter()
        name = path.split("/checkpoints/")[-1]
        total = len(utils.training_speeds()) * len(seeds)
        finished = []

        def write(r, name=name, sims=sims, temperature=temperature, t=t):
            out.writerow([name, sims, temperature, r["speed"], r["seed"], r["score"], r["steps"], int(r["cleared"]), r["pellets_left"],
                          r["boards"], r["lives"]])
            f.flush()
            finished.append(r)
            if len(finished) % 48 == 0 or len(finished) == total:
                print(f"  {name} sims {sims}: {len(finished)} of {total} games done, "
                      f"{time.perf_counter()-t:.0f}s", flush=True)

        res = evaluate_games(net, device, utils.ENV_VERSION, obs_config, utils.training_speeds(), seeds, search=search,
                             evaluator=evaluator, on_done=write)
        sc = np.array([r["score"] for r in res])
        by = lambda lo, hi: np.mean([r["score"] for r in res if lo <= r["speed"] <= hi])
        slow = f" 15-18 {by(15, 18):.0f}" if utils.max_speed() > 14 else ""
        boards = f" | boards {np.mean([r['boards'] for r in res]):.2f}" if utils.full_game() else ""
        print(f"{name} sims {sims}{f' temp {temperature}' if temperature != '' else ''}: mean {sc.mean():.0f} (se {sc.std()/np.sqrt(len(sc)):.0f}) | 4-8 {by(4,8):.0f} 9-11 {by(9,11):.0f} "
              f"12-14 {by(12,14):.0f}{slow} | cleared {np.mean([r['cleared'] for r in res]):.0%}{boards} | {time.perf_counter()-t:.0f}s", flush=True)
