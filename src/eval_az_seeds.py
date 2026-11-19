"""
Fixed-seed evaluation with search: seeds 1..N (or A-B) at each speed, as game_report, all games
at once (speeds 4-14, or up to PACBOT_MAX_SPEED). AlphaZero checkpoints play with their network; DQN checkpoints (any other model) play
with the same search, their Q values as the evaluator (alphazero.QNetworkEvaluator, policy
temperature TEMP, default 0.1).
Usage: eval_az_seeds.py SEEDS OUT.csv CKPT:SIMS[:TEMP] ...
"""
import csv, sys, time
import numpy as np, torch
import models, utils
from alphazero import QNetworkEvaluator, evaluate_games
from pacbot_rs import GumbelConfig
utils.set_single_model(True)
obs_config = utils.replay_buffer_obs_config()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
lo, _, hi = sys.argv[1].partition("-")
seeds = range(int(lo), int(hi) + 1) if hi else range(1, int(lo) + 1)
with open(sys.argv[2], "w", newline="") as f:
    out = csv.writer(f)
    out.writerow(["checkpoint", "sims", "temperature", "speed", "seed", "score", "steps", "cleared", "pellets_left"])
    for spec in sys.argv[3:]:
        path, sims, *temp = spec.split(":")
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["config"]
        net = getattr(models, cfg["model"])(utils.OBS_SHAPE, utils.NUM_ACTIONS).to(device)
        net.load_state_dict(ck["state_dict"]); net = net.to(memory_format=torch.channels_last).eval()
        alphazero = cfg["model"] == "PVNetV2"
        temperature = float(temp[0]) if temp else 0.1
        if alphazero:
            search = GumbelConfig(num_simulations=int(sims), discount=cfg["discount"], reward_scale=cfg["reward_scale"],
                                  c_visit=cfg["c_visit"], c_scale=cfg["c_scale"], gumbel_scale=0.0)
            evaluator, temperature = None, ""
        else:  # DQN: its own reward scale and discount, AlphaZero's search settings
            search = GumbelConfig(num_simulations=int(sims), discount=cfg["discount_factor"],
                                  reward_scale=cfg["reward_scale"], gumbel_scale=0.0)
            evaluator = QNetworkEvaluator(net, device, temperature)
        t = time.perf_counter()
        res = evaluate_games(net, device, utils.ENV_VERSION, obs_config, utils.training_speeds(), seeds, search=search,
                             evaluator=evaluator)
        name = path.split("/checkpoints/")[-1]
        for r in res:
            out.writerow([name, sims, temperature, r["speed"], r["seed"], r["score"], r["steps"], int(r["cleared"]), r["pellets_left"]])
        f.flush()
        sc = np.array([r["score"] for r in res])
        by = lambda lo, hi: np.mean([r["score"] for r in res if lo <= r["speed"] <= hi])
        slow = f" 15-18 {by(15, 18):.0f}" if utils.max_speed() > 14 else ""
        print(f"{name} sims {sims}{f' temp {temperature}' if temperature != '' else ''}: mean {sc.mean():.0f} (se {sc.std()/np.sqrt(len(sc)):.0f}) | 4-8 {by(4,8):.0f} 9-11 {by(9,11):.0f} "
              f"12-14 {by(12,14):.0f}{slow} | cleared {np.mean([r['cleared'] for r in res]):.0%} | {time.perf_counter()-t:.0f}s", flush=True)
