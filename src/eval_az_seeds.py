"""Fixed-seed evaluation of AlphaZero checkpoints with search: seeds 1..N at each speed (as
game_report), all games at once. Usage: eval_az_seeds.py SEEDS OUT.csv CKPT:SIMS ..."""
import csv, sys, time
import numpy as np, torch
import models, utils
from alphazero import evaluate_games
from pacbot_rs import GumbelConfig
utils.set_single_model(True)
obs_config = utils.replay_buffer_obs_config()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
seeds = range(1, int(sys.argv[1]) + 1)
with open(sys.argv[2], "w", newline="") as f:
    out = csv.writer(f)
    out.writerow(["checkpoint", "sims", "speed", "seed", "score", "steps", "cleared", "pellets_left"])
    for spec in sys.argv[3:]:
        path, sims = spec.rsplit(":", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        net = getattr(models, ck["config"]["model"])(utils.OBS_SHAPE, utils.NUM_ACTIONS).to(device)
        net.load_state_dict(ck["state_dict"]); net = net.to(memory_format=torch.channels_last).eval()
        cfg = ck["config"]
        search = GumbelConfig(num_simulations=int(sims), discount=cfg["discount"], reward_scale=cfg["reward_scale"],
                              c_visit=cfg["c_visit"], c_scale=cfg["c_scale"], gumbel_scale=0.0)
        t = time.perf_counter()
        res = evaluate_games(net, device, utils.ENV_VERSION, obs_config, utils.TRAINING_SPEEDS, seeds, search=search)
        name = path.split("/checkpoints/")[-1]
        for r in res:
            out.writerow([name, sims, r["speed"], r["seed"], r["score"], r["steps"], int(r["cleared"]), r["pellets_left"]])
        f.flush()
        sc = np.array([r["score"] for r in res])
        by = lambda lo, hi: np.mean([r["score"] for r in res if lo <= r["speed"] <= hi])
        print(f"{name} sims {sims}: mean {sc.mean():.0f} (se {sc.std()/np.sqrt(len(sc)):.0f}) | 4-8 {by(4,8):.0f} 9-11 {by(9,11):.0f} "
              f"12-14 {by(12,14):.0f} | cleared {np.mean([r['cleared'] for r in res]):.0%} | {time.perf_counter()-t:.0f}s", flush=True)
