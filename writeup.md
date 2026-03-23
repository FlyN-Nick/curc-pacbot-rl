# DQN Hyperparameter Writeup

This writeup is based on `src/algorithms/train_dqn.py` and the helper modules it depends on:

- `src/algorithms/train_dqn.py`
- `src/replay_buffer.py`
- `src/policies.py`
- `src/utils.py`
- `src/models.py`

The current trainer is a dueling Double DQN setup:

- `QNetV2` is the default model.
- action selection is epsilon-greedy during data collection.
- target computation uses Double DQN logic.
- training uses a uniform replay buffer.
- loss is plain MSE between predicted Q-values and 1-step bootstrapped targets.

There are also two important implementation details that affect tuning:

1. Training only controls the "purgatory mode" portion of the game. When ghosts are close or frightened, control is delegated to a fixed pretrained policy in `src/utils.py`.
2. `random_start_proportion` is exposed as a hyperparameter, but the current `ReplayBuffer` implementation does not actually use it when constructing environments.

## How the training loop works

Per training iteration, the script does this:

1. Optionally refresh the target network.
2. Sample one batch uniformly from replay.
3. Compute the Double DQN target:
   `target = scaled_reward + discount_factor * target_q(next_state, argmax online_q(next_state))`
4. Run one Adam update on MSE loss.
5. Log metrics and sometimes run an evaluation episode.
6. Anneal epsilon linearly.
7. Collect `experience_steps` new replay steps from each of `num_parallel_envs` environments.

So one outer loop iteration is not "one environment step". It is:

- 1 gradient step
- `experience_steps * num_parallel_envs` new transitions added to replay

With the defaults, that is:

- `4 * 32 = 128` new transitions per iteration
- `2,000,000` optimizer steps total
- about `256,000,000` environment transitions generated over a full run

That ratio matters a lot when tuning replay freshness, exploration, and target staleness.

## Hyperparameters

### `learning_rate = 1e-4`

Controls Adam's step size.

What it does:

- Higher values learn faster early, but make Q-learning more likely to diverge or oscillate.
- Lower values are safer, but may waste compute because the policy improves too slowly.

What it interacts with:

- `batch_size`: larger batches usually support a somewhat larger learning rate, but in DQN they can also reduce gradient noise so much that overestimation errors become more systematic.
- `reward_scale`: scaling rewards down also scales target magnitudes down, which effectively makes the same learning rate more conservative.
- `target_network_update_steps`: if the target network is stale for too long, a high learning rate can make the online network drift too far from the target network.
- `grad_clip_norm`: large clipping can hide an overly high learning rate instead of fixing it.

Current read:

- `1e-4` is a conservative, reasonable default.

### `batch_size = 512`

Controls how many replay items are used in each SGD update.

What it does:

- Larger batches reduce gradient variance.
- Larger batches increase GPU utilization.
- Larger batches require more replay diversity to avoid repeatedly fitting nearly identical data.

What it interacts with:

- `replay_buffer_size`: a small replay buffer plus large batch means strong sample reuse and a narrower data distribution.
- `experience_steps` and `num_parallel_envs`: these determine how quickly new data enters the replay buffer. If data enters slowly relative to batch size, training becomes too off-policy on stale samples.
- `learning_rate`: the stable learning-rate range depends on batch size.

Current read:

- `512` is fairly large for a replay buffer of only `30,000`.
- You are sampling about `512 / 30,000 = 1.7%` of the entire buffer each update.
- Since the loop also performs 1 optimizer step every 128 newly collected transitions, the trainer is fairly update-heavy.

### `num_iters = 2_000_000`

Controls the total number of outer-loop iterations.

What it does:

- Sets total optimizer updates.
- Sets total data collection volume.
- Defines the length of the linear epsilon schedule.

What it interacts with:

- `initial_epsilon` and `final_epsilon`: epsilon anneals linearly over the full `num_iters`.
- `evaluate_steps`: small `evaluate_steps` becomes very expensive as `num_iters` grows.
- `target_network_update_steps`: determines how many target refreshes happen over the run.

Current read:

- This is a very long run.
- Because epsilon anneals across the full run, the agent stays exploratory for a long time unless you lower `num_iters` or change the epsilon schedule.

### `replay_buffer_size = 30_000`

Controls replay capacity.

What it does:

- Larger buffers improve sample diversity and reduce short-term correlations.
- Smaller buffers make learning more responsive to recent policy changes.

What it interacts with:

- `batch_size`: bigger batches need bigger buffers.
- `experience_steps * num_parallel_envs`: this determines how fast old data is overwritten.
- `initial_epsilon/final_epsilon`: a larger buffer preserves more exploratory data for longer.

Current read:

- `30,000` is modest relative to:
  - `512` batch size
  - `128` new transitions per iteration
- The whole buffer turns over in about `30,000 / 128 ~= 234` iterations, which is fast.
- That means targets are being learned from a replay distribution that changes quickly.

### `num_parallel_envs = 32`

Controls how many Pacman environments run in parallel inside the replay buffer.

What it does:

- Increases data collection throughput.
- Reduces correlation compared with a single environment.
- Increases CPU-side environment stepping cost and memory traffic.

What it interacts with:

- `experience_steps`: together they determine transitions added per iteration.
- `replay_buffer_size`: more environments cause the buffer to refresh faster.
- `batch_size`: if collection is fast enough, larger batches become more reasonable.

Current read:

- `32` is a good throughput-oriented default if the CPU side can keep up.
- On a single-GPU setup, the bottleneck may still be environment stepping or evaluation, not the neural net.

### `random_start_proportion = 0.5`

Intended meaning:

- Presumably, some fraction of environments should start from randomized starts.

Actual behavior in the current code:

- It is passed into `ReplayBuffer(...)`.
- It is not used when the environments are created.
- All replay environments are currently created with `DEFAULT_GYM_CONFIGURATION`.

Current read:

- Tuning this currently does nothing.
- If you want it to matter, `src/replay_buffer.py` needs to be fixed.

### `experience_steps = 4`

Controls how many replay collection steps happen after each optimizer step.

What it does:

- Higher values make training more on-policy and make replay fresher.
- Lower values increase sample reuse and make learning more update-heavy.

What it interacts with:

- `num_parallel_envs`: together define transitions added per iteration.
- `replay_buffer_size`: together define replay turnover speed.
- `batch_size`: together define update-to-data ratio.
- `target_network_update_steps`: if data distribution changes quickly but target updates are infrequent, the targets become stale.

Current read:

- At `4` and `32 envs`, you add `128` transitions per iteration.
- That is a reasonable start, but the trainer still does a lot of updates relative to replay size because the batch is large and the buffer is small.

### `target_network_update_steps = 5_000`

Controls how often the target network is refreshed by copying the online network.

What it does:

- More frequent updates reduce target staleness.
- Less frequent updates improve stability by slowing the moving target problem.

What it interacts with:

- `learning_rate`: higher LR usually wants more frequent target refreshes.
- `experience_steps`, `num_parallel_envs`, `replay_buffer_size`: if the replay distribution changes quickly, the target network can become stale faster in wall-clock data terms.
- `discount_factor`: higher discount makes future estimates matter more, which increases sensitivity to target quality.

Current read:

- `5,000` iterations means `5,000 * 128 = 640,000` new transitions between target updates at default settings.
- That is a large amount of newly collected experience between target syncs.
- It may be more stale than ideal for this replay size.

### `evaluate_steps = 10`

Controls how often an evaluation episode is run.

What it does:

- Gives dense monitoring.
- Adds overhead because evaluation is a full episode rollout.

What it interacts with:

- `num_iters`: with 2,000,000 iterations and evaluation every 10 iterations, this schedules 200,000 evaluations.
- `max_steps` inside `evaluate_episode`: each evaluation can be nontrivial.

Current read:

- This is far too frequent for efficient training.
- It likely wastes a significant amount of runtime on evaluation.
- It helps observability, but not learning quality.

### `initial_epsilon = 0.8`

Controls initial exploration probability.

What it does:

- Higher values increase random action selection early.
- Lower values exploit the learned Q-network sooner.

What it interacts with:

- `final_epsilon`
- `num_iters` because the script uses linear annealing over the full run
- `replay_buffer_size` because the replay buffer stores exploratory behavior

Current read:

- `0.8` is plausible, though many DQN setups start at `1.0`.
- Because the game has action masks and only 5 actions, very high early exploration is less dangerous than in large-action domains.

### `final_epsilon = 0.1`

Controls the exploration floor.

What it does:

- Prevents the data distribution from becoming fully greedy.
- Keeps collecting broad replay support late in training.

What it interacts with:

- `replay_buffer_size`: if replay is small, a high epsilon floor keeps the dataset noisy.
- `evaluate_steps`: evaluation uses greedy `MaxQPolicy`, so train/eval mismatch grows as `final_epsilon` rises.

Current read:

- `0.1` is on the exploratory side for a late-stage DQN policy.
- Good for robustness, but it can cap asymptotic training performance if the replay never becomes sharp enough.

### `discount_factor = 0.99`

Controls how much future return matters in the TD target.

What it does:

- Higher values make the agent more far-sighted.
- Higher values also amplify bootstrap error.

What it interacts with:

- `target_network_update_steps`: stale targets are more harmful at high discount.
- `reward_scale`: discounted targets still need to stay in a numerically comfortable range.
- episode length and delayed rewards in Pacman: if survival and board clearing require long-horizon planning, too low a discount hurts badly.

Current read:

- `0.99` is standard and likely correct for Pacman.
- I would not change this first.

### `reward_scale = 1 / 50`

Rewards are multiplied by this value before training.

What it does:

- Shrinks the TD target magnitude.
- Makes optimization numerically easier.

What it interacts with:

- `learning_rate`: smaller targets effectively make the optimization problem gentler.
- `grad_clip_norm`: if rewards are small, huge clipping thresholds become even less meaningful.
- logged Q-value metrics are divided by `reward_scale` again to report them in environment reward units.

Current read:

- This is a reasonable stabilizer.
- Unless raw rewards are already very small, keeping some reward normalization is sensible.

### `grad_clip_norm = 10_000`

Controls gradient clipping threshold.

What it does:

- Prevents catastrophic gradient explosions if they happen.

What it interacts with:

- `learning_rate`
- `reward_scale`
- `batch_size`

Current read:

- `10,000` is so large that it is effectively "no clipping" for most normal training.
- If clipping is intended as a stabilizer, this value is not doing much.

### `model = "QNetV2"`

Chooses the Q-network class from `src/models.py`.

What it does:

- Changes model capacity, representation quality, and compute cost.

What it interacts with:

- `batch_size`: a smaller model can support bigger batches.
- `learning_rate`: larger models sometimes need a smaller LR to remain stable.
- replay quality: bigger models overfit more easily to a small, fast-turnover replay buffer.

Current read:

- `QNetV2` is the strongest default in this repo and matches the trainer design well.

## Hyperparameter interactions that matter most

If you only track a few interactions, track these:

### 1. Replay freshness: `batch_size`, `replay_buffer_size`, `experience_steps`, `num_parallel_envs`

These four parameters decide whether the trainer is learning from a healthy mix of data or repeatedly fitting a narrow, quickly changing replay distribution.

Right now the setup is:

- large batch
- modest replay
- moderately fast replay turnover

That combination often produces unstable or plateaued DQN learning.

### 2. Stability triangle: `learning_rate`, `target_network_update_steps`, `discount_factor`

These three control bootstrap stability.

- higher `learning_rate` pushes the online network faster
- higher `discount_factor` makes targets more sensitive to future Q estimates
- larger `target_network_update_steps` makes the target network older

If you increase one aggressive term, you usually need to make at least one of the others more conservative.

### 3. Exploration schedule: `initial_epsilon`, `final_epsilon`, `num_iters`

Because epsilon anneals across the full run, `num_iters` is not just a runtime parameter. It is also part of the exploration policy.

This means:

- doubling `num_iters` slows epsilon decay
- lowering `final_epsilon` only helps late if the schedule reaches it early enough

### 4. Optimization scale: `reward_scale`, `learning_rate`, `grad_clip_norm`

These define the numerical scale of the TD update.

Right now:

- rewards are scaled down
- LR is conservative
- clipping is effectively disabled

That is safe, but possibly slower and looser than necessary.

## Recommendations to improve performance

Below are the changes I would make first for stronger training performance on your setup.

### High-priority changes

1. Increase `replay_buffer_size` substantially.

Recommended range:

- `100_000` to `300_000`

Why:

- Your current replay is small relative to batch size and collection rate.
- A larger replay buffer should reduce overfitting to short-term policy behavior and improve TD target diversity.

Best first try:

- `--replay_buffer_size 150000`

2. Reduce evaluation frequency drastically.

Recommended range:

- `500` to `2_000`

Why:

- `evaluate_steps=10` is monitoring-heavy and almost certainly wasting runtime.
- Those cycles would be better spent collecting data and training.

Best first try:

- `--evaluate_steps 1000`

3. Refresh the target network more often.

Recommended range:

- `1_000` to `2_000`

Why:

- With your current data collection rate, `5,000` iterations is a lot of replay churn between target syncs.
- More frequent target refreshes usually help when the replay buffer is not huge and data distribution changes quickly.

Best first try:

- `--target_network_update_steps 1500`

4. Lower the late-stage exploration floor.

Recommended range:

- `0.02` to `0.05`

Why:

- `final_epsilon=0.1` keeps training replay fairly noisy forever.
- In a masked 5-action domain, you can usually afford a lower floor once the agent is competent.

Best first try:

- `--final_epsilon 0.03`

5. Use meaningful gradient clipping.

Recommended range:

- `10` to `100`

Why:

- `10_000` is effectively not clipping.
- DQN benefits from a real failsafe against rare TD spikes.

Best first try:

- `--grad_clip_norm 50`

### Medium-priority changes

6. Try a slightly smaller batch first, not a larger one.

Recommended range:

- `256` or `384`

Why:

- With the current replay design, `512` is large enough to increase replay reuse pressure.
- A somewhat smaller batch often improves learning dynamics before it hurts throughput.

Best first try:

- `--batch_size 256`

7. Keep `learning_rate` near `1e-4` initially, then test `2e-4` only after the replay changes.

Why:

- Fix replay quality and target freshness first.
- If those changes stabilize learning, then a moderate LR increase may improve wall-clock progress.

Suggested order:

- first run at `1e-4`
- second sweep at `2e-4`

8. Consider more data collection per update if learning still overfits replay.

Recommended range:

- `experience_steps 6` to `8`

Why:

- This shifts the update-to-data ratio toward fresher replay.
- It is useful if loss drops but evaluation score stagnates.

Best first try:

- `--experience_steps 6`

### Low-priority changes

9. Leave `discount_factor` at `0.99` for now.

Why:

- This is the least suspicious hyperparameter in the current setup.

10. Leave `reward_scale` alone unless you see obvious optimization issues.

Why:

- It is serving a stabilizing role.
- Change it only if Q-values become badly mis-scaled or gradients are consistently tiny.

## Suggested next experiment

If you want one concrete next run instead of a sweep, I would start with:

```bash
python3 -m algorithms.train_dqn \
  --replay_buffer_size 150000 \
  --batch_size 256 \
  --target_network_update_steps 1500 \
  --evaluate_steps 1000 \
  --final_epsilon 0.03 \
  --grad_clip_norm 50 \
  --experience_steps 6
```

I would keep:

- `--learning_rate 0.0001`
- `--initial_epsilon 0.8`
- `--discount_factor 0.99`
- `--reward_scale 0.02`
- `--model QNetV2`

## Recommended tuning order

Do not sweep everything at once. Change in this order:

1. `replay_buffer_size`
2. `evaluate_steps`
3. `target_network_update_steps`
4. `final_epsilon`
5. `batch_size`
6. `experience_steps`
7. `learning_rate`

That order fixes the biggest structural issues first.

## Things worth fixing in code before trusting hyperparameter sweeps

### `random_start_proportion` currently has no effect

If you want start-state diversity to matter, wire this into environment construction in `ReplayBuffer.__init__`.

### AMP is disabled

`use_amp = False` even on CUDA. If training is GPU-bound rather than environment-bound, enabling AMP could improve throughput.

### Evaluation is expensive by default

Even if you keep dense logging, sparse evaluation is the better tradeoff for long runs.

## Bottom line

The current defaults are conservative in optimizer scale, but inefficient and somewhat weak in replay design:

- replay is too small for the current batch size
- target updates are too infrequent for the replay churn
- evaluation is far too frequent
- final exploration is probably too high
- gradient clipping is effectively disabled

If I had to bet on the highest-value changes, they would be:

1. increase replay size
2. reduce evaluation frequency
3. update the target network more often
4. lower `final_epsilon`
5. use real gradient clipping
