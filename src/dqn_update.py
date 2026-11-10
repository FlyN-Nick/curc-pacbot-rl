"""
The DQN update step (double DQN targets, MSE loss, gradient clipping, optimizer step), run
eagerly or replayed as a CUDA graph.

A CUDA graph records the whole update once, including sampling the batch out of the replay
storage, and replays it with a single launch, instead of Python launching each of its
hundreds of small GPU operations every iteration.
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from replay_storage import Batch, BatchIndices, ReplayStorage


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    # No cast cache: copies cast under no_grad (the target passes) can't carry gradients
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False)


def dqn_update(
    q_net: torch.nn.Module,
    target_q_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    reward_scale: float,
    discount_factor: float,
    grad_clip_norm: float,
    bf16: str = "none",
    zero_grad: bool = True,
) -> torch.Tensor:
    """
    One update on `batch`. Returns, as one float32 tensor: loss, gradient norm (before
    clipping), mean of the max predicted Q values and mean target Q value (both unscaled).
    """
    device = batch.obs.device
    reward = (batch.reward * reward_scale).float()

    # Double DQN target: the online network picks the next action, the target network rates it.
    # masked_fill and gather instead of boolean-mask assignment and range indexing: the same
    # values, without making the CPU wait for the GPU. For a finished game, next_obs is the
    # reset game's observation and its mask is all False; its value is masked out.
    with torch.no_grad(), _autocast(device, bf16 == "all"):
        next_q_values = target_q_net(batch.next_obs)
        next_q_values = next_q_values.masked_fill(~batch.next_action_mask, -torch.inf)
        online_next_q_values = q_net(batch.next_obs)
        online_next_q_values = online_next_q_values.masked_fill(~batch.next_action_mask, -torch.inf)
        next_actions = online_next_q_values.argmax(dim=1)
        returns = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        discounted_returns = (discount_factor * returns).masked_fill(batch.done, 0.0)
        target_q_values = reward + discounted_returns

    if zero_grad:
        optimizer.zero_grad()
    with _autocast(device, bf16 in ("update", "all")):
        all_predicted_q_values = q_net(batch.obs)
        predicted_q_values = all_predicted_q_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(predicted_q_values.float(), target_q_values.float())
    loss.backward()
    # A non-finite norm is caught later (error_if_nonfinite would make the CPU wait for the GPU)
    grad_norm = torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=grad_clip_norm)
    optimizer.step()

    with torch.no_grad():
        return torch.stack([
            loss.detach().float(),
            grad_norm.detach().float(),
            all_predicted_q_values.detach().float().amax(dim=1).mean() / reward_scale,
            target_q_values.float().mean() / reward_scale,
        ])


def set_capturable(optimizer: torch.optim.Optimizer, capturable: bool) -> None:
    """
    Switches an Adam optimizer (and its state, e.g. after load_state_dict) to or from the
    GPU-capturable form CUDA graphs need. Checkpoints keep the flag they were saved with, so
    this is applied after loading either way.
    """
    for group in optimizer.param_groups:
        group["capturable"] = capturable
        for param in group["params"]:
            state = optimizer.state.get(param)
            if state and "step" in state:
                step = state["step"]
                state["step"] = step.to(param.device if capturable else "cpu", dtype=torch.float32)


class EagerUpdate:
    """Runs dqn_update() directly, each iteration."""

    def __init__(self, q_net, target_q_net, optimizer, storage: ReplayStorage, batch_size: int, **kwargs):
        self.args = (q_net, target_q_net, optimizer)
        self.storage = storage
        self.batch_size = batch_size
        self.kwargs = kwargs

    def run(self, indices: BatchIndices) -> torch.Tensor:
        return dqn_update(*self.args, self.storage.gather(indices), **self.kwargs)


class GraphedUpdate:
    """
    Captures dqn_update(), batch sampling from `storage` included, as a CUDA graph, and replays
    it. The optimizer must be capturable (set_capturable). The graph keeps using the same
    tensors, so the networks' parameters must only be changed in place (e.g. load_state_dict
    or copy_, not replacing them), and storage must already hold `batch_size` transitions.

    Capturing needs a few warm-up updates, which would change the network; its parameters and
    the optimizer's state are restored afterwards, so the first replay is the first update.
    """

    def __init__(self, q_net, target_q_net, optimizer, storage: ReplayStorage, batch_size: int,
                 warmup: int = 3, **kwargs):
        self.storage = storage
        device = storage.device
        self.indices = storage.sample_indices(batch_size)  # placeholder indices, overwritten

        def update(zero_grad: bool):
            return dqn_update(q_net, target_q_net, optimizer, storage.gather(self.indices),
                              zero_grad=zero_grad, **kwargs)

        params = [p.detach().clone() for p in q_net.parameters()]
        saved_state = {
            p: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
            for p, s in optimizer.state.items()
        }
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(warmup):
                update(zero_grad=True)
        torch.cuda.current_stream(device).wait_stream(side)

        # Undo the warm-up updates, in place: the graph will use these same tensors
        with torch.no_grad():
            for p, saved in zip(q_net.parameters(), params):
                p.copy_(saved)
            for p, state in optimizer.state.items():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        if p in saved_state:
                            value.copy_(saved_state[p][key])
                        else:  # state the warm-up created: back to Adam's fresh state
                            value.zero_()
                    elif p in saved_state:
                        state[key] = saved_state[p][key]

        optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.metrics = update(zero_grad=False)

    def run(self, indices: BatchIndices) -> torch.Tensor:
        for dst, src in zip(self.indices, indices):
            dst.copy_(src)
        self.graph.replay()
        return self.metrics
