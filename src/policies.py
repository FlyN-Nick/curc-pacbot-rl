import random
from typing import Callable, Generic, TypeVar

import torch
from torch.distributions import Categorical

from models import NetV2, QNet


# A Policy takes a batch of observations and action masks and returns a batch of actions.
Policy = Callable[[torch.FloatTensor, torch.BoolTensor], torch.IntTensor]

P = TypeVar("P", bound=Policy)


class EpsilonGreedy(Generic[P]):
    """ϵ-greedy policy that wraps another policy."""

    def __init__(self, original_policy: P, num_actions: int, epsilon: float) -> None:
        self.original_policy = original_policy
        self.num_actions = num_actions
        self.epsilon = epsilon

    def __call__(self, obs: torch.FloatTensor, action_masks: torch.BoolTensor) -> torch.IntTensor:
        batch_size = obs.shape[0]
        # Determine device from observations and ensure all tensors live there.
        device = obs.device

        # Sample uniformly-random valid actions (use Python lists for sampling).
        actions = [
            random.choice([i for i, valid in enumerate(action_mask.tolist()) if valid])
            for action_mask in action_masks
        ]
        # Create the tensor on the same device as `obs`.
        actions = torch.tensor(actions, device=device, dtype=torch.int64)

        # Generate a mask that will determine which actions will be greedy (on same device).
        greedy_mask = (torch.rand(batch_size, device=device) > self.epsilon)

        # If any indices are greedy, replace those actions with the wrapped policy's choices.
        if greedy_mask.any():
            greedy_actions = self.original_policy(obs[greedy_mask], action_masks[greedy_mask])
            # Ensure dtype and device match before assignment.
            greedy_actions = greedy_actions.to(device=device)
            actions[greedy_mask] = greedy_actions

        return actions


class MaxQPolicy:
    """
    A policy that selects the action with the highest Q(s, a) value, as predicted by a Q network.

    The Q network is expected to take a (batched) observation tensor and return a (batched) vector
    of Q values, with shape (batch_size, num_actions).
    """

    def __init__(self, q_net: QNet) -> None:
        self.q_net = q_net

    @torch.no_grad()
    def __call__(self, obs: torch.FloatTensor, action_masks: torch.BoolTensor) -> torch.IntTensor:
        action_values = self.q_net(obs)
        action_values[~action_masks] = -torch.inf
        return action_values.argmax(dim=1)


class PNetPolicy:
    """
    A policy that selects an action according to the categorical distribution predicted by a policy
    network.

    The policy network is expected to take a (batched) observation tensor and return a (batched) vector
    of action logits, with shape (batch_size, num_actions).
    """

    def __init__(self, policy_net: NetV2) -> None:
        self.policy_net = policy_net

    def __call__(self, obs: torch.FloatTensor, action_masks: torch.BoolTensor) -> torch.IntTensor:
        actions, _ = self.action_and_entropy(obs, action_masks)
        return actions

    @torch.no_grad()
    def action_and_entropy(
        self, obs: torch.FloatTensor, action_masks: torch.BoolTensor
    ) -> tuple[torch.IntTensor, torch.FloatTensor]:
        action_logits = self.policy_net(obs)
        action_logits[~action_masks] = -torch.inf
        action_dist = Categorical(logits=action_logits)
        return action_dist.sample(), action_dist.entropy()
