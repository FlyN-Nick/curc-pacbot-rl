"""
LightZero 0.2.0's Stochastic MuZero policy, patched to run with its own evaluator.

The evaluator (lzero/worker/muzero_evaluator.py) passes `timestep=` to every policy's eval
forward, but StochasticMuZeroPolicy._forward_eval doesn't accept extra keyword arguments the way
MuZeroPolicy's does, so evaluation fails with "unexpected keyword argument 'timestep'". The policy
ignores the time step anyway, so the patch drops it. The policy keeps its registered name,
"stochastic_muzero", which LightZero's train_muzero entry checks for.
"""

from lzero.policy.stochastic_muzero import StochasticMuZeroPolicy

_forward_eval = StochasticMuZeroPolicy._forward_eval
_forward_collect = StochasticMuZeroPolicy._forward_collect


def _forward_eval_any_kwargs(self, data, action_mask, to_play=[-1], ready_env_id=None, **kwargs):
    return _forward_eval(self, data, action_mask, to_play, ready_env_id)


def _forward_collect_any_kwargs(self, data, action_mask=None, temperature=1, to_play=[-1], epsilon=0.25,
                                ready_env_id=None, **kwargs):
    return _forward_collect(self, data, action_mask, temperature, to_play, epsilon, ready_env_id)


if not getattr(StochasticMuZeroPolicy, "_pacbot_patched", False):
    StochasticMuZeroPolicy._forward_eval = _forward_eval_any_kwargs
    StochasticMuZeroPolicy._forward_collect = _forward_collect_any_kwargs
    StochasticMuZeroPolicy._pacbot_patched = True
