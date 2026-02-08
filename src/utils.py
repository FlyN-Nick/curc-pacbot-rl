import torch
from typing import TypeVar

from policies import Policy

from pacbot_rs import PacmanGym, PacmanGymConfiguration
import models
from policies import MaxQPolicy
import safetensors.torch

DEFAULT_GYM_CONFIGURATION = PacmanGymConfiguration({})
DETERMINISTIC_START_CONFIGURATION = PacmanGymConfiguration({"random_start": False, "random_ticks": False})
OBS_SHAPE = PacmanGym(DEFAULT_GYM_CONFIGURATION).obs_numpy(DEFAULT_GYM_CONFIGURATION).shape

REPLAY_BUFFER_OBS_GYM_CONFIGURATION = PacmanGymConfiguration({"random_start": False, "random_ticks": True, "obs_ignore_super_pellets": True})
def replay_buffer_action_mask(env: PacmanGym) -> list[bool]:
    return env.purgatory_action_mask()

P = TypeVar("P", bound=Policy)

# N/E/S/W and stay
NUM_ACTIONS = 5

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

def load_qnetv2_from_file(filename: str, device) -> P:
    if not filename.endswith(".safetensors"):
        print("WARNING: load_model_from_file expects `.safetensors` files")

    q_net_old = models.QNetV2(OBS_SHAPE, NUM_ACTIONS).to(device)
    q_net_old.load_state_dict(safetensors.torch.load_file(filename))
    q_net_old.eval()
    policy = MaxQPolicy(q_net_old)

    return policy

LOADED_POLICIES = {}

def safetensors_qnet(filename: str, force_reload: bool=False, device="cpu") -> Policy:
    """
    Safe to call in a loop, uses cached policies
    """
    if not force_reload and filename in LOADED_POLICIES:
        return LOADED_POLICIES[filename]
    
    LOADED_POLICIES[filename] = load_qnetv2_from_file(filename, device)
    return LOADED_POLICIES[filename]

def _step_always_with(env: PacmanGym, policy: Policy, config: PacmanGymConfiguration, action_mask: list[bool], device) -> tuple[int, bool]:
    obs = torch.from_numpy(env.obs_numpy(config)).to(device).unsqueeze(0)
    action_mask = torch.tensor(action_mask, device=device).unsqueeze(0)
    return env.step(policy(obs, action_mask).item())

def reset_env(env: PacmanGym, config: PacmanGymConfiguration) -> None:
    env.reset(config)

SUPER_PELLET_MODE_CONFIGUARTION = PacmanGymConfiguration({})
PURGATORY_MODE_CONFIGUARTION = PacmanGymConfiguration({"obs_ignore_super_pellets": True})

def step_env_once(env: PacmanGym, policy: Policy, device) -> tuple[int, bool]:
    # purgatory mode is when not all ghosts are free and all ghosts are not frightened 
    if not env.are_ghosts_close() and env.all_ghosts_not_frightened():
        return _step_always_with(env, policy, PURGATORY_MODE_CONFIGUARTION, env.purgatory_action_mask(), device)
    else:
        # temporary - end game when super pellet mode hits
        #return 0, True
        return _step_always_with(env, safetensors_qnet("checkpoints_known/superpelletmode.safetensors", device=device), SUPER_PELLET_MODE_CONFIGUARTION, env.action_mask(), device)

def step_while_not_policy(env: PacmanGym, device, reset_if_necessary: bool = True):
    # purgatory mode is when all ghosts are free and not frightened 
    while env.are_ghosts_close() or not env.all_ghosts_not_frightened():
        # temporary - end game when super pellet mode hits
        # return reset_env(env, DEFAULT_GYM_CONFIGURATION)
         _, done = _step_always_with(env, safetensors_qnet("checkpoints_known/superpelletmode.safetensors", device=device), SUPER_PELLET_MODE_CONFIGUARTION, env.action_mask(), device)
         if done:
             if reset_if_necessary:
                 reset_env(env, DEFAULT_GYM_CONFIGURATION)
             else:
                 return

def step_env_until_done(env: PacmanGym, policy: Policy, device, max_steps: int = 1000) -> tuple[bool, int]:
    for step_num in range(1, max_steps + 1):
        _, done = step_env_once(env, policy, device)
        if done:
            break
    return (env.is_done(), step_num)

def select_device(preferred: str | None = None) -> torch.device:
    """Select a torch.device with the following preference order:
    1. If `preferred` is provided, return torch.device(preferred).
    2. If MPS is available, return 'mps' (Apple Metal on Apple Silicon).
    3. If CUDA is available, return 'cuda'.
    4. Otherwise return 'cpu'.

    This helper centralizes device selection so the rest of the codebase can
    consistently prefer MPS on macOS with Apple Silicon.
    """
    if preferred:
        return torch.device(preferred)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(obj, device: torch.device):
    """Recursively move tensors (or models) to `device`.

    - torch.Tensor -> tensor.to(device)
    - list/tuple -> same container type with elements moved
    - dict -> dict with values moved
    - nn.Module -> module.to(device)
    - other types are returned as-is
    """
    # Import here to avoid circular imports in some execution orders.
    import torch.nn as nn

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj
