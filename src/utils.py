import os
from pathlib import Path

import numpy as np
import torch
from typing import TypeVar

from policies import Policy

from pacbot_rs import PacmanGym, PacmanGymConfiguration
import models
from policies import MaxQPolicy
import safetensors.torch

# Version of the environment rules (see LATEST_ENV_VERSION in pacbot_rs/src/env.rs).
# Checkpoints record the version they were trained with as "env_version";
# checkpoints saved before that field existed are version 1.
ENV_VERSION: int = PacmanGym.LATEST_ENV_VERSION


# Single-model mode: the trained network plays the whole game (super pellets included, normal
# action mask), instead of handing over to the fixed super pellet model outside purgatory.
# Kept in the environment so spawned actor and evaluation processes inherit it.
_SINGLE_MODEL_ENV = "PACBOT_SINGLE_MODEL"


def set_single_model(enabled: bool) -> None:
    os.environ[_SINGLE_MODEL_ENV] = "1" if enabled else "0"


def single_model() -> bool:
    return os.environ.get(_SINGLE_MODEL_ENV) == "1"


# Ghost-next observations: channels 8-11 hold each ghost's next square instead of a copy of
# its current square (PacmanGym.ghost_next_obs). Kept in the environment like single-model mode.
_GHOST_NEXT_OBS_ENV = "PACBOT_GHOST_NEXT_OBS"


def set_ghost_next_obs(enabled: bool) -> None:
    os.environ[_GHOST_NEXT_OBS_ENV] = "1" if enabled else "0"


def ghost_next_obs() -> bool:
    return os.environ.get(_GHOST_NEXT_OBS_ENV) == "1"


# Full games (PacmanGym.full_game): three lives and board after board, as in competition,
# instead of ending at the first lost life or cleared board. Kept in the environment like
# single-model mode, so every game make_gym builds (training, search, evaluation) plays them.
_FULL_GAME_ENV = "PACBOT_FULL_GAME"


def set_full_game(enabled: bool) -> None:
    os.environ[_FULL_GAME_ENV] = "1" if enabled else "0"


def full_game() -> bool:
    return os.environ.get(_FULL_GAME_ENV) == "1"


def parse_bool(text: str) -> bool:
    """An argparse type for true/false flags (bool("False") would be True)."""
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected true or false, got {text!r}")


def checkpoint_full_game(loaded) -> bool:
    """Whether a checkpoint was trained on full games (False for checkpoints from before them)."""
    if not isinstance(loaded, dict):
        return False
    return bool(loaded.get("full_game", (loaded.get("config") or {}).get("full_game", False)))


# Training speeds: Pacman moves every 4..max_speed game ticks (the ghosts every 12), each
# game's speed drawn by speed_sampling.SpeedSampler: "uniform" (every speed in as many games)
# or "equal_moves" (every speed in about as many training moves: slow games last several
# times longer, so uniform games give the slow speeds most of the data). Kept in the
# environment like single-model mode, so actor processes inherit them.
_MAX_SPEED_ENV = "PACBOT_MAX_SPEED"
_SPEED_SAMPLING_ENV = "PACBOT_SPEED_SAMPLING"
MIN_SPEED = 4
SPEED_SAMPLING_MODES = ("uniform", "equal_moves")


def set_max_speed(ticks_per_step: int) -> None:
    if ticks_per_step < MIN_SPEED:
        raise ValueError(f"max_speed must be at least {MIN_SPEED}, got {ticks_per_step}")
    os.environ[_MAX_SPEED_ENV] = str(ticks_per_step)


def max_speed() -> int:
    return int(os.environ.get(_MAX_SPEED_ENV, "14"))


def training_speeds() -> tuple[int, ...]:
    """The speeds (game ticks per Pacman move) games are trained and evaluated at."""
    return tuple(range(MIN_SPEED, max_speed() + 1))


def set_speed_sampling(mode: str) -> None:
    if mode not in SPEED_SAMPLING_MODES:
        raise ValueError(f"speed_sampling must be one of {SPEED_SAMPLING_MODES}, got {mode!r}")
    os.environ[_SPEED_SAMPLING_ENV] = mode


def speed_sampling() -> str:
    return os.environ.get(_SPEED_SAMPLING_ENV, "uniform")


def checkpoint_ghost_next_obs(loaded) -> bool:
    """Whether a loaded checkpoint was trained with ghost-next observations."""
    return isinstance(loaded, dict) and bool(loaded.get("ghost_next_obs", False))


def checkpoint_single_model(loaded) -> bool:
    """Whether a loaded checkpoint was trained in single-model mode."""
    return isinstance(loaded, dict) and bool(loaded.get("single_model", False))


def checkpoint_env_version(loaded) -> int:
    """The environment version a loaded checkpoint was trained with."""
    if isinstance(loaded, dict):
        return int(loaded.get("env_version", 1))
    return 1  # legacy full-module checkpoint


def make_gym(config: "PacmanGymConfiguration", env_version: int, seed: int | None = None) -> PacmanGym:
    """
    A PacmanGym following the rules of the given environment version. With `seed`, the game
    is repeatable (its start and its random choices come from the seed).
    """
    env = PacmanGym(config, seed=seed)
    env.env_version = env_version
    env.ghost_next_obs = ghost_next_obs()
    env.full_game = full_game()
    return env


# The fixed model that plays outside purgatory (see step_env_once), found relative to this
# file so training, evaluation and tests work from any directory.
SUPER_PELLET_MODEL_PATH = str(Path(__file__).resolve().parent / "checkpoints_known" / "superpelletmode.safetensors")

DEFAULT_GYM_CONFIGURATION = PacmanGymConfiguration({})
DETERMINISTIC_START_CONFIGURATION = PacmanGymConfiguration({"random_start": False, "random_ticks": False})

# A game from the normal start is deterministic (the ghosts have no randomness): the only
# thing that varies between training games is Pacman's speed, in game ticks per action. These
# are the default speeds (training_speeds() with the default max_speed, 14), which reports
# keep using so their results stay comparable. 8 is the normal speed
# (DETERMINISTIC_START_CONFIGURATION).
TRAINING_SPEEDS = tuple(range(4, 15))
NORMAL_SPEED = 8


def speed_configuration(ticks_per_step: int) -> PacmanGymConfiguration:
    """The normal start, with Pacman moving every `ticks_per_step` game ticks."""
    return PacmanGymConfiguration({
        "random_start": False, "random_ticks": True,
        "random_ticks_per_step_min": ticks_per_step, "random_ticks_per_step_max": ticks_per_step,
    })
OBS_SHAPE = PacmanGym(DEFAULT_GYM_CONFIGURATION).obs_numpy(DEFAULT_GYM_CONFIGURATION).shape

REPLAY_BUFFER_OBS_GYM_CONFIGURATION = PacmanGymConfiguration({"random_start": False, "random_ticks": True, "obs_ignore_super_pellets": True})
SINGLE_MODEL_REPLAY_BUFFER_OBS_GYM_CONFIGURATION = PacmanGymConfiguration({"random_start": False, "random_ticks": True})
def training_configuration(ticks_per_step: int, random_start: bool = False,
                           random_pellet_removal: bool = False) -> PacmanGymConfiguration:
    """A training game at the given speed: from the normal start, or from a random one (see
    random_start_configuration)."""
    return PacmanGymConfiguration({
        "random_start": random_start, "random_ticks": True, "random_pellet_removal": random_pellet_removal,
        "random_ticks_per_step_min": ticks_per_step, "random_ticks_per_step_max": ticks_per_step,
    })


def random_start_configuration(random_pellet_removal: bool = False) -> PacmanGymConfiguration:
    """
    Training games from a random Pacman position: often on a half-cleared board, or with
    `random_pellet_removal`, with a random share (0-98%) of the pellets removed.
    """
    return PacmanGymConfiguration({
        "random_start": True, "random_ticks": True, "random_pellet_removal": random_pellet_removal,
    })
def replay_buffer_obs_config() -> PacmanGymConfiguration:
    return SINGLE_MODEL_REPLAY_BUFFER_OBS_GYM_CONFIGURATION if single_model() else REPLAY_BUFFER_OBS_GYM_CONFIGURATION
def replay_buffer_action_mask(env: PacmanGym) -> list[bool]:
    allowed = env.action_mask() if single_model() else env.purgatory_action_mask()
    return env.safety_action_mask(allowed)

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
    cache_key = (filename, str(device))
    if not force_reload and cache_key in LOADED_POLICIES:
        return LOADED_POLICIES[cache_key]
    
    LOADED_POLICIES[cache_key] = load_qnetv2_from_file(filename, device)
    return LOADED_POLICIES[cache_key]

def _step_always_with(env: PacmanGym, policy: Policy, config: PacmanGymConfiguration, action_mask: list[bool], device) -> tuple[int, bool]:
    obs = torch.from_numpy(env.obs_numpy(config)).to(device).unsqueeze(0)
    action_mask = torch.tensor(env.safety_action_mask(action_mask), device=device).unsqueeze(0)
    return env.step(policy(obs, action_mask).item())

def reset_env(env: PacmanGym, config: PacmanGymConfiguration) -> None:
    env.reset(config)

SUPER_PELLET_MODE_CONFIGUARTION = PacmanGymConfiguration({})
PURGATORY_MODE_CONFIGUARTION = PacmanGymConfiguration({"obs_ignore_super_pellets": True})

def step_env_once(env: PacmanGym, policy: Policy, device) -> tuple[int, bool]:
    if single_model():
        return _step_always_with(env, policy, SUPER_PELLET_MODE_CONFIGUARTION, env.action_mask(), device)
    # purgatory mode is when not all ghosts are free and all ghosts are not frightened 
    if not env.are_ghosts_close() and env.all_ghosts_not_frightened():
        return _step_always_with(env, policy, PURGATORY_MODE_CONFIGUARTION, env.purgatory_action_mask(), device)
    else:
        # temporary - end game when super pellet mode hits
        #return 0, True
        return _step_always_with(env, safetensors_qnet(SUPER_PELLET_MODEL_PATH, device=device), SUPER_PELLET_MODE_CONFIGUARTION, env.action_mask(), device)

def step_while_not_policy(env: PacmanGym, device, reset_if_necessary: bool = True):
    if single_model():
        return
    # purgatory mode is when all ghosts are free and not frightened 
    while env.are_ghosts_close() or not env.all_ghosts_not_frightened():
        # temporary - end game when super pellet mode hits
        # return reset_env(env, DEFAULT_GYM_CONFIGURATION)
         _, done = _step_always_with(env, safetensors_qnet(SUPER_PELLET_MODEL_PATH, device=device), SUPER_PELLET_MODE_CONFIGUARTION, env.action_mask(), device)
         if done:
             if reset_if_necessary:
                 reset_env(env, DEFAULT_GYM_CONFIGURATION)
             else:
                 return



def in_purgatory(env: PacmanGym) -> bool:
    """Whether the policy being trained acts in this state (see step_env_once)."""
    return single_model() or not env.are_ghosts_close() and env.all_ghosts_not_frightened()


@torch.no_grad()
def step_envs_while_not_policy(envs: list[PacmanGym], device) -> None:
    """
    step_while_not_policy(env, device, reset_if_necessary=True) for many environments at once:
    the super pellet model plays every environment that is outside purgatory, resetting those
    that finish, until all of them are back in purgatory. Each round runs the model once on a
    batch of all environments that still need it, instead of once per environment and step.
    """
    active = [env for env in envs if not in_purgatory(env)]
    if not active:
        return
    policy = safetensors_qnet(SUPER_PELLET_MODEL_PATH, device=device)
    while active:
        obs = torch.from_numpy(
            np.stack([env.obs_numpy(SUPER_PELLET_MODE_CONFIGUARTION) for env in active])
        ).to(device)
        masks = torch.from_numpy(np.array([env.safety_action_mask(env.action_mask()) for env in active])).to(device)
        for env, action in zip(active, policy(obs, masks).tolist()):
            _, done = env.step(action)
            if done:
                reset_env(env, DEFAULT_GYM_CONFIGURATION)
        active = [env for env in active if not in_purgatory(env)]


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
