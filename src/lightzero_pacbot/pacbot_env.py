"""
PacbotLightZeroEnv: our Rust game (pacbot_rs.PacmanGym) as a LightZero environment.

Training games are AlphaZero's (train_alphazero.py): a speed drawn by utils' speed sampling
(e.g. equal moves per speed, up to max_speed), from the normal start or, for
random_start_proportion of them, a random one; one life, or with full_game three lives and
board after board. Evaluation games are the fixed-seed games of eval_az_seeds.py: each
evaluation environment plays its share of the (speed, seed) pairs in eval_games, in order.

The observation is the network input the other algorithms use (17 x 28 x 31), the action mask
the legal moves, and the reward the game's reward (score, death penalty, clear bonus, shaping).
info carries the game's score at the end, as eval_episode_return is the summed reward.
"""

import copy
import random

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

import utils
from speed_sampling import SpeedSampler
from utils import ENV_VERSION, NUM_ACTIONS, OBS_SHAPE, make_gym, reset_env, speed_configuration, training_configuration


@ENV_REGISTRY.register("pacbot_lightzero")
class PacbotLightZeroEnv(BaseEnv):
    config = dict(
        speed_sampling="equal_moves",
        max_speed=18,
        random_start_proportion=0.25,
        full_game=False,
        # Moves before a game is cut off (eval_az_seeds.py: 10,000, or 40,000 for full games)
        max_episode_steps=10_000,
        # Evaluation: play these (speed, seed) games instead of training games
        eval_games=None,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = EasyDict({**self.config, **cfg})
        utils.set_single_model(True)
        utils.set_speed_sampling(self._cfg.speed_sampling)
        utils.set_max_speed(self._cfg.max_speed)
        utils.set_full_game(bool(self._cfg.full_game))
        self._obs_config = utils.replay_buffer_obs_config()
        self._observation_space = gym.spaces.Box(0.0, np.inf, OBS_SHAPE, np.float32)
        self._action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self._reward_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self._env = None
        self._rng = random.Random()
        self._sampler = None
        self._eval_games = list(self._cfg.eval_games) if self._cfg.eval_games else None
        self._eval_index = 0
        self._speed = None

    # ---- LightZero / DI-engine interface ----

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._sampler = SpeedSampler(rng=self._rng)

    def reset(self) -> dict:
        if self._sampler is None:
            self.seed(random.randrange(2**31))
        if self._eval_games is not None:
            speed, seed = self._eval_games[self._eval_index % len(self._eval_games)]
            self._eval_index += 1
            self._env = make_gym(speed_configuration(speed), ENV_VERSION)
            self._env.seed = seed
            self._speed = speed
        else:
            self._speed = self._sampler.draw()
            config = training_configuration(
                self._speed, random_start=self._rng.random() < self._cfg.random_start_proportion)
            if self._env is None:
                self._env = make_gym(config, ENV_VERSION)
            reset_env(self._env, config)
        self._steps = 0
        self._return = 0.0
        return self._observation()

    def step(self, action) -> BaseEnvTimestep:
        action = int(np.asarray(action).reshape(-1)[0])
        if not self._env.action_mask()[action]:
            # The search only picks legal moves; staying put is always legal
            action = 0
        reward, done = self._env.step(action)
        self._steps += 1
        self._return += reward
        if self._steps >= self._cfg.max_episode_steps:
            done = True
        info = {}
        if done:
            if self._sampler is not None and self._eval_games is None:
                self._sampler.record(self._speed, self._steps)
            env = self._env
            info = {"eval_episode_return": self._return, "score": env.score(), "steps": self._steps,
                    "speed": self._speed, "lives": env.lives(), "boards": env.level() - 1}
        return BaseEnvTimestep(self._observation(), np.float32(reward), done, info)

    def _observation(self) -> dict:
        return {
            "observation": self._env.obs_numpy(self._obs_config).astype(np.float32),
            "action_mask": np.array(self._env.action_mask(), dtype=np.int8),
            "to_play": -1,
            "timestep": self._steps,
        }

    def close(self) -> None:
        self._env = None

    def random_action(self) -> np.ndarray:
        legal = np.flatnonzero(self._env.action_mask())
        return np.array([self._rng.choice(legal.tolist())], dtype=np.int64)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def __repr__(self) -> str:
        return "PacbotLightZeroEnv"

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        n = cfg.pop("collector_env_num")
        cfg = copy.deepcopy(cfg)
        cfg.pop("evaluator_env_num", None)
        return [cfg for _ in range(n)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        n = cfg.pop("evaluator_env_num")
        cfg = copy.deepcopy(cfg)
        cfg.pop("collector_env_num", None)
        games = list(cfg.get("eval_games") or [(speed, 1) for speed in range(4, 15)])
        # Each evaluation environment plays its share of the games, in order
        return [{**cfg, "eval_games": games[i::n]} for i in range(n)]
