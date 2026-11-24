"""
LightZero configurations for the Pacbot game: MuZero, EfficientZero, Gumbel MuZero and
Stochastic MuZero, on the same games (PacbotLightZeroEnv) and, where the algorithms allow, the
same search and return settings as our AlphaZero (train_alphazero.py): discount 0.99, 10-step
value targets. The networks are LightZero's residual networks at full board resolution (no
downsampling: the board is only 28 x 31) without image augmentation, which suits camera frames,
not a grid.
"""

from easydict import EasyDict

from utils import NUM_ACTIONS, OBS_SHAPE

ALGORITHMS = {
    "muzero": "lzero.policy.muzero",
    "efficientzero": "lzero.policy.efficientzero",
    "gumbel_muzero": "lzero.policy.gumbel_muzero",
    "stochastic_muzero": "lightzero_pacbot.stochastic_muzero",  # patched, see that file
}


def make_config(algorithm: str, exp_name: str, *, num_simulations: int = 32, collector_env_num: int = 16,
                evaluator_env_num: int = 11, eval_games=None, batch_size: int = 256, replay_ratio: float = 0.25,
                eval_freq: int = 5_000, cuda: bool = True, speed_sampling: str = "equal_moves",
                max_speed: int = 18, random_start_proportion: float = 0.25, full_game: bool = False,
                num_res_blocks: int = 2, num_channels: int = 64, chance_space_size: int = 32,
                learning_rate: float | None = None, reanalyze_ratio: float = 0.0):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {list(ALGORITHMS)}, got {algorithm!r}")
    model = dict(
        observation_shape=tuple(OBS_SHAPE),
        image_channel=OBS_SHAPE[0],
        frame_stack_num=1,
        action_space_size=NUM_ACTIONS,
        downsample=False,
        model_type="conv",
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        self_supervised_learning_loss=algorithm != "muzero",
        discrete_action_encoding_type="one_hot",
        norm_type="BN",
    )
    policy = dict(
        model=model,
        cuda=cuda,
        env_type="not_board_games",
        action_type="varied_action_space",
        game_segment_length=400,
        use_augmentation=False,
        use_priority=False,
        replay_ratio=replay_ratio,
        update_per_collect=None,
        batch_size=batch_size,
        discount_factor=0.99,
        td_steps=10,
        num_unroll_steps=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=collector_env_num,
        eval_freq=eval_freq,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    )
    if algorithm == "stochastic_muzero":
        model["chance_space_size"] = chance_space_size
        policy.update(optim_type="Adam", piecewise_decay_lr_scheduler=False,
                      learning_rate=learning_rate or 3e-3, gumbel_algo=False, mcts_ctree=True, ssl_loss_weight=2)
    else:
        policy.update(optim_type="SGD", piecewise_decay_lr_scheduler=True, learning_rate=learning_rate or 0.2,
                      target_update_freq=100)
        if algorithm != "muzero":
            policy["ssl_loss_weight"] = 2
    if algorithm == "efficientzero":
        model.update(reward_support_size=101, value_support_size=101, support_scale=50)
    if algorithm == "gumbel_muzero":
        policy["max_num_considered_actions"] = NUM_ACTIONS
    main = dict(
        exp_name=exp_name,
        env=dict(
            stop_value=int(1e9),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=len(eval_games) if eval_games else evaluator_env_num,
            eval_games=eval_games,
            speed_sampling=speed_sampling,
            max_speed=max_speed,
            random_start_proportion=random_start_proportion,
            full_game=full_game,
            max_episode_steps=40_000 if full_game else 10_000,
            manager=dict(shared_memory=False),
        ),
        policy=policy,
    )
    create = dict(
        env=dict(type="pacbot_lightzero", import_names=["lightzero_pacbot.pacbot_env"]),
        env_manager=dict(type="base"),
        policy=dict(type=algorithm, import_names=[ALGORITHMS[algorithm]]),
    )
    return EasyDict(main), EasyDict(create)
