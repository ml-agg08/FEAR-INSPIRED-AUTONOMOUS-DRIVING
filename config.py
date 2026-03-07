import torch as th
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule
from stable_baselines3 import PPO, DDPG, SAC

algorithm_params = {
    "DDPG": dict(
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300], n_critics=1),
    )
}

states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"]
}

reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
        # FNI-RL Fear hyperparameters
        fear_beta=0.8,            # Weight between cost and uncertainty
        fear_threshold=0.7,       # f₀ — override threshold (relaxed from 0.5)
        fear_lambda=0.5,          # Lagrangian penalty multiplier (reduced from 2.0)
        fear_lookahead_dt=0.75,   # Prediction horizon (seconds)
    ),
     "reward_fn_5_no_early_stop": dict(
         early_stop=False,
         min_speed=20.0,  # km/h
         max_speed=35.0,  # km/h
         target_speed=25.0,  # kmh
         max_distance=3.0,  # Max distance from center before terminating
         max_std_center_lane=0.4,
         max_angle_center_lane=90,
         penalty_reward=-10,
         # FNI-RL Fear hyperparameters
         fear_beta=0.8,
         fear_threshold=0.7,
         fear_lambda=0.5,
         fear_lookahead_dt=0.75,
     ),
    "reward_fn_5_best": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
        # FNI-RL Fear hyperparameters
        fear_beta=0.8,
        fear_threshold=0.7,
        fear_lambda=0.5,
        fear_lookahead_dt=0.75,
    ),
}


_CONFIG_1 = {
    "algorithm": DDPG,
    "algorithm_params": algorithm_params["DDPG"],
    "state": states["3"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG = None


def set_config(config_name):
    global CONFIG
    CONFIG = _CONFIG_1

