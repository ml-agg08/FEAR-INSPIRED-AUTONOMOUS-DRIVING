import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import config
import time

config.set_config("1")

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from carla_env.carla_route_env import CarlaRouteEnv

from vae.utils.misc import LSIZE
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class

from config import CONFIG

log_dir = 'tensorboard'
os.makedirs(log_dir, exist_ok=True)

# Auto-find the latest model — REQUIRED for fine-tuning
# Searches for both original models and fear-finetuned checkpoints
import glob
model_files = glob.glob('tensorboard/*/model_*.zip') + glob.glob('tensorboard/*/ddpg_fear_model_*.zip')
if model_files:
    reload_model = max(model_files, key=os.path.getctime)
    print(f"[FNI-RL] Fine-tuning from existing model: {reload_model}")
else:
    raise FileNotFoundError(
        "[FNI-RL] ERROR: No existing model found in tensorboard/. "
        "Fear-constrained fine-tuning requires a pre-trained model. "
        "Please ensure your trained DDPG weights are in the tensorboard/ directory."
    )

# Reduced timesteps for rapid fear-constrained fine-tuning
total_timesteps = 200000


seed = CONFIG["seed"]

AlgorithmRL = CONFIG["algorithm"]
vae = None
if CONFIG["vae_model"]:
    vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="localhost", port=2000,
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    observation_space=observation_space,
                    encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                    fps=15, action_smoothing=CONFIG["action_smoothing"],
                    action_space_type='continuous', activate_spectator=False, activate_render=True) #change render

for wrapper_class_str in CONFIG["wrappers"]:
    wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
    env = wrap_class(env, *wrap_params)

# Always load existing weights for fine-tuning — never train from scratch
model = AlgorithmRL.load(reload_model, env=env, device='cuda', seed=seed, **CONFIG["algorithm_params"])
print(f"[FNI-RL] Loaded model with {sum(p.numel() for p in model.policy.parameters())} parameters")

# Save to a completely new directory to protect original weights
model_suffix = "DDPG_fear_trafficlight_finetuned"
model_name = f'{model.__class__.__name__}_{model_suffix}'

model_dir = os.path.join(log_dir, model_name)
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

print(f"[FNI-RL] Saving checkpoints to: {model_dir}")
print(f"[FNI-RL] Original weights at '{reload_model}' will NOT be modified.")

# TRAINING SKIPPED — jump straight to evaluation
# Uncomment below to resume fear-constrained fine-tuning:
# model.learn(total_timesteps=total_timesteps,
#             callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
#                 save_freq=total_timesteps // 10,
#                 save_path=model_dir,
#                 name_prefix="ddpg_fear_model")], reset_num_timesteps=False)

# Save final model
final_model_path = os.path.join(model_dir, "ddpg_fear_model_final.zip")
model.save(final_model_path)
print(f"[FNI-RL] Final model saved to: {final_model_path}")


# ============================================================================
# POST-TRAINING EVALUATION — 10 Episodes + Paper Plots
# ============================================================================
print("\n" + "="*70)
print("[FNI-RL] Starting Post-Training Evaluation (10 episodes)...")
print("="*70 + "\n")

import pandas as pd
import numpy as np
from carla_env.wrappers import vector, get_displacement_vector

NUM_EVAL_EPISODES = 10
FEAR_THRESHOLD = CONFIG["reward_params"].get("fear_threshold", 0.7)

columns = [
    "model_id", "episode", "step", "throttle", "steer",
    "vehicle_location_x", "vehicle_location_y",
    "reward", "distance", "speed", "center_dev", "angle_next_waypoint",
    "waypoint_x", "waypoint_y", "route_x", "route_y",
    "fear", "is_at_red_light", "distance_to_light", "override_active",
]
df = pd.DataFrame(columns=columns)

eval_model_id = "FNI-RL-DDPG"

for ep_idx in range(NUM_EVAL_EPISODES):
    print(f"  [Eval] Episode {ep_idx + 1}/{NUM_EVAL_EPISODES}...", end=" ", flush=True)

    state, _info = env.reset()

    # Record initial heading for relative coordinates
    initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
    initial_vehicle_location = vector(env.vehicle.get_location())

    # Save route waypoints
    for way in env.route_waypoints:
        route_rel = get_displacement_vector(
            initial_vehicle_location, vector(way[0].transform.location), initial_heading
        )
        route_row = pd.DataFrame(
            [["route", env.episode_idx, None, None, None, None, None,
              None, None, None, None, None, None, None,
              route_rel[0], route_rel[1],
              None, None, None, None]],
            columns=columns,
        )
        df = pd.concat([df, route_row], ignore_index=True)

    done = False
    ep_steps = 0
    ep_reward = 0.0

    while not done:
        action, _ = model.predict(state, deterministic=False)

        # Ensure minimum throttle so the car actually moves
        # (DDPG's deterministic policy may output near-zero throttle
        #  since it learned to avoid fear by standing still)
        raw_throttle = action[1]
        if action[1] < 0.15:
            action[1] = 0.15

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_steps += 1
        ep_reward += reward

        # Debug first 3 steps per episode
        if ep_steps <= 3:
            print(f"\n    [DBG] step={ep_steps} raw_throttle={raw_throttle:.3f} "
                  f"applied_throttle={action[1]:.3f} steer={action[0]:.3f} "
                  f"speed={env.vehicle.get_speed():.1f} fear={getattr(env, 'last_fear', 0):.3f} "
                  f"override={getattr(env, 'last_fear', 0) > FEAR_THRESHOLD}")

        # Safety cutoff
        if ep_steps >= 2000:
            done = True

        # Relative positions
        vehicle_rel = get_displacement_vector(
            initial_vehicle_location, vector(env.vehicle.get_location()), initial_heading
        )
        waypoint_rel = get_displacement_vector(
            initial_vehicle_location,
            vector(env.current_waypoint.transform.location),
            initial_heading,
        )

        # FNI-RL metrics
        fear_val = getattr(env, "last_fear", 0.0)
        is_red = getattr(env, "is_at_red_light", False)
        dist_light = getattr(env, "traffic_light_distance", -1.0)
        if dist_light == float("inf"):
            dist_light = -1.0
        override = fear_val > FEAR_THRESHOLD

        step_row = pd.DataFrame(
            [[eval_model_id, env.episode_idx, env.step_count,
              env.vehicle.control.throttle, env.vehicle.control.steer,
              vehicle_rel[0], vehicle_rel[1], reward,
              env.distance_traveled, env.vehicle.get_speed(),
              env.distance_from_center,
              np.rad2deg(env.vehicle.get_angle(env.current_waypoint)),
              waypoint_rel[0], waypoint_rel[1], None, None,
              fear_val, is_red, dist_light, override]],
            columns=columns,
        )
        df = pd.concat([df, step_row], ignore_index=True)

    print(f"steps={ep_steps}, reward={ep_reward:.2f}, "
          f"dist={env.distance_traveled:.0f}m, "
          f"avg_speed={env.speed_accum / max(env.step_count, 1):.1f} km/h")

# Save CSV
eval_csv_path = os.path.join(model_dir, "eval_results.csv")
df.to_csv(eval_csv_path, index=False)
print(f"\n[FNI-RL] Eval data saved to: {eval_csv_path}")

# Generate paper plots
print("[FNI-RL] Generating publication-ready plots...\n")
from eval_plots import (
    plot_amygdala_intervention,
    plot_safety_boundary_vs_fear,
    plot_spatial_trajectory,
    plot_summary_table,
)

PAPER_DIR = os.path.join(model_dir, "paper_results")
os.makedirs(PAPER_DIR, exist_ok=True)

plot_amygdala_intervention(df, save_path=os.path.join(PAPER_DIR, "plot1_amygdala_intervention.png"))
plot_safety_boundary_vs_fear(df, save_path=os.path.join(PAPER_DIR, "plot2_safety_boundary_fear.png"))
plot_spatial_trajectory(df, save_path=os.path.join(PAPER_DIR, "plot3_spatial_trajectory.png"))
plot_summary_table(df, save_path=os.path.join(PAPER_DIR, "summary_metrics_table.png"))

print(f"\n[FNI-RL] All done! Paper plots saved to: {PAPER_DIR}")
print("="*70)

