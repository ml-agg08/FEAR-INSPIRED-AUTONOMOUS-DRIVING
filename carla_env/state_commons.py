import torch
from torchvision import transforms

import os
from vae.models import VAE
import numpy as np
import gym
from config import CONFIG

from vae.utils.misc import LSIZE
from carla_env.wrappers import vector, get_displacement_vector

torch.cuda.empty_cache()


# ===========================================================================
# Fear-Neuro-Inspired RL (FNI-RL) — Kinematic Fear Heuristic
# ===========================================================================

def compute_fear(speed_kmh, steer, throttle, distance_from_center,
                 angle_to_waypoint, beta=0.8, f0=0.5, dt=0.75,
                 max_speed=35.0, max_distance=3.0, wheelbase=2.5):
    """
    Kinematic fear heuristic  f(s, ã) = β·ĉ(s,ã) + (1-β)·σ̂(s,ã).

    Projects the vehicle forward using a bicycle model and calculates:
      - ĉ (cost): normalized probability of breaching safety limits
      - σ̂ (uncertainty): proxy for epistemic uncertainty (curve + deviation)

    Parameters
    ----------
    speed_kmh : float
        Current speed in km/h.
    steer : float
        Proposed steering in [-1, 1].
    throttle : float
        Proposed throttle in [0, 1].
    distance_from_center : float
        Current lateral distance from the lane center (metres).
    angle_to_waypoint : float
        Signed angle to the next waypoint (radians).
    beta : float
        Weight between cost and uncertainty (default 0.8).
    f0 : float
        Safety threshold (returned but not used for clamping here).
    dt : float
        Look-ahead horizon in seconds (default 0.75).
    max_speed : float
        Speed limit in km/h (default 35).
    max_distance : float
        Lane deviation limit in metres (default 3.0).
    wheelbase : float
        Approximate wheelbase of the ego vehicle in metres.

    Returns
    -------
    fear : float
        Fear score in [0, 1].
    cost : float
        Projected constraint-violation cost in [0, 1].
    uncertainty : float
        Epistemic uncertainty proxy in [0, 1].
    """
    # --- Convert units ---
    speed_ms = speed_kmh / 3.6  # m/s

    # --- Kinematic bicycle projection ---
    # Approximate acceleration from throttle (rough linear mapping)
    accel_ms2 = throttle * 3.0  # ~3 m/s² at full throttle
    projected_speed_ms = speed_ms + accel_ms2 * dt
    projected_speed_kmh = projected_speed_ms * 3.6

    # Steering angle mapped from [-1, 1] to radians (max ~35°)
    max_steer_rad = np.deg2rad(35.0)
    steer_rad = steer * max_steer_rad

    # Lateral displacement during dt (bicycle model approximation)
    if abs(steer_rad) > 1e-4:
        turn_radius = wheelbase / np.tan(abs(steer_rad))
        lateral_shift = turn_radius * (1.0 - np.cos(speed_ms * dt / max(turn_radius, 0.1)))
    else:
        lateral_shift = 0.0

    projected_center_dist = distance_from_center + lateral_shift

    # --- Cost ĉ: worst-case breach probability ---
    speed_cost = float(np.clip((projected_speed_kmh - max_speed) / max_speed, 0.0, 1.0))
    lane_cost = float(np.clip((projected_center_dist - max_distance) / max_distance, 0.0, 1.0))

    # Also penalise being *close* to the limits (soft margin)
    speed_margin = float(np.clip((projected_speed_kmh - max_speed * 0.85) / (max_speed * 0.15), 0.0, 1.0))
    lane_margin = float(np.clip((projected_center_dist - max_distance * 0.6) / (max_distance * 0.4), 0.0, 1.0))

    cost = float(np.clip(max(speed_cost, lane_cost) * 0.6 + max(speed_margin, lane_margin) * 0.4, 0.0, 1.0))

    # --- Uncertainty σ̂: curve sharpness + current deviation ---
    angle_uncertainty = abs(angle_to_waypoint) / np.pi  # [0, 1]
    deviation_uncertainty = distance_from_center / max_distance  # [0, 1+]
    uncertainty = float(np.clip(0.5 * angle_uncertainty + 0.5 * deviation_uncertainty, 0.0, 1.0))

    # --- Fear ---
    fear = float(np.clip(beta * cost + (1.0 - beta) * uncertainty, 0.0, 1.0))

    return fear, cost, uncertainty


def load_vae(vae_dir, latent_size):
    model_dir = os.path.join(vae_dir, 'best.tar')
    model = VAE(latent_size)
    if os.path.exists(model_dir):
        state = torch.load(model_dir)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
        return model
    raise Exception("Error - VAE model does not exist")


def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = preprocess(frame).unsqueeze(0)
    return frame


def create_encode_state_fn(vae, measurements_to_include):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "angle_next_waypoint" in measurements_to_include,
                     "maneuver" in measurements_to_include,
                     "waypoints" in measurements_to_include,
                     "rgb_camera" in measurements_to_include,
                     "seg_camera" in measurements_to_include,
                     "end_wp_vector" in measurements_to_include,
                     "end_wp_fixed" in measurements_to_include,
                     "distance_goal" in measurements_to_include]

    def create_observation_space():
        observation_space = {}
        if vae: observation_space['vae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(LSIZE, ), dtype=np.float32)
        low, high = [], []
        if measure_flags[0]: low.append(-1), high.append(1)
        if measure_flags[1]: low.append(0), high.append(1)
        if measure_flags[2]: low.append(0), high.append(120)
        if measure_flags[3]: low.append(-3.14), high.append(3.14)
        observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        if measure_flags[4]: observation_space['maneuver'] = gym.spaces.Discrete(4)

        if measure_flags[5]: observation_space['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2),
                                                                             dtype=np.float32)

        if measure_flags[6]: observation_space['rgb_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)
        if measure_flags[7]: observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)
        if measure_flags[8]: observation_space['end_wp_vector'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2), dtype=np.float32)
        if measure_flags[9]: observation_space['end_wp_fixed'] = gym.spaces.Box(low=-50, high=50, shape=(1, 2), dtype=np.float32)
        if measure_flags[10]: observation_space['distance_goal'] = gym.spaces.Box(low=0, high=1500, shape=(1, 1), dtype=np.float32)

        return gym.spaces.Dict(observation_space)

    def encode_state(env):
        encoded_state = {}
        if vae:
            with torch.no_grad():
                frame = preprocess_frame(env.observation)
                mu, logvar = vae.encode(frame)
                vae_latent = vae.reparameterize(mu, logvar)[0].cpu().detach().numpy().squeeze()
            encoded_state['vae_latent'] = vae_latent
        vehicle_measures = []
        if measure_flags[0]: vehicle_measures.append(env.vehicle.control.steer)
        if measure_flags[1]: vehicle_measures.append(env.vehicle.control.throttle)
        if measure_flags[2]: vehicle_measures.append(env.vehicle.get_speed())
        if measure_flags[3]: vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint))
        encoded_state['vehicle_measures'] = vehicle_measures
        if measure_flags[4]: encoded_state['maneuver'] = env.current_road_maneuver.value

        if measure_flags[5]:
            next_waypoints_state = env.route_waypoints[env.current_waypoint_index: env.current_waypoint_index + 15]
            waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)

            relative_waypoints = np.zeros((15, 2))
            for i, w_location in enumerate(waypoints):
                relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
            if len(waypoints) < 15:
                start_index = len(waypoints)
                reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
                for i in range(start_index, 15):
                    relative_waypoints[i] = relative_waypoints[i-1] + reference_vector

            encoded_state['waypoints'] = relative_waypoints

        if measure_flags[6]: encoded_state['rgb_camera'] = env.observation
        if measure_flags[7]: encoded_state['seg_camera'] = env.observation
        if measure_flags[8]:
            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_vector'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]
        if measure_flags[9]:
            vehicle_location = vector(env.start_wp.transform.location)
            theta = np.deg2rad(env.start_wp.transform.rotation.yaw)
            end_wp_location = vector(env.end_wp.transform.location)
            encoded_state['end_wp_fixed'] = get_displacement_vector(vehicle_location, end_wp_location, theta)[:2]
        if measure_flags[10]:
            encoded_state['distance_goal'] = [[len(env.route_waypoints) - env.current_waypoint_index]]

        return encoded_state

    def decode_vae_state(z):
        with torch.no_grad():
            sample = torch.tensor(z)
            sample = vae.decode(sample).cpu()
            generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0)) * 255
        return generated_image
    if not vae:
        decode_vae_state = None
    return create_observation_space(), encode_state, decode_vae_state

