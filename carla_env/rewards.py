import numpy as np
from config import CONFIG
import carla
from carla_env.state_commons import compute_fear

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}

emotion_weights = {
    "C": 0.5,    # Calmness weight
    "Rt": -0.3,  # Restlessness weight (penalizes erratic driving)
    "U": -0.4,    # Urgency weight
    "L": -0.2    # Laziness weight (penalizes idling)
}

def create_reward_fn(reward_fn):
    """
    Wraps a reward function to handle terminal conditions and logging.
    """
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            # Stop if speed is less than 1.0 km/h after the first 5s of an episode
            global low_speed_timer
            low_speed_timer += 1.0 / env.fps
            speed = env.vehicle.get_speed()
            # Removed vehicle stopped termination - episodes should only end on real failures
            # if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1 and not getattr(env, 'is_at_red_light', False):
            #     env.terminal_state = True
            #     terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"

            # Stop if collision with another vehicle
            if env.collision_detected:
                env.terminal_state = True
                terminal_reason = "Collision detected"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)  # Pass only the environment
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func

# --- Base Reward Functions ---

def calculate_speed_reward(env):
    speed_kmh = env.vehicle.get_speed()
    min_speed = CONFIG["reward_params"]["min_speed"]
    target_speed = CONFIG["reward_params"]["target_speed"]
    max_speed = CONFIG["reward_params"]["max_speed"]

    if speed_kmh < min_speed:
        return speed_kmh / min_speed
    elif speed_kmh > target_speed:
        return max(0.0, 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed))  # Ensure non-negative
    else:
        return 1.0

def calculate_centering_factor(env):
    return max(0.0, 1.0 - env.distance_from_center / CONFIG["reward_params"]["max_distance"])

def calculate_angle_factor(env):
    angle = env.vehicle.get_angle(env.current_waypoint)
    return max(0.0, 1.0 - abs(angle / np.deg2rad(CONFIG["reward_params"]["max_angle_center_lane"])))

def calculate_std_factor(env):
    std = np.std(env.distance_from_center_history)
    return max(0.0, 1.0 - abs(std / CONFIG["reward_params"]["max_std_center_lane"]))

def calculate_ttc_penalty(ttc, obstacle_type):
    return 0.0  # No TTC penalty without traffic


# --- Fear-Neuro-Inspired Penalty ---

def compute_fear_penalty(env):
    """
    Extracts current state from the environment and computes the
    kinematic fear heuristic for the CURRENT action being executed.
    Returns the scalar fear value in [0, 1].
    """
    speed_kmh = env.vehicle.get_speed()
    steer = env.vehicle.control.steer
    throttle = env.vehicle.control.throttle
    distance_from_center = getattr(env, 'distance_from_center', 0.0)
    angle_to_waypoint = env.vehicle.get_angle(env.current_waypoint)
    is_at_red_light = getattr(env, 'is_at_red_light', False)

    # Read fear hyperparameters from config (with safe defaults)
    fear_beta = CONFIG["reward_params"].get("fear_beta", 0.8)
    fear_dt = CONFIG["reward_params"].get("fear_lookahead_dt", 0.75)
    fear_max_speed = CONFIG["reward_params"].get("max_speed", 35.0)
    fear_max_dist = CONFIG["reward_params"].get("max_distance", 3.0)

    fear, cost, uncertainty = compute_fear(
        speed_kmh=speed_kmh,
        steer=steer,
        throttle=throttle,
        distance_from_center=distance_from_center,
        angle_to_waypoint=angle_to_waypoint,
        beta=fear_beta,
        dt=fear_dt,
        max_speed=fear_max_speed,
        max_distance=fear_max_dist,
        is_at_red_light=is_at_red_light,
    )
    return fear

# --- Emotion Calculation ---

def reward_fn_emotion(env):
    step_count = max(env.step_count, 1)
    delta_v_avg = env.delta_v_accum / step_count
    delta_v_rel = env.delta_v_accum / step_count
    D_lane = max(env.distance_from_center, 0.1)
    a_avg = env.delta_v_accum / step_count
    d_goal = env.distance_to_goal - env.distance_traveled
    d_total = env.distance_to_goal
    delta_d_goal = env.prev_distance_to_goal - env.distance_to_goal
    env.prev_distance_to_goal = env.distance_to_goal
    ttc = env.last_ttc
    obstacle_type = env.obst_type  # This might not be relevant without traffic
    v_max = CONFIG["reward_params"]["max_speed"]

    w1 = 0.1
    w2 = 0.2
    w3 = 0.3
    mu = 0.3
    delta_1, delta_2, delta_3 = 0.1, 0.2, 0.3
    lambda1 = 0.15  # TTC weight (reduced)
    lambda2 = 0.20  # Relative speed
    lambda5 = 0.1  # Goal progress
    c1 = 0.10  # TTC in Calmness
    c2 = 0.15
    c3 = 0.20

    R_prev = getattr(env, "restless_prev", 0)

    # Laziness
    lazy_score = (1 - delta_v_avg / v_max) * w1 + (d_goal / d_total) * w2 + (1 - a_avg) * w3 + (lambda1 * (1 / (1 + np.exp(5 * ttc))))  # TTC penalty

    # Restlessness
    restless_score = (1 - mu) * R_prev + mu * (delta_1 * delta_v_rel + delta_2 * (1 / D_lane) + delta_3 * (1 / (1 + np.exp(5 * ttc))))  # TTC penalty

    # Urgency
    urgency_score = (lambda2 * abs(delta_v_rel)) + (lambda5 * delta_d_goal) + (lambda1 * (1 / (1 + np.exp(5 * ttc)))) # TTC penalty

    # Calmness
    calmness_score = c1 * ttc + c2 * (1 - abs(delta_v_rel)) + c3 * (D_lane)

    env.restless_prev = restless_score

    return lazy_score, restless_score, urgency_score, calmness_score

# --- Main Reward Function ---

def refined_reward_fn(env):
    """
    Combines base rewards with emotion-modulated rewards, simplified for no traffic.
    """

    # 1. Base Rewards
    speed_reward = calculate_speed_reward(env)
    centering_factor = calculate_centering_factor(env)
    angle_factor = calculate_angle_factor(env)
    std_factor = calculate_std_factor(env)

    base_reward = speed_reward * centering_factor * angle_factor * std_factor

    # 2. Waypoint Reward
    waypoint_reward = (env.current_waypoint_index - env.prev_waypoint_index)  # 1.0 per waypoint passed
    base_reward += waypoint_reward * 0.5  # Adjust weight as needed (0.5 is an example)

    # 2b. Anti-reward-hacking: Heavy penalty for near-standstill (< 5 km/h)
    # MASKED when legally stopped at a red/yellow traffic light
    is_at_red_light = getattr(env, 'is_at_red_light', False)
    current_speed = env.vehicle.get_speed()
    if current_speed < 5.0 and not is_at_red_light:
        idle_penalty = -2.0 * (1.0 - current_speed / 5.0)  # -2.0 at 0 km/h, 0 at 5 km/h
        base_reward += idle_penalty

    # 3. Emotion Scores
    lazy_score, restless_score, urgency_score, calmness_score = reward_fn_emotion(env)

    # 4. Dynamic Emotion Weights and Adjustments
    emotion_scores = {"L": lazy_score, "Rt": restless_score, "U": urgency_score, "C": calmness_score}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    max_emotion_score = emotion_scores[dominant_emotion]

    dynamic_weights = emotion_weights.copy()
    emotion_adjustment = 0.0

    # Thresholds (Tunable)
    calm_threshold_high = 2.0  # Example: Above this is "very calm"
    restless_threshold_high = 2.0
    urgent_threshold_high = 2.0
    lazy_threshold_high = 2.0

    if dominant_emotion == "C":
        dynamic_weights["C"] *= 1.2  # Slightly favor calmness
        if max_emotion_score > calm_threshold_high:
            emotion_adjustment += 0.15 * base_reward  # Boost base reward for high calmness

    elif dominant_emotion == "Rt":
        dynamic_weights["Rt"] *= 1.5  # Increase penalty
        if max_emotion_score > restless_threshold_high:
            emotion_adjustment -= 0.15 * base_reward  # More penalty for erratic

    elif dominant_emotion == "U":
        dynamic_weights["U"] *= 1.5
        if max_emotion_score > urgent_threshold_high:
            emotion_adjustment -= 0.15 * base_reward

    elif dominant_emotion == "L":
        # Mask lazy penalty when legally stopped at a red/yellow traffic light
        if not is_at_red_light:
            dynamic_weights["L"] *= 6.0  # 4× amplified (was 1.5) — anti-reward-hacking
            if max_emotion_score > lazy_threshold_high:
                emotion_adjustment -= 0.60 * base_reward  # 4× amplified (was 0.15)

    # 5. Combine Rewards
    emotion_reward = (
        dynamic_weights["C"] * calmness_score +
        dynamic_weights["Rt"] * restless_score +
        dynamic_weights["U"] * urgency_score +
        dynamic_weights["L"] * lazy_score
    )

    # 6. Fear-Neuro-Inspired Penalty (FC-AC via reward shaping)
    fear = compute_fear_penalty(env)
    fear_lambda = CONFIG["reward_params"].get("fear_lambda", 2.0)
    fear_penalty = fear_lambda * fear

    # Store fear on env for logging / HUD / action override
    env.last_fear = fear

    total_reward = base_reward + emotion_reward + emotion_adjustment - fear_penalty

    return total_reward

reward_functions["reward_fn5"] = create_reward_fn(refined_reward_fn)

