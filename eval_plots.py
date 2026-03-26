"""
eval_plots.py — Publication-Ready Plots for FNI-RL Paper
=========================================================
Generates:
  1. Amygdala Intervention Time-Series (dual-axis: speed/throttle + fear)
  2. Safety Boundary vs. Fear scatter
  3. Spatial Trajectory with override markers
  4. Summary Metrics Table image

All plots saved as high-res PNGs in paper_results/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse

# ---------------------------------------------------------------------------
# Style Setup — Academic / Publication Ready
# ---------------------------------------------------------------------------
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('ggplot')  # Fallback

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

PAPER_DIR = 'paper_results'
FEAR_THRESHOLD = 0.7


def _ensure_dir():
    os.makedirs(PAPER_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1: Amygdala Intervention Time-Series
# ---------------------------------------------------------------------------
def plot_amygdala_intervention(df, episode_num=None, save_path=None):
    """
    Dual-axis line plot for a single episode.
      Left Y : Speed (km/h) + Throttle
      Right Y: Fear score
      Highlighted bands where is_at_red_light == True.
    """
    _ensure_dir()
    ep_df = df[df['model_id'] != 'route'].copy()

    if episode_num is not None:
        ep_df = ep_df[ep_df['episode'] == episode_num]
    else:
        # Pick the first episode with at least one override
        episodes = ep_df['episode'].unique()
        for ep in episodes:
            subset = ep_df[ep_df['episode'] == ep]
            if subset['override_active'].any():
                ep_df = subset
                episode_num = ep
                break
        else:
            ep_df = ep_df[ep_df['episode'] == episodes[0]]
            episode_num = episodes[0]

    steps = ep_df['step'].values
    speed = ep_df['speed'].values
    throttle = ep_df['throttle'].values
    fear = ep_df['fear'].values
    is_red = ep_df['is_at_red_light'].astype(bool).values

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left axis: Speed + Throttle
    color_speed = '#2196F3'
    color_throttle = '#4CAF50'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Speed (km/h) / Throttle', color='black')
    ln1 = ax1.plot(steps, speed, color=color_speed, linewidth=1.8, label='Speed (km/h)', alpha=0.9)
    ln2 = ax1.plot(steps, throttle * 40, color=color_throttle, linewidth=1.4, linestyle='--',
                   label='Throttle (×40)', alpha=0.8)
    ax1.set_ylim(-2, 45)
    ax1.tick_params(axis='y')

    # Right axis: Fear
    ax2 = ax1.twinx()
    color_fear = '#F44336'
    ax2.set_ylabel('Fear Score', color=color_fear)
    ln3 = ax2.plot(steps, fear, color=color_fear, linewidth=2.0, label='Fear', alpha=0.9)
    ax2.axhline(y=FEAR_THRESHOLD, color=color_fear, linestyle=':', linewidth=1.2, alpha=0.6, label=f'f₀ = {FEAR_THRESHOLD}')
    ax2.set_ylim(-0.05, 1.1)
    ax2.tick_params(axis='y', labelcolor=color_fear)

    # Highlight red-light regions
    in_red = False
    start_idx = 0
    for i in range(len(is_red)):
        if is_red[i] and not in_red:
            start_idx = i
            in_red = True
        elif not is_red[i] and in_red:
            ax1.axvspan(steps[start_idx], steps[i], alpha=0.15, color='red', label='Red Light Zone' if start_idx == 0 else '')
            in_red = False
    if in_red:
        ax1.axvspan(steps[start_idx], steps[-1], alpha=0.15, color='red')

    # Combined legend
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    red_patch = mpatches.Patch(color='red', alpha=0.15, label='Red Light Zone')
    threshold_line = plt.Line2D([0], [0], color=color_fear, linestyle=':', linewidth=1.2, alpha=0.6, label=f'f₀ = {FEAR_THRESHOLD}')
    ax1.legend(lns + [red_patch, threshold_line], labs + ['Red Light Zone', f'f₀ = {FEAR_THRESHOLD}'],
               loc='upper right', framealpha=0.9, edgecolor='gray')

    ax1.set_title(f'Amygdala Fast-Response Circuit — Episode {episode_num}', fontweight='bold')
    fig.tight_layout()

    out = save_path or os.path.join(PAPER_DIR, 'plot1_amygdala_intervention.png')
    plt.savefig(out)
    plt.close()
    print(f"  → Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Safety Boundary vs. Fear
# ---------------------------------------------------------------------------
def plot_safety_boundary_vs_fear(df, save_path=None):
    """
    Scatter: center_dev (X) vs. fear (Y).
    Colour-coded by override_active.
    """
    _ensure_dir()
    ep_df = df[df['model_id'] != 'route'].copy()

    center_dev = ep_df['center_dev'].values.astype(float)
    fear = ep_df['fear'].values.astype(float)
    override = ep_df['override_active'].astype(bool).values

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normal points
    mask_normal = ~override
    ax.scatter(center_dev[mask_normal], fear[mask_normal],
               c='#2196F3', alpha=0.3, s=15, label='Normal Driving', edgecolors='none')

    # Override points
    ax.scatter(center_dev[override], fear[override],
               c='#F44336', alpha=0.5, s=25, label='Override Active', edgecolors='none', marker='x')

    # Reference lines
    ax.axhline(y=FEAR_THRESHOLD, color='#F44336', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Fear Threshold (f₀={FEAR_THRESHOLD})')
    ax.axvline(x=3.0, color='#FF9800', linestyle='--', linewidth=1.5, alpha=0.7, label='Lane Limit (3.0m)')

    ax.set_xlabel('Center Lane Deviation (m)')
    ax.set_ylabel('Fear Score')
    ax.set_title('Safety Boundary vs. Fear Response', fontweight='bold')
    ax.set_xlim(-0.1, max(3.5, center_dev.max() * 1.1))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')

    fig.tight_layout()
    out = save_path or os.path.join(PAPER_DIR, 'plot2_safety_boundary_fear.png')
    plt.savefig(out)
    plt.close()
    print(f"  → Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Spatial Trajectory with Override Markers
# ---------------------------------------------------------------------------
def plot_spatial_trajectory(df, episode_num=None, save_path=None):
    """
    2D trajectory: vehicle path vs. route waypoints.
    Red dots mark where override was active (fear > f₀).
    """
    _ensure_dir()
    episodes = df[df['model_id'] != 'route']['episode'].unique()

    if episode_num is not None:
        episodes = [episode_num]

    n_eps = min(len(episodes), 4)  # Show up to 4 episodes
    fig, axes = plt.subplots(1, n_eps, figsize=(6 * n_eps, 5), squeeze=False)

    for idx, ep in enumerate(episodes[:n_eps]):
        ax = axes[0, idx]
        ep_df = df[(df['episode'] == ep) & (df['model_id'] != 'route')]
        route_df = df[(df['episode'] == ep) & (df['model_id'] == 'route')]

        # Route waypoints
        if not route_df.empty:
            ax.plot(route_df['route_x'].astype(float), route_df['route_y'].astype(float),
                    color='#4CAF50', linewidth=2.5, alpha=0.6, label='Planned Route', zorder=1)
            ax.plot(float(route_df['route_x'].iloc[0]), float(route_df['route_y'].iloc[0]),
                    'go', markersize=10, label='Start', zorder=3)
            ax.plot(float(route_df['route_x'].iloc[-1]), float(route_df['route_y'].iloc[-1]),
                    'rs', markersize=10, label='Goal', zorder=3)

        # Vehicle path
        vx = ep_df['vehicle_location_x'].astype(float).values
        vy = ep_df['vehicle_location_y'].astype(float).values
        ax.plot(vx, vy, color='#2196F3', linewidth=1.5, alpha=0.8, label='Vehicle Path', zorder=2)

        # Override points
        override = ep_df['override_active'].astype(bool).values
        if override.any():
            ax.scatter(vx[override], vy[override],
                       c='#F44336', s=30, zorder=4, label='Fear Override', edgecolors='darkred', linewidths=0.5)

        # Red light stops
        is_red = ep_df['is_at_red_light'].astype(bool).values
        if is_red.any():
            ax.scatter(vx[is_red], vy[is_red],
                       c='#FF9800', s=20, marker='D', zorder=4, label='Red Light Stop',
                       edgecolors='darkorange', linewidths=0.5, alpha=0.7)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Episode {ep}', fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor='gray')
        ax.set_aspect('equal', adjustable='datalim')

    fig.suptitle('Spatial Trajectory with FNI-RL Override Points', fontweight='bold', fontsize=16, y=1.02)
    fig.tight_layout()
    out = save_path or os.path.join(PAPER_DIR, 'plot3_spatial_trajectory.png')
    plt.savefig(out)
    plt.close()
    print(f"  → Saved: {out}")


# ---------------------------------------------------------------------------
# Summary Metrics Table Image
# ---------------------------------------------------------------------------
def plot_summary_table(df, save_path=None):
    """
    Renders a publication-ready metrics summary table as PNG.
    """
    _ensure_dir()
    ep_df = df[df['model_id'] != 'route'].copy()
    route_df = df[df['model_id'] == 'route'].copy()

    episodes = ep_df['episode'].unique()
    n_episodes = len(episodes)

    # --- Calculate metrics ---
    # Route Completion Rate
    completed = 0
    for ep in episodes:
        ep_data = ep_df[ep_df['episode'] == ep]
        rt_data = route_df[route_df['episode'] == ep]
        if not rt_data.empty and not ep_data.empty:
            last_vx = float(ep_data['vehicle_location_x'].iloc[-1])
            last_vy = float(ep_data['vehicle_location_y'].iloc[-1])
            goal_x = float(rt_data['route_x'].iloc[-1])
            goal_y = float(rt_data['route_y'].iloc[-1])
            dist = np.sqrt((last_vx - goal_x)**2 + (last_vy - goal_y)**2)
            if dist < 5.0:
                completed += 1

    route_completion = (completed / n_episodes) * 100 if n_episodes > 0 else 0

    # Average speed
    avg_speed = ep_df['speed'].astype(float).mean()
    std_speed = ep_df['speed'].astype(float).std()

    # Average lane deviation
    avg_dev = ep_df['center_dev'].astype(float).mean()
    std_dev = ep_df['center_dev'].astype(float).std()

    # Amygdala interventions per episode
    overrides_per_ep = []
    for ep in episodes:
        ep_data = ep_df[ep_df['episode'] == ep]
        n_overrides = ep_data['override_active'].astype(bool).sum()
        overrides_per_ep.append(n_overrides)
    avg_overrides = np.mean(overrides_per_ep)
    std_overrides = np.std(overrides_per_ep)

    # Red Light Violation Rate
    # A violation = moving through a red light (is_at_red_light=True AND speed > 5 km/h)
    red_light_steps = ep_df[ep_df['is_at_red_light'].astype(bool)]
    if len(red_light_steps) > 0:
        violations = red_light_steps[red_light_steps['speed'].astype(float) > 5.0]
        rlvr = (len(violations) / len(red_light_steps)) * 100
    else:
        rlvr = 0.0

    # Average fear
    avg_fear = ep_df['fear'].astype(float).mean()

    # --- Build table ---
    metrics = [
        ['Route Completion Rate', f'{route_completion:.1f}%'],
        ['Average Speed', f'{avg_speed:.2f} ± {std_speed:.2f} km/h'],
        ['Average Lane Deviation', f'{avg_dev:.3f} ± {std_dev:.3f} m'],
        ['Average Fear Score', f'{avg_fear:.3f}'],
        ['Amygdala Interventions / Episode', f'{avg_overrides:.1f} ± {std_overrides:.1f}'],
        ['Red Light Violation Rate (RLV)', f'{rlvr:.1f}%'],
        ['Total Episodes', f'{n_episodes}'],
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')
    ax.set_title('FNI-RL Evaluation Summary', fontweight='bold', fontsize=16, pad=20)

    table = ax.table(
        cellText=metrics,
        colLabels=['Metric', 'Value'],
        cellLoc='center',
        loc='center',
        colWidths=[0.55, 0.35],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    # Header styling
    for col in range(2):
        cell = table[0, col]
        cell.set_facecolor('#1976D2')
        cell.set_text_props(color='white', fontweight='bold', fontsize=13)

    # Alternate row colours
    for row in range(1, len(metrics) + 1):
        for col in range(2):
            cell = table[row, col]
            if row % 2 == 0:
                cell.set_facecolor('#E3F2FD')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#BBDEFB')

    fig.tight_layout()
    out = save_path or os.path.join(PAPER_DIR, 'summary_metrics_table.png')
    plt.savefig(out)
    plt.close()
    print(f"  → Saved: {out}")


# ---------------------------------------------------------------------------
# Legacy compatibility: original plot_eval + summary_eval from eval.py
# ---------------------------------------------------------------------------
def eucldist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def plot_eval(eval_csv_paths, output_name=None):
    """Original multi-episode subplot grid — kept for backward compatibility."""
    episode_numbers = pd.read_csv(eval_csv_paths[0])['episode'].unique()

    cols = ['Steer', 'Throttle', 'Speed (km/h)', 'Reward', 'Center Deviation (m)', 'Distance (m)',
            'Angle next waypoint (grad)', 'Trayectory']

    fig, axs = plt.subplots(len(episode_numbers), len(cols), figsize=(4 * len(cols), 3 * len(episode_numbers)))

    if len(eval_csv_paths) == 1:
        eval_plot_path = eval_csv_paths[0].replace(".csv", ".png")
    else:
        os.makedirs('tensorboard/eval_plots', exist_ok=True)
        eval_plot_path = f'./tensorboard/eval_plots/{output_name}'

    models = ['Waypoints']

    for e, path in enumerate(eval_csv_paths):
        df = pd.read_csv(path)
        model_id = df.loc[df['model_id'] != 'route', 'model_id'].unique()[0]
        models.append(model_id)
        for i, episode_number in enumerate(episode_numbers):
            episode_df = df[(df['episode'] == episode_number) & (df['model_id'] != 'route')]
            route_df = df[(df['episode'] == episode_number) & (df['model_id'] == 'route')]

            axs[i, 0].plot(episode_df['step'], episode_df['steer'], label=model_id)
            axs[i, 0].set_xlabel('Step')
            axs[i, 0].set_ylim(-1, 1)

            axs[i][1].plot(episode_df['step'], episode_df['throttle'], label=model_id)
            axs[i][1].set_xlabel('Step')
            axs[i, 1].set_ylim(0, 1)

            axs[i][2].plot(episode_df['step'], episode_df['speed'], label=model_id)
            axs[i][2].set_xlabel('Step')
            axs[i, 2].set_ylim(0, 40)

            axs[i][3].plot(episode_df['step'], episode_df['reward'], label=model_id)
            axs[i][3].set_xlabel('Step')
            axs[i, 3].set_ylim(-0.2, 1)

            axs[i][4].plot(episode_df['step'], episode_df['center_dev'], label=model_id)
            axs[i][4].set_xlabel('Step')
            axs[i, 4].set_ylim(0, 3)

            axs[i][5].plot(episode_df['step'], episode_df['distance'], label=model_id)
            axs[i][5].set_xlabel('Step')

            axs[i][6].plot(episode_df['step'], episode_df['angle_next_waypoint'], label=model_id)
            axs[i][6].set_xlabel('Step')

            if e == 0:
                axs[i][7].plot(route_df['route_x'].head(1), route_df['route_y'].head(1), 'go',
                               label='Start')
                axs[i][7].plot(route_df['route_x'].tail(1), route_df['route_y'].tail(1), 'ro',
                               label='End')
                axs[i][7].plot(route_df['route_x'], route_df['route_y'], label='Waypoints', color="green")

                axs[i, 7].set_xlim(left=min(-5, min(route_df['route_x'] - 3)))
                axs[i, 7].set_xlim(right=max(5, max(route_df['route_x'] + 3)))
            axs[i][7].plot(episode_df['vehicle_location_x'], episode_df['vehicle_location_y'], label=model_id)

    pad = 5
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axs[:, 0], episode_numbers):
        ax.annotate(f"Episode {row}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    handles, labels = axs[0][7].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.1 + 0.02 * len(labels), 1, 1))
    plt.savefig(eval_plot_path)
    plt.close()
    print(f"  → Saved legacy plots: {eval_plot_path}")


def summary_eval(eval_csv_path):
    """Original summary CSV — kept for backward compatibility, with paper plots added."""
    df = pd.read_csv(eval_csv_path)

    # --- Generate paper plots ---
    print("\n[FNI-RL] Generating publication-ready plots...")
    plot_amygdala_intervention(df)
    plot_safety_boundary_vs_fear(df)
    plot_spatial_trajectory(df)
    plot_summary_table(df)
    print(f"[FNI-RL] All plots saved to {PAPER_DIR}/\n")

    # --- Original summary logic ---
    df_route = df[df['model_id'] == 'route']
    df_data = df[df['model_id'] != 'route'].copy()
    df_data = df_data.drop(['model_id', 'route_x', 'route_y'], axis=1)

    df_distance = df_data.groupby(['episode'], as_index=False).last()[['episode', 'distance']].rename(
        columns={'distance': 'total_distance'})

    df_reward = df_data.groupby(['episode'], as_index=False).sum(numeric_only=True)[['episode', 'reward']].rename(
        columns={'reward': 'total_reward'})

    df_mean_std = df_data.groupby(['episode'], as_index=False).agg(
        {'speed': ['mean', 'std'], 'center_dev': ['mean', 'std'], 'reward': ['mean', 'std']})
    df_mean_std.columns = ['episode', 'speed_mean', 'speed_std', 'center_dev_mean', 'center_dev_std', 'reward_mean',
                           'reward_std']

    df_waypoint = df_route.groupby(['episode'], as_index=False).last()[['episode', 'route_x', 'route_y']]
    df_success = df_data.groupby(['episode'], as_index=False).last()[
        ['episode', 'vehicle_location_x', 'vehicle_location_y']]
    df_success = pd.merge(df_success, df_waypoint, on='episode')
    df_success['success'] = df_success.apply(
        lambda x: eucldist(x['vehicle_location_x'], x['vehicle_location_y'], x['route_x'], x['route_y']) < 5, axis=1)
    df_success = df_success[['episode', 'success']]

    df_summary = pd.merge(df_distance, df_reward, on='episode')
    df_summary = pd.merge(df_summary, df_mean_std, on='episode')
    df_summary = pd.merge(df_summary, df_success, on='episode')

    df_summary['episode'] = df_summary['episode'].astype(str)
    df_summary.loc['total'] = df_summary.mean(numeric_only=True)
    df_summary.loc['total', 'episode'] = 'total'
    df_summary.loc['total', 'total_reward'] = df_summary['total_reward'].sum()
    df_summary.loc['total', 'total_distance'] = df_summary['total_distance'].sum()

    if True in df_summary['success'].unique():
        df_summary.loc['total', 'success'] = df_summary['success'].value_counts()[True] / len(df_summary['success'])
    else:
        df_summary.loc['total', 'success'] = 0

    output_path = eval_csv_path.replace("eval.csv", "eval_summary.csv")
    df_summary.to_csv(output_path, index=False)
    print(f"  → Saved summary CSV: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point: generate paper plots from existing CSV
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots from eval CSV")
    parser.add_argument("--csv", type=str, required=False, help="Path to eval CSV file")
    parser.add_argument("--models", nargs='+', type=str, default=None, help="Model IDs for legacy comparison")
    args = vars(parser.parse_args())

    if args['csv']:
        df = pd.read_csv(args['csv'])
        print("[FNI-RL] Generating publication-ready plots...")
        plot_amygdala_intervention(df)
        plot_safety_boundary_vs_fear(df)
        plot_spatial_trajectory(df)
        plot_summary_table(df)
        print(f"[FNI-RL] All plots saved to {PAPER_DIR}/")
    elif args['models']:
        compare_models = args['models']
        eval_csv_paths = []
        for model in compare_models:
            model_id, steps = model.split("-")
            eval_csv_paths.append(os.path.join("tensorboard", model_id, "eval", f"model_{steps}_steps_eval.csv"))
        plot_eval(eval_csv_paths, output_name="+".join(compare_models))
    else:
        print("Usage: python eval_plots.py --csv <path_to_eval.csv>")
        print("       python eval_plots.py --models <model1-steps> <model2-steps>")


if __name__ == '__main__':
    main()

