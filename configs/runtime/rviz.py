"""Configuration for FlowPilot RViz closed-loop test (scripts/run_rviz_sim.py)."""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- FlowPilot model paths ---
# These paths are the single source of truth for simulation and for
# `python docker/docker.py start`; no environment variables are read.
# Files outside this repository are mounted read-only at the same absolute
# path inside the container.
vae_pth           = os.path.join(_REPO_ROOT, "ckpt/Wan2.2_VAE.pth")
phase2_ckpt       = os.path.join(_REPO_ROOT, "ckpt/phase2/ckpt_final.pt")
action_stats_path = os.path.join(_REPO_ROOT, "ckpt/action_stats.pt")
coeff_stats_path  = os.path.join(_REPO_ROOT, "ckpt/coeff_stats.pt")

# --- Model / inference ---
device         = "cuda"
num_layers     = 12
num_steps      = 3
v_cmd          = 6.0          # Commanded cruise speed (m/s). 3 ~ 10

# --- Rollout / environment ---
scene_id         = 0
spacing          = 3.16          # Tree spacing for spawn_trees_and_save_ply
rollout_points   = 6             # correspond to 5 hz
goal_reach_dist  = 5.0
hover_rate_hz    = 20.0
real_time        = True        # If True, render depth in real-time cadence
camera_pitch_deg = -15.0
launch_unity     = True        # Auto-launch flightmare.x86_64 before connecting
max_depth_m      = 20.0         # Effective sensor visual range (metres). Pixels beyond
                                # this distance are clamped to max_depth_m (hard wall).
                                # Kept at 20.0 by default to match the model's depth
                                # normalization; set lower
                                # (e.g. 10.0) to mimic a shorter-range depth sensor.
depth_noise_std_m      = 0.0    # Baseline additive Gaussian noise std sigma0 (metres).
                                # 0.0 = disabled (baseline).
depth_noise_quad_coef  = 0.0    # Quadratic coefficient k (metres / metre^2) for axial
                                # noise that grows with distance:
                                #     sigma(z) = sigma0 + k * z^2   (z, sigma in metres)
                                # 0.0 = no distance dependence. Reference: RealSense D435
                                # ~ 0.005 (2 cm std at 2 m, 50 cm std at 10 m).
depth_dropout_prob     = 0.0    # Per-pixel probability of an invalid reading (pixel is
                                # forced to max_depth_m, as SimEnv does for NaN). 0.0 =
                                # disabled. Try 0.01-0.05 to emulate sparse holes from
                                # reflective / absorptive surfaces.
