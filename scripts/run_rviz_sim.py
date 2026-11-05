"""
FlowPilot poly-action closed-loop env test with RViz visualization.

Inference + Flightmare rendering loop, with ROS topics for RViz:
  SUB  /move_base_simple/goal        -> goal position (set via RViz 2D Nav Goal)
  PUB  /flowpilot/best_traj_visual   -> planned trajectory (PointCloud2, world frame)
  PUB  /depth_image                  -> current depth frame (sensor_msgs/Image)
  PUB  /uav_mesh                     -> drone position marker (visualization_msgs/Marker)
  PUB  /tf                           -> world->drone TF for RViz frame tracking

Usage (recommended, inside the simulation container):
    python docker/docker.py run
    # then in RViz: set Fixed Frame = world, add topics above, click 2D Nav Goal

Direct invocation is also possible when FLIGHTMARE_PATH and the Python
dependencies are already available on the host:
    python scripts/run_rviz_sim.py
"""

import faulthandler
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML, RoundTripDumper, dump

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from simulation import flightmare_proc, ply_utils
from simulation.sim_env import SimEnv
from ros_io import geometry, visualization as ros_viz

from inference.runtime   import FlowPilotRuntime
from inference.state     import WorldState, build_state_goal
from inference.depth_aug import clip_and_noise_depth_from_cfg

from ros_io.buffers import GoalBuffer

import configs.runtime.rviz as cfg

# ---------------------------------------------------------------------------
# WorldState and body-frame helpers (RViz-test-specific)
# ---------------------------------------------------------------------------

def yaw_frame_from_start_to_goal(start, goal, eps=1e-6):
    dx, dy = float(goal[0]-start[0]), float(goal[1]-start[1])
    if dx*dx + dy*dy < eps*eps:
        return None, None
    yaw = float(np.arctan2(dy, dx))
    cy, sy = float(np.cos(0.5*yaw)), float(np.sin(0.5*yaw))
    q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    return q, geometry.quat_to_rot(q)


# ---------------------------------------------------------------------------
# FlowPilot inference
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Camera intrinsics for depth back-projection
# ---------------------------------------------------------------------------
# 160x96 image, 90 deg horizontal FOV -> fx = fy = 80
_CAM_INTRINSICS = {"fx": 80.0, "fy": 80.0, "cx": 79.5, "cy": 47.5}
_CAM_MAX_M = 20.0   # denormalization scale (hardcoded by SimEnv.get_depth)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    faulthandler.enable(all_threads=True)

    script_dir = Path(__file__).resolve().parent

    # --- ROS init ---
    rospy.init_node('flowpilot_rviz', anonymous=False)
    traj_pub        = rospy.Publisher('/flowpilot/best_traj_visual', PointCloud2, queue_size=1)
    map_pub         = rospy.Publisher('/local_map',                  PointCloud2, queue_size=1, latch=True)
    local_cloud_pub = rospy.Publisher('/flowpilot/local_cloud',      PointCloud2, queue_size=1)
    depth_pub       = rospy.Publisher('/depth_image',                Image,       queue_size=1)
    marker_pub      = rospy.Publisher('/uav_mesh',                   Marker,      queue_size=1)
    vel_pub         = rospy.Publisher('/flowpilot/vel_marker',       Marker,      queue_size=1)
    goal_pub        = rospy.Publisher('/flowpilot/goal_marker',      Marker,      queue_size=1)
    goals = GoalBuffer(goal_z=2.0)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goals.callback, queue_size=1)
    tf_br = tf2_ros.TransformBroadcaster()

    ros_thread = threading.Thread(target=rospy.spin, daemon=True)
    ros_thread.start()
    print('[RViz] ROS node started. Set goal in RViz with 2D Nav Goal.')

    # --- Flightmare env ---
    # Flightmare install root: the Unity binary (flightrender/) and the C++
    # flightgym binding are external, not shipped in this repo. Point
    # FLIGHTMARE_PATH at the directory that contains flightrender/ and the
    # run/ output tree (or set it in the environment).
    flightmare_path = os.environ.get("FLIGHTMARE_PATH")
    if not flightmare_path:
        raise RuntimeError(
            "FLIGHTMARE_PATH is not set. Point it at the Flightmare install "
            "root (the directory containing flightrender/ and run/)."
        )
    os.environ["FLIGHTMARE_PATH"] = flightmare_path

    from flightgym import QuadrotorEnv_v1

    if cfg.launch_unity:
        binary_path = os.path.join(flightmare_path, 'flightrender/RPG_Flightmare/flightmare.x86_64')
        flightmare_proc.launch(binary_path)

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(_repo_root, 'flightlib', 'configs', 'vec_env.yaml')
    env_cfg = YAML().load(open(cfg_path))
    env_cfg['env']['num_envs'] = 1
    env_cfg['env']['num_threads'] = 1
    env_cfg['env']['render'] = True
    env_cfg['env']['supervised'] = False
    env_cfg['env']['imitation'] = False
    env = SimEnv(QuadrotorEnv_v1(dump(env_cfg, Dumper=RoundTripDumper), False))
    env.connect_unity()
    env.set_map_id(np.array([cfg.scene_id]))
    wb = env.world_box.astype(np.float64)
    print(f'[env] scene {cfg.scene_id} ready, world_box={wb.tolist()}')

    env.spawn_trees_and_save_ply(cfg.scene_id, spacing=cfg.spacing)
    env.render()

    # --- Publish scene point cloud (latched, published once) ---
    ply_path = os.path.join(flightmare_path, 'run', 'flowpilot_sim', f'pointcloud-{cfg.scene_id}.ply')
    if os.path.exists(ply_path):
        try:
            pts = ply_utils.load_ply_xyz(ply_path)
            pts = ply_utils.voxel_downsample(pts, voxel_size=0.3)
            map_header = Header()
            map_header.stamp = rospy.Time.now()
            map_header.frame_id = 'world'
            map_pub.publish(point_cloud2.create_cloud_xyz32(map_header, pts))
            print(f'[map] published {len(pts)} points from {ply_path} (voxel 0.3m)')
        except Exception as e:
            print(f'[map] failed to load/publish pointcloud: {e}')
    else:
        print(f'[map] PLY not found: {ply_path} (skip)')

    # --- FlowPilot runtime ---
    runtime = FlowPilotRuntime(cfg)

    # flowpilot.coords is imported lazily here (it's a root package now, not a
    # path-injected dependency).
    from flowpilot.coords import body_to_world, get_body_frame

    # traj_dt for real-time rollout pacing (basis hz may fall back to num_waypoints/T).
    traj_hz = float(getattr(runtime.basis, 'hz',
                            runtime.basis.num_waypoints / runtime.basis.T))
    traj_dt = 1.0 / max(1e-3, traj_hz)

    runtime.warmup()
    print('[model] warm-up done. Waiting for goal from RViz...')

    # Camera pitch
    pitch_rad = float(cfg.camera_pitch_deg) * math.pi / 180.0
    q_pitch   = np.array([math.cos(0.5*pitch_rad), 0.0, math.sin(0.5*pitch_rad), 0.0], dtype=np.float64)

    # --- Helper closures for publish ---
    def _stamp():
        return rospy.Time.now()

    def _pub_depth(depth_2d):
        ros_viz.publish_depth(depth_pub, depth_2d, _stamp())

    def _pub_drone(pos, quat):
        ros_viz.publish_drone_marker(marker_pub, pos, quat, _stamp())

    def _pub_vel(pos, vel):
        ros_viz.publish_vel_marker(vel_pub, pos, vel, _stamp())

    def _pub_goal(goal_pos):
        ros_viz.publish_goal_marker(goal_pub, goal_pos, _stamp())

    def _pub_tf(pos, quat):
        ros_viz.broadcast_tf(tf_br, pos, quat, "world", "drone", _stamp())

    def _pub_local_cloud(depth_2d, pos, quat):
        msg = ros_viz.depth_to_local_cloud_msg(
            depth_2d, pos, quat, _CAM_INTRINSICS, _stamp(),
            max_range=_CAM_MAX_M, vis_max_m=float(cfg.max_depth_m))
        if msg is not None:
            local_cloud_pub.publish(msg)

    def _pub_traj(wp_world):
        header = Header(frame_id="world", stamp=_stamp())
        pts = wp_world[:, 0:3].astype(np.float32).tolist()
        traj_pub.publish(point_cloud2.create_cloud_xyz32(header, pts))

    # --- Main loop ---
    cur = None  # persists across trajectories; None only on first run

    while not rospy.is_shutdown():
        # Wait for goal
        snap = goals.snapshot()
        if snap is None:
            time.sleep(0.1)
            continue
        goal_pos = snap.pos
        goal_seq = snap.seq

        # First run: initialize from env.reset(); subsequent runs: reuse last position
        if cur is None:
            obs = env.reset()
            pos0, _, _ = geometry.obs_to_world_pva(obs)
            cur = WorldState(pos=pos0, vel=np.zeros(3), acc=np.zeros(3), jerk=np.zeros(3))

        # Initial yaw pointing toward goal
        init_q, init_r = yaw_frame_from_start_to_goal(cur.pos, goal_pos)
        if init_q is not None:
            state_goal, quat_wb, r_wb = build_state_goal(
                cur, goal_pos, get_body_frame,
                quat_wb_override=init_q, r_wb_override=init_r)
        else:
            state_goal, quat_wb, r_wb = build_state_goal(cur, goal_pos, get_body_frame)

        quat_render = geometry.safe_quat(quat_wb)
        if abs(pitch_rad) > 1e-12:
            quat_render = geometry.safe_quat(geometry.quat_mul(quat_render, q_pitch))
        env.set_state(cur.pos, cur.vel, cur.acc, quat_render)
        env.render()
        cond_depth = clip_and_noise_depth_from_cfg(
            env.get_depth()[0].astype(np.float32), cfg, resize_to_net=True,
        )
        _pub_depth(cond_depth)

        print(f'[run] start=({cur.pos[0]:.1f},{cur.pos[1]:.1f},{cur.pos[2]:.1f}) '
              f'goal=({goal_pos[0]:.1f},{goal_pos[1]:.1f},{goal_pos[2]:.1f})')

        loop_idx = 0
        last_quat_render = quat_render.copy()
        while not rospy.is_shutdown():
            # Check for new goal mid-trajectory
            snap = goals.snapshot()
            if snap is not None and snap.seq != goal_seq:
                goal_seq = snap.seq
                goal_pos = snap.pos
                runtime.reset_prev()    # drop FM prev-traj anchor on goal change
                print(f'[run] goal updated mid-traj: ({goal_pos[0]:.1f},{goal_pos[1]:.1f})')

            # Inference
            t0 = time.perf_counter()
            res = runtime.infer(
                cond_depth, state_goal,
                r_wb=r_wb, pos_world=cur.pos,
            )
            wp_body = res.wp_body
            infer_ms = (time.perf_counter() - t0) * 1000.0

            if not np.all(np.isfinite(wp_body)):
                print(f'[run loop {loop_idx}] NaN/Inf in wp_body, resetting.')
                break

            wp_world = body_to_world(wp_body, cur.pos, r_wb).astype(np.float64)

            # Clamp to world box
            wp_world[:, 0] = np.clip(wp_world[:, 0], wb[0]+0.2, wb[3]-0.2)
            wp_world[:, 1] = np.clip(wp_world[:, 1], wb[1]+0.2, wb[4]-0.2)
            wp_world[:, 2] = np.clip(wp_world[:, 2], max(0.05, wb[2]+0.05), wb[5]-0.1)

            # Publish trajectory to RViz
            _pub_traj(wp_world)
            _pub_goal(goal_pos)

            print(f'[run loop {loop_idx}] infer={infer_ms:.1f}ms '
                  f'dist={np.linalg.norm(goal_pos - cur.pos):.2f}m')

            # Rollout in Flightmare
            exec_n = min(cfg.rollout_points, wp_world.shape[0])
            if cfg.real_time:
                step_deadline = time.perf_counter()
            for k in range(exec_n):
                pk = wp_world[k, 0:3]
                vk = wp_world[k, 3:6]
                ak = wp_world[k, 6:9]
                jk = wp_world[k, 9:12]
                if not (np.all(np.isfinite(pk)) and np.all(np.isfinite(vk)) and np.all(np.isfinite(ak))):
                    break
                quat_k, _ = get_body_frame(vk, ak, jk)
                quat_k = geometry.safe_quat(np.asarray(quat_k, dtype=np.float64))
                if abs(pitch_rad) > 1e-12:
                    quat_k = geometry.safe_quat(geometry.quat_mul(quat_k, q_pitch))
                env.set_state(pk, vk, ak, quat_k)
                env.render()
                cond_depth = clip_and_noise_depth_from_cfg(
                    env.get_depth()[0].astype(np.float32), cfg, resize_to_net=True,
                )
                last_quat_render = quat_k.copy()

                # Publish depth + drone pose to RViz at each render step
                _pub_depth(cond_depth)
                _pub_drone(pk, quat_k)
                _pub_vel(pk, vk)
                _pub_tf(pk, quat_k)
                _pub_local_cloud(cond_depth, pk, quat_k)

                if cfg.real_time:
                    step_deadline += traj_dt
                    sleep_t = step_deadline - time.perf_counter()
                    if sleep_t > 0.0:
                        time.sleep(sleep_t)

            # Advance state to last executed point
            last = wp_world[exec_n - 1]
            cur  = WorldState(pos=last[0:3].copy(), vel=last[3:6].copy(),
                               acc=last[6:9].copy(), jerk=last[9:12].copy())
            state_goal, quat_wb, r_wb = build_state_goal(cur, goal_pos, get_body_frame)

            dist = float(np.linalg.norm(goal_pos - cur.pos))
            if dist < cfg.goal_reach_dist:
                print(f'[run] arrived! dist={dist:.2f}m. Enter hover mode (waiting new goal).')
                hover_dt  = 1.0 / max(1e-3, float(cfg.hover_rate_hz))
                hover_vel = np.zeros(3, dtype=np.float64)
                hover_acc = np.zeros(3, dtype=np.float64)

                while not rospy.is_shutdown():
                    snap = goals.snapshot()
                    if snap is not None and snap.seq != goal_seq:
                        goal_seq = snap.seq
                        goal_pos = snap.pos
                        runtime.reset_prev()    # drop FM prev-traj anchor on goal change
                        print(f'[run] leave hover, new goal=({goal_pos[0]:.1f},{goal_pos[1]:.1f})')
                        break

                    env.set_state(cur.pos, hover_vel, hover_acc, last_quat_render)
                    env.render()
                    cond_depth = clip_and_noise_depth_from_cfg(
                        env.get_depth()[0].astype(np.float32), cfg, resize_to_net=True,
                    )

                    _pub_depth(cond_depth)
                    _pub_drone(cur.pos, last_quat_render)
                    _pub_vel(cur.pos, hover_vel)
                    _pub_goal(goal_pos)
                    _pub_tf(cur.pos, last_quat_render)
                    _pub_local_cloud(cond_depth, cur.pos, last_quat_render)

                    cur = WorldState(pos=cur.pos.copy(), vel=hover_vel.copy(),
                                     acc=hover_acc.copy(), jerk=np.zeros(3))
                    time.sleep(hover_dt)
                break

            loop_idx += 1


if __name__ == '__main__':
    main()
