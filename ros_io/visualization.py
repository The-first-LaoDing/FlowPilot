"""ROS visualization publishers for depth, markers, TF, and point clouds.

All ROS imports are lazy -- this module can be imported without rospy installed.
Publishers are passed as arguments (no global state).
"""

import numpy as np


def publish_depth(pub, depth_2d, stamp, encoding="32FC1"):
    """Publish a single depth image as sensor_msgs/Image.

    Args:
        pub: rospy.Publisher (sensor_msgs.msg.Image)
        depth_2d: [H, W] float32
        stamp: rospy.Time
    """
    from sensor_msgs.msg import Image

    h, w = depth_2d.shape[:2]
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "camera"
    msg.height = h
    msg.width = w
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = w * 4
    msg.data = depth_2d.astype(np.float32).tobytes()
    pub.publish(msg)


def publish_drone_marker(pub, pos, quat, stamp, frame_id="world"):
    """Publish a sphere marker at drone position."""
    from visualization_msgs.msg import Marker

    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.ns = "drone"
    m.id = 0
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2])
    m.pose.orientation.w = float(quat[0])
    m.pose.orientation.x = float(quat[1])
    m.pose.orientation.y = float(quat[2])
    m.pose.orientation.z = float(quat[3])
    m.scale.x = m.scale.y = m.scale.z = 0.4
    m.color.r = 0.2
    m.color.g = 0.6
    m.color.b = 1.0
    m.color.a = 1.0
    pub.publish(m)


def publish_vel_marker(pub, pos, vel, stamp, frame_id="world"):
    """Publish a text marker showing speed."""
    from visualization_msgs.msg import Marker

    speed = float(np.linalg.norm(vel))
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.ns = "velocity"
    m.id = 2
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2]) + 5.5
    m.pose.orientation.w = 1.0
    m.scale.z = 3.0
    m.text = f"{speed:.1f} m/s"
    m.color.r = 0.0
    m.color.g = 0.0
    m.color.b = 0.0
    m.color.a = 1.0
    pub.publish(m)


def publish_goal_marker(pub, goal_pos, stamp, frame_id="world"):
    """Publish an orange cylinder marker at the goal position.

    Previously a green sphere (scale 0.4, color 0.2/0.8/0.2) with zero external
    callers; now matches the orange-cylinder style both FlowPilot test scripts
    adopted locally.
    """
    from visualization_msgs.msg import Marker

    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.ns = "goal"
    m.id = 1
    m.type = Marker.CYLINDER
    m.action = Marker.ADD
    m.pose.position.x = float(goal_pos[0])
    m.pose.position.y = float(goal_pos[1])
    m.pose.position.z = float(goal_pos[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = 0.8
    m.scale.z = 4.0
    m.color.r = 1.0
    m.color.g = 0.4
    m.color.b = 0.0
    m.color.a = 0.6
    pub.publish(m)


def publish_trajectory(pub, points_xyz, stamp, frame_id="world"):
    """Publish a point sequence as a PointCloud2 message.

    Args:
        pub:        rospy.Publisher(..., PointCloud2, ...).
        points_xyz: [N, >=3] - only first 3 columns are used.
        stamp:      rospy.Time.
        frame_id:   TF frame name (default 'world').
    """
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2
    from std_msgs.msg import Header

    header = Header(frame_id=frame_id, stamp=stamp)
    pts = points_xyz[:, 0:3].astype(np.float32).tolist()
    msg = point_cloud2.create_cloud_xyz32(header, pts)
    pub.publish(msg)


def broadcast_tf(broadcaster, pos, quat, frame_id, child_id, stamp):
    """Broadcast a TF transform using tf2_ros.

    Args:
        broadcaster: tf2_ros.TransformBroadcaster
        pos: [3]
        quat: [4] wxyz
    """
    from geometry_msgs.msg import TransformStamped
    import rospy

    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_id
    t.transform.translation.x = float(pos[0])
    t.transform.translation.y = float(pos[1])
    t.transform.translation.z = float(pos[2])
    t.transform.rotation.w = float(quat[0])
    t.transform.rotation.x = float(quat[1])
    t.transform.rotation.y = float(quat[2])
    t.transform.rotation.z = float(quat[3])
    broadcaster.sendTransform(t)


def depth_to_local_cloud_msg(depth_2d, pos, quat, intrinsics, stamp,
                              max_range=20.0, vis_max_m=8.0, skip=3,
                              frame_id="world"):
    """Back-project depth to a colored PointCloud2 in world frame.

    Args:
        depth_2d: [H, W] float32, normalized [0,1]
        pos: [3] world position
        quat: [4] wxyz quaternion
        intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
        stamp: rospy.Time
        max_range: depth denormalization range in meters
        vis_max_m: visualization clip distance
        skip: pixel skip for downsampling
        frame_id: TF frame

    Returns:
        sensor_msgs.msg.PointCloud2 or None
    """
    from ros_io.geometry import quat_to_rot
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    H, W = depth_2d.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    us = np.arange(0, W, skip, dtype=np.int32)
    vs = np.arange(0, H, skip, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    dd = depth_2d[vv, uu] * max_range
    valid = (dd > 0.2) & (dd < vis_max_m) & np.isfinite(dd)
    uu, vv, dd = uu[valid], vv[valid], dd[valid]
    if len(dd) == 0:
        return None

    x_c = (uu.astype(np.float32) - cx) / fx * dd
    y_c = (vv.astype(np.float32) - cy) / fy * dd

    # Flightmare camera frame: X=forward(depth), Y=left(-x_c), Z=up(-y_c)
    pts_body = np.stack([dd, -x_c, -y_c], axis=1)
    R_wb = quat_to_rot(quat)
    pts_w = (R_wb @ pts_body.T).T + pos

    # Colormap: near=red→green→blue=far
    t = np.clip(dd / vis_max_m, 0.0, 1.0)
    r_ch = np.clip(255.0 * (2.0 * t - 1.0), 0, 255).astype(np.uint8)
    g_ch = np.clip(255.0 * (1.0 - np.abs(2.0 * t - 1.0)), 0, 255).astype(np.uint8)
    b_ch = np.clip(255.0 * (1.0 - 2.0 * t), 0, 255).astype(np.uint8)
    rgb_u32 = (r_ch.astype(np.uint32) << 16
               | g_ch.astype(np.uint32) << 8
               | b_ch.astype(np.uint32))

    N = len(dd)
    buf = np.zeros(N, dtype=[("x", np.float32), ("y", np.float32),
                              ("z", np.float32), ("rgb", np.float32)])
    buf["x"] = pts_w[:, 0].astype(np.float32)
    buf["y"] = pts_w[:, 1].astype(np.float32)
    buf["z"] = pts_w[:, 2].astype(np.float32)
    buf["rgb"] = rgb_u32.view(np.float32)

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]
    header = Header(frame_id=frame_id, stamp=stamp)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = N
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * N
    msg.data = buf.tobytes()
    msg.is_dense = True
    return msg
