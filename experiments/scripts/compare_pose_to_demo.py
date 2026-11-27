#!/usr/bin/env python3
"""Check whether the robot today reproduces a demo pose the way it looked when recorded.

1. Print a demo frame's logged arm pose (to move the arm there with the inference node's init_arm):
       python compare_pose_to_demo.py --demo 79 --step 12
2. With the arm at that pose and the cameras running, capture and compare:
       python compare_pose_to_demo.py --demo 79 --step 12 --capture
   Writes compare_demo<N>_t<T>.png: demo frame | live frame | 50/50 blend, for both cameras,
   and prints the live logged pose minus the demo's.

If the setup matches the recording, the arm and hand appear in the same place in both images.
"""
import argparse
import glob
import os

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

DATASET = os.path.expanduser("~/rwr_humble_ws/tmp/cube_sorting_red_processed_10hz_lzf")
CAMS = {"side": ("oakd_front_view", "/oakd_front_view/color"), "wrist": ("oakd_wrist_view", "/oakd_wrist_view/color")}
POSE_TOPIC = "/right/arm/end_effector_pose"
HAND_TOPIC = "/right/hand/joint_position_from_motor"


def load_demo(index, step):
    path = sorted(glob.glob(os.path.join(DATASET, "*.h5")))[index]
    with h5py.File(path, "r") as f:
        images = {name: f[f"observations/images/{cam}/color"][step] for name, (cam, _) in CAMS.items()}
        pose = f["observations/right/qpos_arm"][step]
        hand = f["observations/right/qpos_hand"][step]
    return path, images, pose, hand


def nearest_demo_frame(live_pose, rot_weight_cm_per_deg=0.2):
    """(index, step, distance) of the demo frame whose logged arm pose is closest to live_pose.
    Distance = position difference in cm + rot_weight * rotation difference in degrees."""
    best = None
    live_rot = Rotation.from_quat(live_pose[3:])
    for i, path in enumerate(sorted(glob.glob(os.path.join(DATASET, "*.h5")))):
        with h5py.File(path, "r") as f:
            q = f["observations/right/qpos_arm"][()]
        pos = np.linalg.norm(q[:, :3] - live_pose[:3], axis=1) * 100
        rot = np.degrees((Rotation.from_quat(q[:, 3:]) * live_rot.inv()).magnitude())
        d = pos + rot_weight_cm_per_deg * rot
        t = int(np.argmin(d))
        if best is None or d[t] < best[2]:
            best = (i, t, float(d[t]))
    return best


def describe(pose):
    rot = Rotation.from_quat(pose[3:])
    z = rot.apply([0, 0, 1]); x = rot.apply([1, 0, 0])
    tilt = np.degrees(np.arccos(np.clip(-z[2], -1, 1)))
    heading = np.degrees(np.arctan2(x[1], x[0]))
    return f"xyz [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]  tilt {tilt:.0f} deg  heading {heading:.0f} deg"


def capture():
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = rclpy.create_node("compare_pose_to_demo")
    bridge, got = CvBridge(), {}
    for name, (_, topic) in CAMS.items():
        node.create_subscription(Image, topic, lambda m, n=name: got.setdefault(n, bridge.imgmsg_to_cv2(m, "rgb8")), qos_profile_sensor_data)
    node.create_subscription(PoseStamped, POSE_TOPIC, lambda m: got.setdefault("pose", np.array(
        [m.pose.position.x, m.pose.position.y, m.pose.position.z,
         m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w])), 1)
    node.create_subscription(Float32MultiArray, HAND_TOPIC, lambda m: got.setdefault("hand", np.array(m.data)), 1)
    for _ in range(300):
        rclpy.spin_once(node, timeout_sec=0.1)
        if all(k in got for k in ["side", "wrist", "pose"]):
            break
    node.destroy_node(); rclpy.shutdown()
    missing = [k for k in ["side", "wrist", "pose"] if k not in got]
    if missing:
        raise SystemExit(f"No data on {missing} within 30 s; are the cameras and the arm driver running (ROS domain)?")
    return got


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", type=int, default=79)
    p.add_argument("--step", type=int, default=12)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--nearest", action="store_true",
                   help="capture first, then compare with the demo frame whose pose is closest to the live pose")
    p.add_argument("--out", default=".")
    a = p.parse_args()

    live = None
    if a.nearest:
        live = capture()
        a.demo, a.step, dist = nearest_demo_frame(live["pose"])
        a.capture = True
        print(f"Nearest demo frame: demo {a.demo} step {a.step} (distance score {dist:.1f})")

    path, demo_imgs, demo_pose, demo_hand = load_demo(a.demo, a.step)
    print(f"Demo {os.path.basename(path)} step {a.step}: {describe(demo_pose)}  wrist {np.degrees(demo_hand[0]):.0f} deg (demo-time units)")
    print("arm_init_pose for the inference launch:")
    print("  arm_init_pose:='[" + ", ".join(f"{v:.5f}" for v in demo_pose) + "]'")
    if not a.capture:
        return

    if live is None:
        live = capture()
    print(f"Live: {describe(live['pose'])}" + (f"  wrist {np.degrees(live['hand'][0]):.0f} deg" if "hand" in live else ""))
    d = (live["pose"][:3] - demo_pose[:3]) * 100
    rot = Rotation.from_quat(live["pose"][3:]) * Rotation.from_quat(demo_pose[3:]).inv()
    print(f"Live minus demo: dx {d[0]:+.1f} dy {d[1]:+.1f} dz {d[2]:+.1f} cm, rotation {np.degrees(rot.magnitude()):.1f} deg")

    rows = []
    for name in CAMS:
        h, w = demo_imgs[name].shape[:2]
        dem = demo_imgs[name]
        liv = cv2.resize(live[name], (w, h))
        blend = cv2.addWeighted(dem, 0.5, liv, 0.5, 0)
        tiles = []
        for img, label in [(dem, f"DEMO {a.demo} t{a.step} {name}"), (liv, f"LIVE {name}"), (blend, "50/50 blend")]:
            img = cv2.resize(img, (480, 270)).copy()
            cv2.rectangle(img, (0, 0), (480, 22), (0, 0, 0), -1)
            cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    out = os.path.join(a.out, f"compare_demo{a.demo}_t{a.step}.png")
    cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
