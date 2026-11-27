#!/bin/env python3
"""
Usage:
ros2 topic pub /model_inference_command std_msgs/msg/String "{data: 'stop'}"

This node is data-driven from the inference config yaml (see
experiments/cfgs/inference_config_*.yaml). The config declares:
  - camera_topics / camera_names   : the camera streams to subscribe to
  - observations                   : model obs key -> ros topic to read it from
  - actions                        : model action key -> {topic} (hardware destination)
Keys containing "arm" are treated as a 7-dim arm pose (PoseStamped:
position + quaternion); keys containing "hand" as a Float32MultiArray of
joint angles. Single- vs dual-arm is just how many entries the dicts have,
so the same node handles both unimanual and bimanual policies.

[PATCH] Optional runtime settings, read from the `runtime_patches` section of
the inference config yaml (all keys optional, defaults shown):

  runtime_patches:
    renormalize_quat: true        # q / |q| after aggregation, if |q| is within quat_renorm_tol of 1
    quat_renorm_tol: 0.05         # farther than this from 1 -> still discarded (real bug, not rounding)
    log_timing: true              # step_i/timing/* in the hdf5 log + periodic latency summary
    arm_stall_warning: true       # warn if the arm does not follow the commands (reflex, controller off)
    hand_latch:
      enabled: false              # keep the hand closed through short "reopen" blips of the policy
      joints: [6, 9, 12, 15]      # joints whose mean decides closed/open (ORCA v2: index/middle/ring/pinky MCP)
      close_deg: 50.0             # mean above this -> hand counts as closed, latch engages
      open_deg: 30.0              # mean below this for release_steps steps -> latch released
      release_steps: 5            # 5 steps = 0.5 s at 10 Hz
    arm_lookahead_steps: 0        # >0: execute each predicted ARM chunk starting this many steps in.
                                  # The arm lags its commands by ~5 steps; the operators' commands in the
                                  # demos ran ahead of it, while the policy's chunk starts at the arm.
                                  # Also switches to plain chunk execution (no averaging) and
                                  # re-predicts every T_target - arm_lookahead_steps steps. Hand is not shifted.
    wrist_offset_deg: 0.0         # hand wrist (joint 0) calibration now minus at recording time, in degrees.
                                  # The policy sees wrist - offset and its wrist command is sent as cmd + offset,
                                  # so it works in the recording-time calibration. Logs keep hardware values.
    arm_lead_gain: 1.0            # >1: amplify each arm command's lead beyond the measured position (faster arm)
    arm_lead_max_m: 0.15          # cap on that lead
    arm_init_max_step_m: 0.002    # init_arm: max command lead ahead of the measured position per step
    arm_init_max_step_deg: 0.86   # init_arm: max command lead in rotation per step
    arm_init_tolerance_m: 0.015   # init_arm done: within this and arm_init_tolerance_deg, with the arm stopped
    arm_init_tolerance_deg: 5.0
    arm_init_integral_gain: 0.05  # init_arm: pushes the command past the target until the arm arrives
    arm_init_stalled_ok_deg: 20.0 # init_arm also done if the arm stops (arm_init_stalled_wait_s) within 2x the position
                                  # tolerance and this rotation (yaw about the tool axis barely follows)
    auto_return:
      enabled: false              # after each grasp attempt (missed, dropped, or no grasp within
                                  # no_grasp_timeout_s) drive back to arm_init_pose with the hand open,
                                  # clear the rest of the action chunk and continue. Needs allow_arm_init.
      no_grasp_timeout_s: 15.0
      lift_m: 0.05
      closed_deg: 50.0
      after_release_s: 1.0
"""
import time
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String as ROSString
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from copy import deepcopy
from cv_bridge import CvBridge
from threading import Lock
from functools import partial
import cv2
from scipy.spatial.transform import Rotation

import torch
import yaml

from faive_system.src.common.ros_utils import numpy_to_float32_multiarray, float32_multiarray_to_numpy
from srl_il.export.policy import get_policy_from_ckpt
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

import h5py
import os


class HandLatch:
    """[PATCH] Hysteresis on hand closure.

    While the policy commands a closed hand, its command passes through and is
    remembered. If the command then drops below the closed threshold, the last
    closed command is held, until the policy has asked for an open hand
    (mean below open_deg) for release_steps steps since it last asked for a
    closed one. In-between commands (open_deg..close_deg) neither count nor
    reset that tally, so a policy hovering around 30-50 deg cannot keep the
    hand trapped. max_hold_steps (if set) releases the hold after that many
    steps in total, however the policy behaves.
    Short reopen blips (e.g. at a chunk change) are filtered; a real release
    (placing the cube) is delayed by release_steps / control_freq seconds.
    """

    def __init__(self, joints=(6, 9, 12, 15), close_deg=50.0, open_deg=30.0, release_steps=5,
                 max_hold_steps=None):
        self.joints = list(joints)
        self.close_rad = np.deg2rad(close_deg)
        self.open_rad = np.deg2rad(open_deg)
        self.release_steps = int(release_steps)
        self.max_hold_steps = None if max_hold_steps is None else int(max_hold_steps)
        self.reset()

    def reset(self):
        self.hold = None
        self.open_count = 0
        self.held_steps = 0

    @property
    def latched(self):
        return self.hold is not None

    def __call__(self, hand):
        closure = float(np.mean(hand[self.joints]))
        if self.hold is not None:
            self.held_steps += 1
            if self.max_hold_steps is not None and self.held_steps > self.max_hold_steps:
                self.reset()
                return hand
        if closure >= self.close_rad:
            if self.hold is None:
                self.held_steps = 0
            self.hold = hand.copy()
            self.open_count = 0
            return hand
        if self.hold is None:
            return hand
        if closure < self.open_rad:
            self.open_count += 1
        if self.open_count >= self.release_steps:
            self.reset()
            return hand
        return self.hold.copy()


class CameraListener(Node):
    def __init__(self, camera_topic, name, node):
        self.camera_topic = camera_topic
        self.lock = Lock()
        self.image = None
        self.image_stamp = None
        self.name = name
        self.im_subscriber = node.create_subscription(
            Image, self.camera_topic, self.recv_im,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE),
            callback_group=node.observation_callbacks,
        )

    def recv_im(self, msg: Image):
        with self.lock:
            self.image = msg
            self.image_stamp = msg.header.stamp

    def get_im(self):
        with self.lock:
            return deepcopy(self.image), self.image_stamp


class PolicyPlayerAgent(Node):
    @staticmethod
    def _is_arm_key(key):
        """Arm keys carry a 7-dim PoseStamped; hand keys a Float32MultiArray."""
        return "arm" in key

    def __init__(self):
        super().__init__("policy_publisher")

        # --- Declare and get parameters ---
        self.declare_parameter("policy_ckpt_path", "")   # assume the policy ckpt is saved with its config
        self.declare_parameter("hand_qpos_dim", 17)
        self.declare_parameter("preview", False)
        self.declare_parameter("allow_arm_init", False)
        self.declare_parameter("arm_init_pose", "")
        self.preview = self.get_parameter("preview").value
        self.allow_arm_init = self.get_parameter("allow_arm_init").value and not self.preview
        arm_init_pose = self.get_parameter("arm_init_pose").value
        self.arm_init_target = None
        if self.allow_arm_init and arm_init_pose:
            target = np.asarray(yaml.safe_load(arm_init_pose), dtype=np.float64)
            if target.shape != (7,) or not np.isfinite(target).all() or not np.isclose(
                np.linalg.norm(target[3:]), 1.0, atol=0.001
            ):
                raise ValueError("arm_init_pose must be [x, y, z, qx, qy, qz, qw] with a unit quaternion")
            self.arm_init_target = target
        self.declare_parameter("logfile_root", "")      # root directory for logging

        self.declare_parameter("inference_cfg_file", "")
        self.policy_ckpt_path = self.get_parameter("policy_ckpt_path").value
        self.hand_qpos_dim = self.get_parameter("hand_qpos_dim").value
        self.logfile_root = self.get_parameter("logfile_root").value

        self.control_freq_hz = 10
        self.max_observation_age = 2.0 / self.control_freq_hz

        # --- Load the inference config (cameras / observation / action mapping) ---
        self.inference_cfg_file = self.get_parameter("inference_cfg_file").value
        if self.inference_cfg_file == "":
            raise RuntimeError("inference_cfg_file parameter is required")
        with open(self.inference_cfg_file, "r") as f:
            inference_config = yaml.safe_load(f)
        self.rgb_process_dict = inference_config.get("rgb_process_dict", {}) or {}
        self.resize_image_size = tuple(inference_config.get("rgb_resize_size", [224, 224]))
        self.hand_qpos_dim = inference_config.get("hand_qpos_dim", self.hand_qpos_dim)
        camera_topics = inference_config["camera_topics"]
        camera_names = inference_config["camera_names"]
        self.observations_cfg = inference_config["observations"]  # obs_key -> ros topic
        self.actions_cfg = inference_config["actions"]
        self.arm_command_frame = inference_config.get("arm_command_frame")

        # [PATCH] optional runtime settings (see module docstring)
        patches = inference_config.get("runtime_patches", {}) or {}
        self.renormalize_quat = bool(patches.get("renormalize_quat", True))
        self.quat_renorm_tol = float(patches.get("quat_renorm_tol", 0.05))
        self.log_timing = bool(patches.get("log_timing", True))
        self.arm_stall_warning = bool(patches.get("arm_stall_warning", True))
        # Max observation age when the action is published (after the forward pass).
        # Inputs must still be max_observation_age fresh before inference; this only
        # gives slower models (e.g. 2-frame history) room for the forward pass.
        self.max_action_latency = float(patches.get("max_action_latency_s", self.max_observation_age))
        latch_cfg = patches.get("hand_latch", {}) or {}
        self.hand_latch = None
        if latch_cfg.get("enabled", False):
            self.hand_latch = HandLatch(
                joints=latch_cfg.get("joints", [6, 9, 12, 15]),
                close_deg=latch_cfg.get("close_deg", 50.0),
                open_deg=latch_cfg.get("open_deg", 30.0),
                release_steps=latch_cfg.get("release_steps", 5),
                max_hold_steps=latch_cfg.get("max_hold_steps"),
            )
        self.wrist_offset_rad = np.deg2rad(float(patches.get("wrist_offset_deg", 0.0)))
        # [PATCH] init_arm step limits: how far ahead of the measured pose each init command may be.
        # The default 2 mm / 0.86 deg is too little lead for the Franka controller to move at all.
        self.arm_init_max_step = float(patches.get("arm_init_max_step_m", 0.002))
        self.arm_init_max_rot = np.deg2rad(float(patches.get("arm_init_max_step_deg", np.rad2deg(0.015))))
        self.arm_init_tol = float(patches.get("arm_init_tolerance_m", 0.015))
        self.arm_init_tol_rot = np.deg2rad(float(patches.get("arm_init_tolerance_deg", 5.0)))
        self.arm_init_stall_rot = np.deg2rad(float(patches.get("arm_init_stalled_ok_deg", 20.0)))
        # Give the integral term time to overcome the twist dead zone before accepting a stalled pose.
        self.arm_init_stall_wait = float(patches.get("arm_init_stalled_wait_s", 4.0))
        self.arm_init_ki = float(patches.get("arm_init_integral_gain", 0.05))
        # [PATCH] auto return: after each grasp attempt, go back to arm_init_pose and continue
        ar_cfg = patches.get("auto_return", {}) or {}
        self.auto_return = bool(ar_cfg.get("enabled", False))
        self.return_timeout_s = float(ar_cfg.get("no_grasp_timeout_s", 15.0))
        self.return_lift_m = float(ar_cfg.get("lift_m", 0.05))
        self.return_closed_rad = np.deg2rad(float(ar_cfg.get("closed_deg", 50.0)))
        self.return_after_release_s = float(ar_cfg.get("after_release_s", 1.0))
        self.return_joints = list(latch_cfg.get("joints", [6, 9, 12, 15]))
        if self.auto_return and (not self.allow_arm_init or self.arm_init_target is None):
            self.get_logger().error("auto_return needs allow_arm_init:=true and an arm_init_pose; disabled.")
            self.auto_return = False
        self.returning = False
        self.attempt = None
        self.attempt_idx = 0
        self.return_hand_cmd = None
        self.arm_lead_gain = float(patches.get("arm_lead_gain", 1.0))
        self.arm_lead_max = float(patches.get("arm_lead_max_m", 0.15))
        self.n_discards = 0                # discards since the last logged step
        self.last_discard_reason = ""
        self.n_quat_renorm = 0              # renormalizations since the last logged step
        self.max_quat_dev = 0.0             # largest |q|-1 seen since the last logged step
        self.timing_hist = deque(maxlen=50)
        self.stall_start = None             # (t, measured position) when the arm started lagging
        self.get_logger().info(
            f"[PATCH] renormalize_quat={self.renormalize_quat} log_timing={self.log_timing} "
            f"arm_stall_warning={self.arm_stall_warning} hand_latch={'on' if self.hand_latch else 'off'} "
            f"wrist_offset_deg={np.rad2deg(self.wrist_offset_rad):.1f} arm_lead_gain={self.arm_lead_gain} "
            f"auto_return={'on' if self.auto_return else 'off'}"
        )

        self.lock = Lock()
        self.observation_callbacks = MutuallyExclusiveCallbackGroup()

        # Relay only messages generated by this node. External messages sent
        # to the observable policy topics cannot pass through to the robot.
        self.action_pubs = {}
        self.control_pubs = {}
        for key, spec in self.actions_cfg.items():
            msg_type = PoseStamped if self._is_arm_key(key) else Float32MultiArray
            self.action_pubs[key] = self.create_publisher(msg_type, f"/policy_inference/{key}", 10)
            if not self.preview:
                self.control_pubs[key] = self.create_publisher(msg_type, spec["topic"], 1)

        # --- Observation subscribers + latest-value buffers (one per obs key) ---
        self.obs_values = {key: None for key in self.observations_cfg}
        self.obs_stamps = {key: None for key in self.observations_cfg}
        self.arm_observation_frames = {}
        self.obs_subs = {}
        for key, topic in self.observations_cfg.items():
            if self._is_arm_key(key):
                self.obs_subs[key] = self.create_subscription(
                    PoseStamped, topic, partial(self.arm_obs_callback, key=key), 10,
                    callback_group=self.observation_callbacks,
                )
            else:
                self.obs_subs[key] = self.create_subscription(
                    Float32MultiArray, topic, partial(self.hand_obs_callback, key=key), 10,
                    callback_group=self.observation_callbacks,
                )

        # --- Camera listeners ---
        self.camera_listeners = [
            CameraListener(camera_topic, camera_name, self)
            for camera_topic, camera_name in zip(camera_topics, camera_names)
        ]

        # --- Utils ---
        self.bridge = CvBridge()

        # --- Load policy ---
        print(f"About to load policy")
        self.policy = get_policy_from_ckpt(self.policy_ckpt_path)
        self.apply_arm_lookahead(int(patches.get("arm_lookahead_steps", 0)))
        self.policy.reset_policy()

        # --- State variables ---
        self.running_policy = False
        self.log_file = None
        self.log_filename = None
        self.step_idx = 0
        self.policy_stamp = None
        self.arm_init_deadline = None
        self.arm_init_settled_at = None
        self.arm_init_failed = False

        # --- Command Subscriber ---
        self.command_sub = self.create_subscription(
            ROSString, "/model_inference_command", self.command_callback, 10
        )

        # --- Timer for policy execution ---
        self.policy_timer = self.create_timer(1 / self.control_freq_hz, self.run_policy_cb)
        # Maintain a step counter to name groups incrementally

    def command_callback(self, msg: ROSString):
        """Handle commands received via /model_inference_command."""
        command = msg.data.strip().lower()
        if command == "start":
            if self.arm_init_deadline is not None or self.arm_init_failed:
                self.get_logger().warn("Arm initialization is pending or failed. Finish it, retry init_arm, or reset.")
                return
            if not self.running_policy:
                self.start_logging()
                self.policy.reset_policy()
                self.policy_stamp = None
                self.reset_patch_state()  # [PATCH]
                self.running_policy = True
                self.get_logger().info("Policy started.")
        elif command == "stop":
            if self.arm_init_deadline is not None:
                self.abort_arm_init("stopped by operator")
            if self.running_policy:
                self.running_policy = False
                self.stop_logging()
                self.get_logger().info("Policy stopped.")
        elif command == "reset":
            if self.running_policy or self.arm_init_deadline is not None:
                self.get_logger().warn("Stop policy execution or arm initialization before resetting.")
                return
            self.policy.reset_policy()
            self.policy_stamp = None
            self.arm_init_failed = False
            self.reset_patch_state()  # [PATCH]
            self.get_logger().info("Policy state reset.")
        elif command == "init_arm":
            self.start_arm_init()
        elif command == "quit":
            self.get_logger().info("Exiting.")
            rclpy.try_shutdown()
        else:
            self.get_logger().warn(f"Unknown command: {command}")

    def reset_patch_state(self):
        """[PATCH] Clear the latch, counters and stall detector."""
        if self.hand_latch is not None:
            self.hand_latch.reset()
        self.n_discards = 0
        self.last_discard_reason = ""
        self.n_quat_renorm = 0
        self.max_quat_dev = 0.0
        self.timing_hist.clear()
        self.stall_start = None
        self.returning = False
        self.attempt = {"phase": "approach", "t0": time.monotonic(), "z0": None, "lifted": False, "released_at": None}
        self.attempt_idx = 0
        self.return_hand_cmd = None

    def start_logging(self):
        """Start a new log file for the policy."""
        if self.log_file is not None:
            self.log_file.close()
            self.get_logger().info(f"Closed previous log file: {self.log_filename}")

        parent_dir = os.path.dirname(self.policy_ckpt_path)         # -> /.../20250117-231131_egocentric/checkpoints
        parent_parent_dir = os.path.dirname(parent_dir)             # -> /.../20250117-231131_egocentric
        experiment_name = os.path.basename(parent_parent_dir)       # -> 20250117-231131_egocentric

        ckpt_filename = os.path.basename(self.policy_ckpt_path)     # -> checkpoint_2150.pth
        ckpt_name_no_ext = os.path.splitext(ckpt_filename)[0]       # -> checkpoint_2150

        # 2) Use current time as the experiment timestamp
        current_time = time.strftime("%Y%m%d-%H%M%S")

        # 3) Construct the log filename
        self.logfile_folder = f"{experiment_name}-{ckpt_name_no_ext}"
        self.logfile_name = f"{current_time}.hdf5"
        os.makedirs(os.path.join(self.logfile_root, self.logfile_folder), exist_ok=True)
        # 4) Now open your HDF5 log file
        self.log_filename = os.path.join(self.logfile_root, self.logfile_folder, self.logfile_name)
        self.log_file = h5py.File(self.log_filename, "w")
        self.get_logger().info(f"Started new log file: {self.log_filename}")
        self.step_idx = 0  # Reset step index for the new log file

    def stop_logging(self):
        """Stop logging and close the log file."""
        if self.log_file is not None:
            self.log_file.close()
            self.get_logger().info(f"Closed log file: {self.log_filename}")
            self.log_file = None
            self.log_filename = None

    def rosimg_to_numpy_resized(self, img, crop,  K=None, D=None, rot90_k=0):
        """
        crop: top, bottom, left, right
        rot90_k, calles np.rot90, counterclockwise
        """
        img = self.bridge.imgmsg_to_cv2(img, "rgb8")
        h, w = img.shape[:2]
        DIM = (w, h)

        if K is not None and D is not None:
            K = np.array(K).reshape(3, 3)
            D = np.array(D).reshape(4, 1)

            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, DIM, np.eye(3), balance=0.1
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2
            )
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            img = undistorted_img

        img = img[crop[0]:crop[1], crop[2]:crop[3]]

        resized = cv2.resize(img, self.resize_image_size).transpose(2, 0, 1) / 255.0
        if rot90_k != 0:
            resized = np.ascontiguousarray(np.rot90(resized, k=rot90_k, axes=(1, 2)))
        return resized

    def __del__(self):
        """Close the log file in destructor to handle Ctrl+C shutdown."""
        if hasattr(self, 'log_file') and self.log_file is not None:
            self.log_file.close()
            print(f"HDF5 file {self.log_filename} closed.")

    def start_arm_init(self):
        if not self.allow_arm_init or self.running_policy or self.arm_init_deadline is not None:
            self.get_logger().warn("init_arm requires live mode, allow_arm_init, and an idle policy.")
            return
        output = self.control_pubs.get("actions_arm")
        if (
            self.arm_init_target is None
            or not self.arm_command_frame
            or self.actions_cfg.get("actions_arm", {}).get("topic") != "/right/arm/end_effector_pose_cmd"
            or self.observations_cfg.get("qpos_arm") != "/right/arm/end_effector_pose"
            or output is None
            or output.get_subscription_count() == 0
        ):
            self.abort_arm_init("set a verified arm_init_pose and frame and check the right-arm controller connection")
            return
        self.begin_arm_move()
        self.get_logger().info("Arm initialization started.")

    def begin_arm_move(self):
        """[PATCH] Start driving the arm to arm_init_target (run_arm_init does the steps)."""
        self.arm_init_failed = False
        self.arm_init_settled_at = None
        self.arm_init_bias_pos = np.zeros(3)     # integral term against the controller's steady-state error
        self.arm_init_bias_rot = np.zeros(3)
        self.arm_init_prev = None
        self.arm_init_deadline = time.monotonic() + 90.0

    def abort_arm_init(self, reason):
        self.arm_init_deadline = None
        self.arm_init_failed = True
        self.get_logger().error(f"Arm initialization stopped: {reason}.")
        if self.returning:                       # [PATCH] a failed auto return stops the policy
            self.returning = False
            if self.running_policy:
                self.running_policy = False
                self.stop_logging()
            self.get_logger().error("Auto return did not complete; policy stopped.")

    def begin_return(self, reason):
        """[PATCH] End the current attempt and drive the arm back to arm_init_pose; the policy pauses."""
        self.get_logger().info(f"Attempt {self.attempt_idx} ended ({reason}); returning to the init pose.")
        self.returning = True
        self.begin_arm_move()

    def finish_return(self):
        """[PATCH] Back at the init pose: drop the rest of the old action chunk and continue the policy."""
        self.returning = False
        self.policy.reset_policy()               # only clears the action buffer; the model stays loaded
        self.policy_stamp = None
        if self.hand_latch is not None:
            self.hand_latch.reset()
        self.attempt_idx += 1
        self.attempt = {"phase": "approach", "t0": time.monotonic(), "z0": None, "lifted": False, "released_at": None}
        self.get_logger().info(f"Back at the init pose; attempt {self.attempt_idx} starts.")

    def publish_hand(self, hand_cmd):
        """[PATCH] Send a hand command (used to keep the hand open while returning)."""
        for key in self.actions_cfg:
            if not self._is_arm_key(key):
                msg = numpy_to_float32_multiarray(hand_cmd)
                self.action_pubs[key].publish(msg)
                if key in self.control_pubs:
                    self.control_pubs[key].publish(msg)

    def track_attempt(self, action_dict, obs_dict):
        """[PATCH] Decide when a grasp attempt is over (missed, dropped, or no grasp in time)."""
        hand_key = next((k for k in action_dict if not self._is_arm_key(k)), None)
        if hand_key is None or obs_dict.get("qpos_arm") is None:
            return
        closure = float(np.mean(action_dict[hand_key][self.return_joints]))
        closed = closure >= self.return_closed_rad
        z = float(obs_dict["qpos_arm"][2])
        now = time.monotonic()
        a = self.attempt
        if self.return_hand_cmd is None and not closed:
            self.return_hand_cmd = np.array(action_dict[hand_key], dtype=np.float64).copy()   # open approach posture
        if a["phase"] == "approach":
            if closed:
                a.update(phase="grasp", z0=z, lifted=False)
            elif now - a["t0"] > self.return_timeout_s:
                self.begin_return(f"no grasp within {self.return_timeout_s:.0f} s")
        elif a["phase"] == "grasp":
            if z - a["z0"] > self.return_lift_m:
                a["lifted"] = True
            if not closed:
                if a["lifted"]:
                    a.update(phase="released", released_at=now)
                else:
                    self.begin_return("missed grasp: hand reopened without lifting")
        elif a["phase"] == "released" and now - a["released_at"] >= self.return_after_release_s:
            self.begin_return("cube lifted and released")

    def run_arm_init(self):
        now = time.monotonic()
        if now >= self.arm_init_deadline:
            self.abort_arm_init("target not reached within 90 s")
            return
        with self.lock:
            current = self.obs_values["qpos_arm"]
            stamp = self.obs_stamps["qpos_arm"]
            frame = self.arm_observation_frames.get("qpos_arm")
        if (
            current is None or not np.isfinite(current).all()
            or not np.isclose(np.linalg.norm(current[3:]), 1.0, atol=0.001)
            or self.observation_problems({"qpos_arm": stamp})
            or frame != self.arm_command_frame
        ):
            self.abort_arm_init("right-arm feedback is missing, stale, or in a different frame")
            return

        target = self.arm_init_target
        rotation = Rotation.from_quat(current[3:])
        target_rot = Rotation.from_quat(target[3:])
        error_pos = target[:3] - current[:3]
        error_rot = (target_rot * rotation.inv()).as_rotvec()
        distance = np.linalg.norm(error_pos)
        angle = np.linalg.norm(error_rot)

        # [PATCH] Done when close enough AND the arm has stopped. The compliant controller
        # settles ~1-2 cm / ~7 deg short of a command, so 5 mm / 3 deg was never reached.
        still = self.arm_init_prev is not None and (
            np.linalg.norm(current[:3] - self.arm_init_prev[:3]) < 0.001                       # < 1 mm per step
            and (Rotation.from_quat(self.arm_init_prev[3:]) * rotation.inv()).magnitude() < np.deg2rad(0.3))
        self.arm_init_prev = current.copy()
        # [PATCH] Also done when the arm has stopped making progress but is close (2 s still within
        # 2x the tolerances): the controller barely follows yaw about the tool axis, so the last
        # ~10-15 deg of yaw may never come (demo start yaw itself varies by about +-15 deg).
        within = distance <= self.arm_init_tol and angle <= self.arm_init_tol_rot
        close = distance <= 2 * self.arm_init_tol and angle <= self.arm_init_stall_rot
        if still and (within or close):
            if self.arm_init_settled_at is None:
                self.arm_init_settled_at = now
            elif now - self.arm_init_settled_at >= (0.5 if within else self.arm_init_stall_wait):
                self.arm_init_deadline = None
                if self.returning:
                    self.finish_return()
                else:
                    self.get_logger().info(
                        f"Arm reached init pose ({100 * distance:.1f} cm, {np.degrees(angle):.1f} deg off). Ready for start.")
                return
        else:
            self.arm_init_settled_at = None

        # [PATCH] Integral term: near the target, push the command past it by the accumulated
        # remaining error, so the arm actually arrives despite the controller's compliance.
        if distance < 0.05 and angle < np.deg2rad(20):
            self.arm_init_bias_pos = np.clip(self.arm_init_bias_pos + self.arm_init_ki * error_pos, -0.03, 0.03)
            self.arm_init_bias_rot = self.arm_init_bias_rot + self.arm_init_ki * error_rot
            n = np.linalg.norm(self.arm_init_bias_rot)
            if n > np.deg2rad(15):
                self.arm_init_bias_rot *= np.deg2rad(15) / n
        desired_pos = target[:3] + self.arm_init_bias_pos
        desired_rot = Rotation.from_rotvec(self.arm_init_bias_rot) * target_rot

        # Limit each requested displacement relative to the measured pose.
        # A stalled arm cannot accumulate commands further along the path.
        offset = desired_pos - current[:3]
        correction = (desired_rot * rotation.inv()).as_rotvec()
        if np.linalg.norm(offset) > self.arm_init_max_step:
            offset *= self.arm_init_max_step / np.linalg.norm(offset)
        if np.linalg.norm(correction) > self.arm_init_max_rot:
            correction *= self.arm_init_max_rot / np.linalg.norm(correction)
        quaternion = (Rotation.from_rotvec(correction) * rotation).as_quat()
        next_pose = np.concatenate([current[:3] + offset, quaternion])
        self.control_pubs["actions_arm"].publish(
            self.pose_message(next_pose, self.get_clock().now().to_msg())
        )
        if self.returning and self.return_hand_cmd is not None:   # [PATCH] keep the hand open on the way back
            self.publish_hand(self.return_hand_cmd)

    def pose_message(self, action, stamp):
        msg = PoseStamped()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = action[:3].astype(np.float64)
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = action[3:7].astype(np.float64)
        msg.header.stamp = stamp
        msg.header.frame_id = self.arm_command_frame or "panda_link0"
        return msg

    def publish(self, action_dict: dict):
        """Publish each generated action and, in live mode, relay it to the controller."""
        stamp = self.get_clock().now().to_msg()
        messages = {}
        for key, action in action_dict.items():
            if not np.isfinite(action).all():
                self.discard_actions(f"invalid {key} action")
                return False
            if self._is_arm_key(key):
                # [PATCH] temporal aggregation averages quaternions element-wise, which
                # shrinks |q|; renormalize small deviations instead of discarding them.
                if action.shape == (7,):
                    norm = float(np.linalg.norm(action[3:]))
                    self.max_quat_dev = max(self.max_quat_dev, abs(norm - 1.0))
                    if self.renormalize_quat and 1e-9 < abs(norm - 1.0) <= self.quat_renorm_tol:
                        action = action.copy()
                        action[3:] = action[3:] / norm
                        action_dict[key] = action
                        self.n_quat_renorm += 1
                if action.shape != (7,) or not np.isclose(np.linalg.norm(action[3:]), 1.0, atol=0.001):
                    self.discard_actions(f"invalid {key} pose")
                    return False
                messages[key] = self.pose_message(action, stamp)
            else:
                messages[key] = numpy_to_float32_multiarray(action)

        self.policy_stamp = stamp
        for key, msg in messages.items():
            self.action_pubs[key].publish(msg)
            if key in self.control_pubs:
                self.control_pubs[key].publish(msg)
        return True

    def arm_obs_callback(self, msg: PoseStamped, key: str):
        """Callback to record the current pose for an arm observation key."""
        pose = msg.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        frame = msg.header.frame_id
        if self.arm_command_frame and frame != self.arm_command_frame:
            with self.lock:
                self.obs_values[key] = None
                self.obs_stamps[key] = None
            self.get_logger().warn(
                f"{key} frame {frame!r} differs from {self.arm_command_frame!r}",
                throttle_duration_sec=2.0,
            )
            return
        with self.lock:
            self.obs_values[key] = np.concatenate([position, quaternion])
            self.obs_stamps[key] = msg.header.stamp
            self.arm_observation_frames[key] = frame

    def hand_obs_callback(self, msg: Float32MultiArray, key: str):
        """Callback to record the current joint state for a hand observation key."""
        state = float32_multiarray_to_numpy(msg)
        value = np.zeros(self.hand_qpos_dim)
        value[:len(state)] = state
        with self.lock:
            self.obs_values[key] = value
            self.obs_stamps[key] = self.get_clock().now().to_msg()

    def get_current_observations(self):
        """Gather current observations from cameras, arm(s), and hand(s)."""
        images_and_stamps = {cam.name: cam.get_im() for cam in self.camera_listeners}
        with self.lock:
            state_obs = dict(self.obs_values)
            stamp_dict = dict(self.obs_stamps)
        images = {key: value[0] for key, value in images_and_stamps.items()}
        stamp_dict.update({key: value[1] for key, value in images_and_stamps.items()})
        observations = {**images, **state_obs}
        missing = [key for key, value in observations.items() if value is None]
        if missing or self.observation_problems(stamp_dict):
            return None, stamp_dict

        # Convert ROS images to numpy arrays only after all inputs are available.
        if missing_keys:=(set(self.rgb_process_dict.keys()) - set(images.keys())):
            self.get_logger().warning(f"image keys {missing_keys} are defined in rgb_process_dict but not used. \nCheck if the match {images.keys()}" )
        for k,v in images.items():
            crop = (None,None,None,None)
            rot90_k = 0
            K, D = None, None
            if k in self.rgb_process_dict:
                for operations in self.rgb_process_dict[k]:
                    if operations["type"] == "crop":
                        # Crop the image
                        crop = (operations["top"], operations["bottom"],
                                operations["left"], operations["right"])
                    if operations["type"] == "rotate90":
                        # Rotate the image
                        rot90_k = operations["k"]
                    if operations["type"] == "undistort":
                        # Resize the image
                        K, D = operations["K"], operations["D"]
            images[k] = self.rosimg_to_numpy_resized(v, crop, K, D, rot90_k)

        return {**images, **state_obs}, stamp_dict

    def observation_ages(self, stamp_dict):
        """[PATCH] Age in seconds of every observation (None if missing)."""
        now = self.get_clock().now()
        return {
            key: None if stamp is None else (now - rclpy.time.Time.from_msg(stamp)).nanoseconds / 1e9
            for key, stamp in stamp_dict.items()
        }

    def observation_problems(self, stamp_dict, max_age=None):
        max_age = self.max_observation_age if max_age is None else max_age
        now = self.get_clock().now()
        problems = []
        for key, stamp in stamp_dict.items():
            if stamp is None:
                problems.append(f"{key} missing")
                continue
            age = (now - rclpy.time.Time.from_msg(stamp)).nanoseconds / 1e9
            if not 0.0 <= age <= max_age:
                problems.append(f"{key} age {age:.3f} s")
        return problems

    def discard_actions(self, reason):
        self.policy.reset_policy()
        self.policy_stamp = None
        self.n_discards += 1                # [PATCH] counted and written in the next logged step
        self.last_discard_reason = reason
        self.get_logger().warn(f"Policy output paused: {reason}", throttle_duration_sec=2.0)

    def apply_arm_lookahead(self, steps):
        """[PATCH] Execute each predicted arm chunk from `steps` entries in (see module docstring)."""
        if steps <= 0:
            return
        from srl_il.algo.base_algo import PolicyAggregatorSimple
        horizon = int(self.policy.T_target)
        if steps >= horizon:
            raise ValueError(f"arm_lookahead_steps={steps} must be smaller than the chunk length {horizon}")
        generate = self.policy.generate

        def generate_shifted(*args, **kwargs):
            out = generate(*args, **kwargs)
            for key, value in out.items():
                if self._is_arm_key(key):   # value: [B, T, D]; entries past the end are never executed
                    out[key] = torch.cat([value[:, steps:], value[:, -1:].expand(-1, steps, -1)], dim=1)
            return out

        self.policy.generate = generate_shifted
        self.policy._policy_aggregator = PolicyAggregatorSimple()
        self.policy._policy_update_every = horizon - steps
        self.get_logger().info(
            f"[PATCH] arm_lookahead_steps={steps}: arm chunk shifted, plain chunk execution, "
            f"re-predict every {horizon - steps} steps"
        )

    def check_arm_stall(self, action_dict, obs_dict):
        """[PATCH] Warn when the commanded pose is far from the measured one and the arm does not move."""
        if not self.arm_stall_warning or self.preview:
            return
        if "actions_arm" not in action_dict or obs_dict.get("qpos_arm") is None:
            return
        now = time.monotonic()
        meas = np.asarray(obs_dict["qpos_arm"][:3], dtype=np.float64)
        dist = float(np.linalg.norm(action_dict["actions_arm"][:3] - meas))
        if dist <= 0.03:
            self.stall_start = None
            return
        if self.stall_start is None or np.linalg.norm(meas - self.stall_start[1]) > 0.002:
            self.stall_start = (now, meas)
            return
        if now - self.stall_start[0] > 2.0:
            self.get_logger().error(
                f"Arm is not following the policy: command {100 * dist:.1f} cm away and no motion for "
                f"{now - self.stall_start[0]:.1f} s. Check the Franka controller (reflex? not active?).",
                throttle_duration_sec=5.0,
            )

    def run_policy_cb(self):
        """Compute an action only when every observation is fresh."""
        if self.arm_init_deadline is not None:
            self.run_arm_init()
            return
        if not self.running_policy:
            return

        t_start = time.monotonic()  # [PATCH] timing
        obs_dict, stamp_dict = self.get_current_observations()
        t_obs = time.monotonic()
        obs_ages = self.observation_ages(stamp_dict) if self.log_timing else {}
        problems = self.observation_problems(stamp_dict)
        if obs_dict is None or problems:
            self.discard_actions(", ".join(problems) or "missing observation")
            return

        dbg_info = {} if self.logfile_root != "" else None
        # Convert to torch tensors, run policy
        with torch.inference_mode():
            obs_policy = obs_dict
            if self.wrist_offset_rad != 0.0:    # [PATCH] wrist reading -> recording-time calibration
                obs_policy = dict(obs_dict)
                for k in obs_policy:
                    if "hand" in k:
                        obs_policy[k] = np.array(obs_policy[k], dtype=np.float64)
                        obs_policy[k][0] -= self.wrist_offset_rad
            obs_torch = {k: torch.tensor(v).float().unsqueeze(0) for k, v in obs_policy.items()} # add batch dimension
            actions = self.policy.predict_action(obs_torch, dbg_info)
            action_dict = {key: actions[key][0].cpu().numpy() for key in self.actions_cfg}
            if self.wrist_offset_rad != 0.0:    # [PATCH] wrist command -> today's calibration
                for k in action_dict:
                    if not self._is_arm_key(k):
                        action_dict[k] = action_dict[k].copy()
                        action_dict[k][0] += self.wrist_offset_rad
        t_fwd = time.monotonic()

        problems = self.observation_problems(stamp_dict, self.max_action_latency)
        if problems:
            self.discard_actions(", ".join(problems))
            return

        # [PATCH] hand latch: filter short reopen blips of the policy
        raw_actions = {k: v.copy() for k, v in action_dict.items()}
        latched = False
        if self.hand_latch is not None:
            for key in action_dict:
                if not self._is_arm_key(key):
                    action_dict[key] = self.hand_latch(action_dict[key])
            latched = self.hand_latch.latched

        # [PATCH] lead gain: amplify how far each arm command lies beyond the measured position, so the
        # compliant arm moves at demo speed (demo approach: 11.5 cm/s with ~6.4 cm lead; policy ~3.9 cm).
        # Not cumulative: a command at the measured position stays there.
        if self.arm_lead_gain != 1.0 and obs_dict.get("qpos_arm") is not None:
            meas = np.asarray(obs_dict["qpos_arm"][:3], dtype=np.float64)
            for key in action_dict:
                if self._is_arm_key(key) and action_dict[key].shape == (7,):
                    pose = action_dict[key].copy()
                    lead = (pose[:3] - meas) * self.arm_lead_gain
                    norm = np.linalg.norm(lead)
                    if norm > self.arm_lead_max:
                        lead *= self.arm_lead_max / norm
                    pose[:3] = meas + lead
                    action_dict[key] = pose

        # Warn about the gap between action publications.
        publish_gap = None
        if self.policy_stamp is not None:
            time_diff = rclpy.time.Time.from_msg(self.get_clock().now().to_msg()) - rclpy.time.Time.from_msg(self.policy_stamp)
            publish_gap = time_diff.nanoseconds / 1e9
            if time_diff.nanoseconds / 1e9 > self.max_observation_age:
                self.get_logger().warn(f"Policy execution is too slow ({time_diff.nanoseconds / 1e9:.3f} s).")
        if not self.publish(action_dict):
            return
        t_end = time.monotonic()

        self.check_arm_stall(action_dict, obs_dict)  # [PATCH]
        if self.auto_return and not self.returning:  # [PATCH]
            self.track_attempt(action_dict, obs_dict)

        # [PATCH] periodic latency summary in the terminal
        if self.log_timing:
            self.timing_hist.append((1000 * (t_obs - t_start), 1000 * (t_fwd - t_obs), 1000 * (t_end - t_start)))
            th = np.array(self.timing_hist)
            self.get_logger().info(
                f"[timing] last {len(th)} steps: obs {th[:, 0].mean():.0f} ms, forward {th[:, 1].mean():.0f} ms "
                f"(max {th[:, 1].max():.0f}), total {th[:, 2].mean():.0f} ms (max {th[:, 2].max():.0f}); "
                f"max |q|-1 {self.max_quat_dev:.5f}",
                throttle_duration_sec=5.0,
            )

        # Log the actions if logging is enabled
        if self.log_file is not None:
            # Log actions in actions/{key} format
            action_group = self.log_file.create_group(f"step_{self.step_idx}/actions")
            for k, v in action_dict.items():
                action_group.create_dataset(f"{k}", data=v)

            # [PATCH] policy output before the hand latch / quaternion renormalization
            raw_group = self.log_file.create_group(f"step_{self.step_idx}/actions_raw")
            for k, v in raw_actions.items():
                raw_group.create_dataset(f"{k}", data=v)

            # Log observations in obs_{k} format
            obs_group = self.log_file.create_group(f"step_{self.step_idx}/observations")
            for k, v in obs_dict.items():
                obs_group.create_dataset(f"{k}", data=v)

            # Log debug information in dbg/{k} format (if available)
            if dbg_info is not None:
                dbg_group = self.log_file.create_group(f"step_{self.step_idx}/dbg")
                for k, v in dbg_info.items():
                    if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                        if isinstance(v, torch.Tensor):
                            v = v.detach().cpu()
                        dbg_group.create_dataset(f"{k}", data=v)

            # [PATCH] timing and health of this step
            if self.log_timing:
                tg = self.log_file.create_group(f"step_{self.step_idx}/timing")
                tg.create_dataset("t_wall", data=time.time())
                tg.create_dataset("obs_ms", data=1000 * (t_obs - t_start))
                tg.create_dataset("forward_ms", data=1000 * (t_fwd - t_obs))
                tg.create_dataset("total_ms", data=1000 * (t_end - t_start))
                tg.create_dataset("publish_gap_s", data=np.nan if publish_gap is None else publish_gap)
                tg.create_dataset("n_discards_before", data=self.n_discards)
                tg.create_dataset("last_discard_reason", data=self.last_discard_reason)
                tg.create_dataset("n_quat_renorm", data=self.n_quat_renorm)
                tg.create_dataset("max_quat_dev", data=self.max_quat_dev)
                tg.create_dataset("hand_latched", data=int(latched))
                tg.create_dataset("attempt_idx", data=self.attempt_idx)
                for k, age in obs_ages.items():
                    tg.create_dataset(f"age_{k}_s", data=np.nan if age is None else age)
            self.n_discards = 0
            self.last_discard_reason = ""
            self.n_quat_renorm = 0
            self.max_quat_dev = 0.0
            self.log_file.flush()
            self.step_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = PolicyPlayerAgent()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.stop_logging()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
