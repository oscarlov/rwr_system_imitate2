# Cube sorting inference: setup and findings (27 Nov 2026)

How to run `cube40_v4_diff_77` / `checkpoint_150` on the current hardware, and what was learned
while getting it to grasp. All runtime fixes live in `experiments/model_inference_node.py` under
`runtime_patches` in `experiments/cfgs/inference_config_cube_sorting.yaml`.

## Hardware state this config assumes

- **Franka Desk, End-Effector "Other"**: mass 1.33 kg, CoM (0.02, -0.02, 0.05); flange-to-EE transform
  with **identity rotation and 0.1034 m along z**. With the bare flange (identity, no offset) the
  reported pose is ~10 cm off from the demos and the policy hovers above the cube. Of the rotations
  tried (-45, -22.5, 0, +45 deg), 0 kept the hand heading stable.
- **ORCA hand calibration** from 27 Nov 12:17 (manual wrist recalibration). The demos (24 Nov) were
  recorded with an uncalibrated wrist, hence `wrist_offset_deg`.
- ROS: every terminal needs `export ROS_DOMAIN_ID=7 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.

## Running

```bash
ros2 launch experiments run_policy_inference.launch.py \
  policy_ckpt_path:=/home/imitate/rwr_humble_ws/runs/cube40_v4_diff_77/checkpoints/checkpoint_150.pth \
  allow_arm_init:=true \
  arm_init_pose:='[0.448, 0.0, 0.204, 0.99373, 0.07896, -0.07688, 0.01862]'
ros2 topic pub --once -w 1 /model_inference_command std_msgs/msg/String "{data: 'init_arm'}"
ros2 topic pub --once -w 1 /model_inference_command std_msgs/msg/String "{data: 'start'}"
```

The init pose is the median start pose of the demos. With `auto_return` the arm goes back there after
every attempt (miss, drop, or no grasp within 15 s) and continues; `stop` ends the run. Stop stale
launches first: leftover camera / hand nodes hold the devices.

Evaluate a run with `experiments/scripts/summarize_trials.py --since HH:MM` (one row per attempt).

## What helped (in the config)

| Setting | Why |
|---|---|
| Desk EE 0.1034 m | demo-time reported pose; removed the 8-11 cm hover |
| `wrist_offset_deg: -30` | wrist calibration changed after the demos; 0, -35, -50, -70 were worse |
| `arm_lookahead_steps: 4` | the arm lags its commands ~0.5 s; without it the arm barely lifts |
| `arm_lead_gain: 2.0` | the policy keeps its target only ~4 cm ahead of the compliant arm (demos ~6.4 cm), so the arm moved at half demo speed and closed before reaching far cubes |
| hand latch (fixed) | holds a grasp through short "open" blips; capped at 5 s |
| `init_arm` + `auto_return` | starts every attempt from the demo start pose; starts away from it (or twisted >14 deg) fail |

Results with the corrected success check (cube visibly comes along in the wrist camera): about
45-57% of attempts, varying strongly from run to run with the same settings (one run was 20%).

## What did not help (removed; still on branch `inference-fixes-2026-11-27-full-history`)

- Constant arm position offsets and an "approach push": the policy is relative and closed-loop, so
  offsets either accumulate (drift) or get steered away.
- Tilt gain and grip boost: looked good on the first (faulty) success check, not better on the corrected one.
- Lead gain 3.0: worse.

## Known limits of this checkpoint on this hardware

- **Right column / near the robot** (robot's right, back row of the 9-spot grid) fails almost always.
  There the demos turned the hand clockwise, and front/middle-right have only 2 and 5 demos
  (back-right has both turning directions). The policy turns the other way; the arm follows it fine.
- After a miss the policy does not re-find a cube that rolled away: no recovery demos in its training data.
- Twist about the tool axis has a dead zone (the arm twists reliably only for commands > 12-15 deg ahead).

## Next steps

Record targeted demos with today's setup (Desk as above, current hand calibration): right column
(10-15 per spot, one consistent turning direction), back row, plus the 11 recovery demos in
`tmp/recovery_red_processed_10hz_lzf`; fine-tune `checkpoint_150` on them mixed with the old demos.
