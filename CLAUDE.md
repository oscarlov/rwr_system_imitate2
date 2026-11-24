# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ROS 2 control stack for the ORCA v2 tendon-driven hand (RWR 2026 course, ETH SRL): Rokoko-glove teleoperation of the hand and a Franka arm, demonstration recording to HDF5, and running trained IL/RL policies on the hardware. See `README.md` for task workflows and `docs/rwr26_teleop.md` for the station setup, recording and validation procedure.

## Workspace / environment

- This checkout lives at `~/rwr_humble_ws/src/rwr_system_imitate2` (the docs say `src/rwr_system`; substitute this path). The colcon workspace root is `~/rwr_humble_ws`.
- Environment is pixi (RoboStack, **ROS 2 Humble**, Python 3.12), defined in `~/rwr_humble_ws/pixi.toml`. Its activation sources `install/setup.bash`, so run everything through `pixi run …` or `pixi shell` from the workspace root.
- The Python package `faive_system` is pip-installed **editable** from this repo root (`package-dir = {faive_system = "."}`), so code imports as `from faive_system.src.common.utils import …`. Python edits take effect without reinstalling.
- `orca_core==0.5.2` is installed with `--no-deps` because it declares a newer numpy than the `numpy==1.26.4` pin. `pip check` reporting that conflict is expected; don't "fix" it by changing numpy.

## Commands

Run from `~/rwr_humble_ws`:

```bash
pixi run colcon build --symlink-install                     # build all packages
pixi run colcon build --symlink-install --packages-select retargeter
pixi run pip install -e './src/rwr_system_imitate2[all]'    # (re)install the Python package
# if the build breaks after changes: rm -rf build install log, then rebuild
```

Tests (pytest; no hardware is touched):

```bash
# ORCA chirp acquisition logic (pure unit tests)
pixi run python -m pytest -q src/rwr_system_imitate2/tests/test_orca_chirp.py
# ROS integration tests: spawn nodes/launches, so isolate the DDS domain
ROS_DOMAIN_ID=181 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  pixi run python -m pytest -q src/rwr_system_imitate2/tests/offline
# single test
pixi run python -m pytest -q src/rwr_system_imitate2/tests/offline/test_data_collection.py::test_synchronization_rejects_disjoint_streams
```

`tests/offline` imports the built `ingress` messages, so build first. `tests/retargeter_test.py` is not a pytest test; it is a ROS node (`ros2 launch faive_test retargeting_test.launch.py`) that replays recorded glove poses from `tests/artifacts/` through the retargeter.

No linter is configured.

## Architecture

Each folder under `src/`, plus `experiments/`, `tests/` (`faive_test`) and `third_party/franka_msgs`, is its own **ament_cmake** package. Python scripts are exposed to `ros2 run` only through `install(PROGRAMS …)` in that package's `CMakeLists.txt` (and need a shebang and the executable bit). A new node script must be added there and the package rebuilt. Launch files and configs are installed to `share/` the same way. Shared library code is imported through `faive_system`, not through the ROS package.

Data flow for teleop (wired in `experiments/launch/teleop_rokoko.launch.py`; `record_demonstration.launch.py` adds the cameras and logger; `run_policy_inference.launch.py` swaps the human for `experiments/model_inference_node.py`):

1. **ingress** (`rokoko_node.py`): UDP JSON from Rokoko Studio (port 14043) → MANO keypoints plus wrist/forearm `PoseStamped` on `/ingress/{left,right}/…`. `oakd_node.py`/`zed_node.py` publish RGB(-D) and `ingress/msg/CameraIntrinsics` (the package's only custom message).
2. **retargeter** (`retargeter_node.py` wraps `retargeter.py`): a gradient-based optimization in torch/pytorch_kinematics over the hand URDF that fits robot fingertip/keypoint vectors to the normalized MANO keypoints. It is configured by `experiments/cfgs/retargeter/retargeter_cfgs_orcahand_v2.yaml` and `mano_adjustment_orcahand_v2.yaml`, and outputs 17 joint angles in **hardware joint order** as `Float32MultiArray` on `/right/hand/policy_output`. Enabled and paused with the `/right/retargeter/set_enabled` (SetBool) service; it starts disabled in the launch files.
3. **hand_control** (`hand_control_node.py` → `orca/hand_controller_v2.py` → `orca_core`): drives the motors over USB serial and publishes `/right/hand/joint_{position,velocity}_from_motor`. It exposes the estop and torque services. Per-hand calibration lives in `orca/configs`. Without `calibration.yaml`, the controller auto-calibrates and **moves the physical hand**.
4. **Arm**: `experiments/run_teleop_rokoko.py` maps the glove wrist to `/right/arm/end_effector_pose_cmd`. The Franka Cartesian controller is external and publishes `/right/arm/end_effector_pose`.
5. **keyboard_runner.py**: a state machine for operators (or foot pedals) that calls the retargeter, teleop pause/resume and logger services.
6. **logger**: `logger_node.py` records the topics listed in `experiments/cfgs/logger/*.yaml` to a rosbag and converts it to `.h5` with `bag2h5_converter.py`. It is controlled by the `/demo_logger/{start,stop,save}_recording` Trigger services. `syncronize_data.py` (spelling is intentional) resamples a folder of raw `.h5` files onto a common rate over their shared time interval, using ROS receive timestamps. `experiments/scripts/inspect_demonstration.py` validates raw and processed files.

Policy inference configs (observation/action topic specs) are `experiments/cfgs/inference_config_*.yaml`. Camera MXID→view mappings are `experiments/cfgs/cams/*.yaml`. RL cube-spin inference and the cube pose detector are in `experiments/rl_inference/`. Robot models (URDF/xacro, meshes) are in `src/viz/models/`.

## Hardware safety conventions

The code assumes it may be connected to a real hand and arm. Keep any code path that tests or previews a change from opening serial ports or publishing arm commands, as the existing tests do (`preview:=true`, `teleop_arm:=false`, `enable_hand_controller:=false`, and the chirp preview mode). Software pause holds the last target; it is not an e-stop.
