#!/usr/bin/env python3
"""Summarize cube-grasp trials from inference logs (one log file per start/stop).

For each trial (a log file, or each attempt within it when auto_return was on) it reports the FIRST hand closing, while the cube still sits where it
was placed:
  - grasp error: arm position at that closing minus the position where the demos
    grasped a cube seen at the same place in the side camera (forward x, sideways y, up z).
    This estimate has a bias of a few cm of its own, so compare settings with each
    other rather than reading the numbers as absolute.
  - pushed: how far (pixels in the 224x224 side image) the cube moved before that closing.
  - lifted: whether the arm rose > 5 cm while the hand stayed closed.
  - closure: measured finger closure while closed (demos holding the cube: ~71 deg); >= 78 = closed on air.

    pixi run python summarize_trials.py --since 15:45            # all trials after 15:45 today
    pixi run python summarize_trials.py LOG1.hdf5 LOG2.hdf5      # specific logs
"""
import argparse
import glob
import os
import time

import cv2
import h5py
import numpy as np

DATASET = os.path.expanduser("~/rwr_humble_ws/tmp/cube_sorting_red_processed_10hz_lzf")
LOGS = os.path.expanduser("~/rwr_humble_ws/inference_logs")
FINGERS = [6, 9, 12, 15]          # index/middle/ring/pinky MCP: mean > 50 deg = hand closed


def find_cube(rgb):
    """Red dice in a 224x224 side-view RGB image -> (u, v), or None (the red wall is excluded by size/shape)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (0, 110, 70), (8, 255, 255)) | cv2.inRange(hsv, (172, 110, 70), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 60 <= a <= 1500 and w <= 3 * h and h <= 3 * w and a / (w * h) >= 0.45:
            if best is None or a > best[1]:
                best = (cen[i], a)
    return None if best is None else tuple(best[0])


def feat(x):
    x = np.atleast_2d(x)
    return np.c_[np.ones(len(x)), x, x ** 2, x[:, 0] * x[:, 1]]


def demo_grasp_model():
    """Fit side-camera cube pixel -> arm position at the demos' first hand closing."""
    xs, ps = [], []
    for path in sorted(glob.glob(os.path.join(DATASET, "*.h5"))):
        with h5py.File(path, "r") as f:
            hand = np.rad2deg(f["right/actions_hand"][()][:, FINGERS].mean(1))
            closed = np.where(hand > 50)[0]
            if not len(closed):
                continue
            s = closed[0]
            for t in range(s):
                c = find_cube(cv2.resize(f["observations/images/oakd_front_view/color"][t], (224, 224)))
                if c:
                    xs.append(c); ps.append(f["observations/right/qpos_arm"][s][:3])
                    break
    return np.linalg.lstsq(feat(np.array(xs)), np.array(ps), rcond=None)[0]


def side_image(f, i):
    return (f[f"step_{i}/observations/side_view_images"][()].transpose(1, 2, 0) * 255).astype(np.uint8)


def summarize(path, model):
    """One row per attempt: per log file, or per attempt_idx when auto_return split the log."""
    with h5py.File(path, "r") as f:
        n = len([k for k in f if k.startswith("step_")])
        if n < 20:
            return []
        arm_all = np.array([f[f"step_{i}/observations/qpos_arm"][()][:3] for i in range(n)])
        hand_all = np.array([np.rad2deg(f[f"step_{i}/actions/actions_hand"][()][FINGERS].mean()) for i in range(n)])
        meas_all = np.array([np.rad2deg(f[f"step_{i}/observations/qpos_hand"][()][FINGERS].mean()) for i in range(n)])
        if f"step_0/timing/attempt_idx" in f:
            att = np.array([f[f"step_{i}/timing/attempt_idx"][()] for i in range(n)])
            bounds = [0] + [i for i in range(1, n) if att[i] != att[i - 1]] + [n]
        else:
            bounds = [0, n]
        rows = []
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            label = os.path.basename(path) + (f" #{k}" if len(bounds) > 2 else "")
            rows.append(summarize_segment(f, a, b, arm_all[a:b], hand_all[a:b], meas_all[a:b], model, label))
        return [r for r in rows if r]


def summarize_segment(f, a, b, arm, hand, meas, model, label):
    n = b - a
    if n < 10:
        return None
    start_cube = next((c for c in (find_cube(side_image(f, a + i)) for i in range(min(n, 15))) if c), None)
    closes = np.where((hand[1:] > 50) & (hand[:-1] <= 50))[0] + 1
    row = {"log": label, "steps": n, "closes": len(closes), "start": arm[0]}
    if not len(closes) or start_cube is None:
        row["note"] = "no closing" if not len(closes) else "cube not seen at start"
        return row
    s = closes[0]
    cube = next((c for c in (find_cube(side_image(f, a + t)) for t in range(s - 1, max(s - 30, -1), -1)) if c), None)
    e = s
    while e + 1 < n and hand[e + 1] > 50:
        e += 1
    row["first_close_step"] = s
    row["error_cm"] = (arm[s] - (feat(cube if cube else start_cube) @ model)[0]) * 100
    row["pushed_px"] = np.hypot(*(np.subtract(cube, start_cube))) if cube else np.nan
    row["lifted"] = (arm[s:e + 1, 2].max() - arm[s, 2]) > 0.05
    # measured finger closure while closed: demos holding the cube ~71 deg (p90 75); >= 78 = closed on air
    row["closure"] = float(np.median(meas[s:e + 1]))
    any_lift = 0
    for c in closes:
        d = c
        while d + 1 < n and hand[d + 1] > 50:
            d += 1
        any_lift += (arm[c:d + 1, 2].max() - arm[c, 2]) > 0.05
    row["lifts_total"] = any_lift
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*")
    p.add_argument("--since", help="HH:MM today; summarize all logs written after it")
    a = p.parse_args()
    paths = list(a.logs)
    if a.since:
        t0 = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + a.since, "%Y-%m-%d %H:%M"))
        paths += [q for q in glob.glob(os.path.join(LOGS, "*", "*.hdf5")) if os.path.getmtime(q) >= t0]
    if not paths:
        raise SystemExit("no logs given")
    model = demo_grasp_model()
    rows = [r for q in sorted(paths, key=os.path.getmtime) for r in summarize(q, model)]
    print(f"{'trial (log #attempt)':26s} {'steps':>5s} {'closes':>6s}  first-close error dx/dy/dz (cm)   |err|  pushed(px) lifted  lifts  closure")
    errs = []
    for r in rows:
        if "error_cm" not in r:
            print(f"{r['log']:26s} {r['steps']:5d} {r['closes']:6d}  {r['note']}")
            continue
        d = r["error_cm"]; errs.append(d)
        print(f"{r['log']:26s} {r['steps']:5d} {r['closes']:6d}  {d[0]:+6.1f} {d[1]:+6.1f} {d[2]:+6.1f}            "
              f"{np.linalg.norm(d):5.1f}  {r['pushed_px']:8.1f}   {'yes' if r['lifted'] else 'no ':3s}   {r['lifts_total']}   {r['closure']:4.0f}{' (air)' if r['closure'] >= 78 else ''}")
    if errs:
        e = np.array(errs)
        print(f"\n{len(e)} trials: median first-close error dx {np.median(e[:, 0]):+.1f} dy {np.median(e[:, 1]):+.1f} "
              f"dz {np.median(e[:, 2]):+.1f} cm, median |err| {np.median(np.linalg.norm(e, axis=1)):.1f} cm")


if __name__ == "__main__":
    main()
