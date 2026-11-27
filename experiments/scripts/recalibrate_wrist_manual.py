#!/usr/bin/env python3
"""Recalibrate only the ORCA wrist, capturing its end positions by hand.

Use this when the wrist motor cannot drive itself onto its hard stops (for
example the hand's weight stops it early), so the calibrated range is smaller
than what the wrist can physically do. Torque stays OFF for the whole run:
you move the wrist to each end yourself and press Enter. Finger calibration is
not touched.

Stop the hand_control node first (it holds the serial port).

    pixi run python src/rwr_system_imitate2/experiments/scripts/recalibrate_wrist_manual.py

Afterwards it asks you to hold the wrist in the demo posture once more and
prints its reading, which is what you need for `wrist_offset_deg`.
"""
import argparse
import os
import shutil
import time

from orca_core import load_hand

DEFAULT_CONFIG = os.path.expanduser(
    "~/rwr_humble_ws/install/hand_control/share/hand_control/orca/configs/orcahand_v2/right/config.yaml"
)
DESCRIBE = {
    "flex": "FLEXED: palm side of the hand bent towards the inner forearm, as far as it goes",
    "extend": "EXTENDED (bent back): back of the hand towards the forearm, as far as it goes",
}


def wrist_deg(hand):
    return hand._motor_to_joint_pos(hand.get_motor_state().position).get("wrist")


def prompt(info):
    direction = info.get("direction")
    print(f"\n>>> Hold the wrist {DESCRIBE.get(direction, direction)}.")
    print("    Keep it there and press Enter to record ('s' + Enter = skip, 'a' + Enter = abort).")
    answer = input("    ").strip().lower()
    return {"s": "skip", "a": "abort"}.get(answer, "record")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="hand config.yaml (calibration.yaml sits next to it)")
    parser.add_argument("--mock", action="store_true", help="dry run against orca_core's mock hand, writes nothing")
    parser.add_argument("--read-only", action="store_true", help="skip calibration, only take the posture readings")
    args = parser.parse_args()

    calib = os.path.join(os.path.dirname(os.path.realpath(args.config)), "calibration.yaml")
    calib_installed = os.path.join(os.path.dirname(args.config), "calibration.yaml")
    if not args.mock and not args.read_only:
        for path in {calib, calib_installed}:
            if os.path.exists(path):
                backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.copy2(path, backup)
                print(f"Backed up {path} -> {backup}")

    hand = load_hand(config_path=args.config, mock=args.mock)
    ok, message = hand.connect(interactive=False)
    if not ok:
        raise SystemExit(f"Could not connect to the hand: {message}")
    try:
        motor_id = hand.config.joint_to_motor_map["wrist"]
        print(f"Wrist reading with the current calibration: {wrist_deg(hand)}")
        print("Torque will be OFF. Support the hand: the wrist can drop under its own weight.")
        if not args.read_only:
            before = list(hand.calibration.motor_limits_dict.get(motor_id))
            # Blocking calibrate() returns None; success shows as changed wrist limits.
            hand.calibrate(
                blocking=True, force_wrist=True, joints=["wrist"], manual=True,
                prompt_callback=prompt, persist=not args.mock,
            )
            after = list(hand.calibration.motor_limits_dict.get(motor_id))
            if after == before:
                raise SystemExit("Wrist limits unchanged (aborted or refused); calibration.yaml keeps its previous values.")
            print(f"\nWrist motor limits: {before} -> {after}")
            print(f"wrist_calibrated: {hand.calibration.wrist_calibrated}")

        hand.disable_torque()
        print("\n>>> Now hold the wrist in the posture it had at the START of the demos (bent back),")
        input("    then press Enter. ")
        print(f"Wrist reading in that posture: {wrist_deg(hand):.1f} deg")
        print("   -> wrist_offset_deg = (this reading) - 19   (19 deg was the demos' start reading)")
        input("\n>>> Now hold it straight (hand in line with the forearm) and press Enter. ")
        print(f"Wrist reading straight: {wrist_deg(hand):.1f} deg")
    finally:
        hand.disable_torque()
        hand.disconnect()
        print("Disconnected, torque off. Restart hand_control to use the new calibration.")


if __name__ == "__main__":
    main()
