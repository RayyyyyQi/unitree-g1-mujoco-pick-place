#!/usr/bin/env python3
"""Create a rotated-workspace variant from the baseline task scene.

The table, cube, tray, markers, and fixed cameras are rotated together around
the robot.  This isolates the effect of initial waist-yaw alignment instead of
changing the arm's required reach distance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# The baseline scene is the source for every rigid workspace rotation.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "models/task1/baseline.xml"


def numbers(text):
    """Helper: parse a space-separated MJCF vector into a NumPy array."""
    return np.fromstring(text, sep=" ", dtype=float)


def values(array):
    """Helper: serialize a numeric array as a space-separated MJCF vector."""
    return " ".join(f"{float(value):.12g}" for value in array)


def rotate_xyz(text, rotation):
    """Helper: rotate the x/y entries of position ``text`` and return a string."""
    vector = numbers(text)
    vector[:2] = rotation @ vector[:2]
    return values(vector)


def main():
    """CLI entry point that creates and validates one rotated workspace variant."""
    # 1. Read the baseline scene, output path, and requested yaw angle.
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angle-deg", type=float, required=True)
    args = parser.parse_args()

    # 2. Build 2-D/3-D rotation matrices and the equivalent yaw quaternion.
    angle = np.radians(args.angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation2 = np.array([[cosine, -sine], [sine, cosine]])
    rotation3 = np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    yaw_quaternion = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])

    # 3. Load the baseline MJCF without changing the robot configuration.
    tree = ET.parse(args.source)
    root = tree.getroot()
    root.set("model", f"g1_level1_workspace_rot_{args.angle_deg:g}deg")
    world = root.find("worldbody")

    # 4. Rotate the tabletop and all four legs as one rigid workspace.
    for name in ("table_top", "table_leg_0", "table_leg_1", "table_leg_2", "table_leg_3"):
        geom = world.find(f"geom[@name='{name}']")
        geom.set("pos", rotate_xyz(geom.get("pos"), rotation2))
        geom.set("quat", values(yaw_quaternion))

    # 5. Rotate the cube and tray while preserving their relative geometry.
    for name in ("red_cube", "tray"):
        body = world.find(f"body[@name='{name}']")
        body.set("pos", rotate_xyz(body.get("pos"), rotation2))
        body.set("quat", values(yaw_quaternion))

    # 6. Move the workspace-center marker to the corresponding rotated point.
    marker = world.find("site[@name='task_workspace_center']")
    marker.set("pos", rotate_xyz(marker.get("pos"), rotation2))

    # 7. Rotate both camera positions and orientations with the workspace.
    for name in ("task_camera", "task_camera_rear_right"):
        camera = world.find(f"camera[@name='{name}']")
        if camera is None:
            continue
        camera.set("pos", rotate_xyz(camera.get("pos"), rotation2))
        axes = numbers(camera.get("xyaxes")).reshape(2, 3)
        axes = (rotation3 @ axes.T).T
        camera.set("xyaxes", values(axes.reshape(-1)))

    # 8. Save, reload, and report key poses to validate the generated variant.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(args.output))
    print(f"Built: {args.output.resolve()}")
    print(f"workspace_rotation_deg={args.angle_deg:g}")
    print(f"red_cube_xy={model.body_pos[model.body('red_cube').id, :2]}")
    print(f"tray_xy={model.body_pos[model.body('tray').id, :2]}")


if __name__ == "__main__":
    main()
