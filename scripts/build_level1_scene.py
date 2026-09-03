#!/usr/bin/env python3
"""Build a complete Level-1 pick-and-place scene from a YAML configuration.

This script combines the fixed-base G1 model with the table, dynamic red cube,
tray, task markers, and two reproducible camera views.  Changing only the YAML
file is sufficient to rebuild the baseline and most evaluation variants.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from build_smoke_model import DEFAULT_OUTPUT as BASE_MODEL

# Default inputs and output for rebuilding the baseline scene.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/level1_scene.yaml"
DEFAULT_OUTPUT = ROOT / "models/task1/baseline.xml"


def values(items) -> str:
    """Helper: return numeric ``items`` as a space-separated MJCF string."""
    return " ".join(f"{float(x):g}" for x in items)


def camera_xyaxes(position, look_at) -> str:
    """Helper: return camera axes for ``position`` aimed at ``look_at``."""
    # Construct an orthonormal camera frame from its position and focal point.
    position, target = np.asarray(position, float), np.asarray(look_at, float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return values(np.concatenate([right, up]))


def add_table(world: ET.Element, cfg: dict) -> float:
    """Add the configured table to ``world`` and return its surface height."""
    # The tabletop and legs are fixed collision geoms in the world frame.
    center, half = cfg["center"], cfg["half_size"]
    rgba = values(cfg["color"])
    ET.SubElement(world, "geom", name="table_top", type="box", pos=values(center),
                  size=values(half), rgba=rgba, friction="0.9 0.02 0.001")
    leg_half = cfg["leg_half_size"]
    for index, (xsign, ysign) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        x = center[0] + xsign * (half[0] - 0.055)
        y = center[1] + ysign * (half[1] - 0.055)
        z = leg_half[2]
        ET.SubElement(world, "geom", name=f"table_leg_{index}", type="box",
                      pos=values([x, y, z]), size=values(leg_half), rgba=rgba,
                      friction="0.9 0.02 0.001")
    return center[2] + half[2]


def add_cube(world: ET.Element, cfg: dict) -> None:
    """Add the configured dynamic cube and its reference sites to ``world``."""
    # A free joint lets MuJoCo determine the cube's motion from contact forces.
    body = ET.SubElement(world, "body", name="red_cube", pos=values(cfg["position"]))
    ET.SubElement(body, "freejoint", name="red_cube_freejoint")
    half = float(cfg["half_size"])
    ET.SubElement(body, "geom", name="red_cube_geom", type="box",
                  size=values([half] * 3), mass=f"{float(cfg['mass']):g}",
                  friction=values(cfg["friction"]), rgba=values(cfg["color"]))
    # Sites are massless reference points used by IK, evaluation, and the viewer.
    ET.SubElement(body, "site", name="red_cube_center", type="sphere",
                  pos="0 0 0", size="0.006", rgba="1 1 0 1")
    # Visible top-surface projection of the true cube center. Placement logic
    # uses the body's actual center x/y; this cyan site is only its visual aid.
    ET.SubElement(body, "site", name="red_cube_place_marker", type="sphere",
                  pos=values([0, 0, half + 0.002]), size="0.005", rgba="0 1 1 1")
    ET.SubElement(body, "site", name="red_cube_grasp_site", type="sphere",
                  pos=values([0, 0, half + 0.01]), size="0.007", rgba="1 0.8 0 1")


def add_tray(world: ET.Element, cfg: dict) -> None:
    """Add the configured collision tray and target site to ``world``."""
    body = ET.SubElement(world, "body", name="tray", pos=values(cfg["position"]))
    hx, hy = map(float, cfg["inner_half_size"])
    wall, height = float(cfg["wall_thickness"]), float(cfg["wall_height"])
    bottom = float(cfg["bottom_thickness"])
    rgba = values(cfg["color"])
    # Build the tray as one bottom and four fixed collision walls.
    ET.SubElement(body, "geom", name="tray_bottom", type="box", pos=values([0, 0, bottom]),
                  size=values([hx + wall, hy + wall, bottom]), rgba=rgba,
                  friction="0.9 0.02 0.001")
    for name, pos, size in (
        ("tray_wall_front", [hx + wall, 0, height], [wall, hy + 2 * wall, height]),
        ("tray_wall_back", [-hx - wall, 0, height], [wall, hy + 2 * wall, height]),
        ("tray_wall_left", [0, hy + wall, height], [hx, wall, height]),
        ("tray_wall_right", [0, -hy - wall, height], [hx, wall, height]),
    ):
        ET.SubElement(body, "geom", name=name, type="box", pos=values(pos),
                      size=values(size), rgba=rgba, friction="0.9 0.02 0.001")
    ET.SubElement(body, "site", name="tray_target_site", type="sphere",
                  pos=values([0, 0, 2 * bottom + 0.015]), size="0.01", rgba="0 1 0 1")


def build(base: Path, config_path: Path, output: Path) -> Path:
    """Build a task MJCF from ``base`` and ``config_path``, returning ``output``."""
    # 1. Load all scene dimensions, physical parameters, and poses from YAML.
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # 2. Start from the reusable fixed-base robot model.
    tree = ET.parse(base)
    root = tree.getroot()
    root.set("model", "g1_level1_red_cube_to_tray")
    world = root.find("worldbody")

    # 3. Add the shared table and an optional raised cube pedestal.
    add_table(world, cfg["table"])
    pedestal = cfg.get("cube_pedestal")
    if pedestal:
        ET.SubElement(
            world, "geom", name="cube_pedestal", type="box",
            pos=values(pedestal["center"]),
            size=values(pedestal["half_size"]),
            rgba=values(pedestal["color"]),
            friction="0.9 0.02 0.001",
        )

    # 4. Add the dynamic manipulation object and fixed destination tray.
    add_cube(world, cfg["red_cube"])
    add_tray(world, cfg["tray"])

    # 5. Add the primary fixed task camera from its position and look-at point.
    camera = cfg["task_camera"]
    ET.SubElement(world, "camera", name="task_camera", pos=values(camera["position"]),
                  xyaxes=camera_xyaxes(camera["position"], camera["look_at"]))

    # 6. Add the optional second view used in the submitted demonstration videos.
    rear_right = cfg.get("task_camera_rear_right")
    if rear_right:
        ET.SubElement(
            world, "camera", name="task_camera_rear_right",
            pos=values(rear_right["position"]),
            xyaxes=camera_xyaxes(rear_right["position"], rear_right["look_at"]),
        )

    # 7. Mark the camera focus/workspace center for visual scene inspection.
    ET.SubElement(world, "site", name="task_workspace_center", type="sphere",
                  pos=values(camera["look_at"]), size="0.008", rgba="1 0 1 1")

    # 8. Save and reload the scene so malformed MJCF fails during generation.
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(output))
    print(f"Built: {output.resolve()}")
    print(f"bodies={model.nbody} geoms={model.ngeom} sites={model.nsite} cameras={model.ncam}")
    print(f"table surface z={cfg['table']['center'][2] + cfg['table']['half_size'][2]:.3f} m")
    print(f"red cube xyz={cfg['red_cube']['position']}")
    print(f"tray xyz={cfg['tray']['position']}")
    return output


def main() -> None:
    """CLI entry point for building one configured task scene."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.base, args.config, args.output)


if __name__ == "__main__":
    main()
