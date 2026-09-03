#!/usr/bin/env python3
"""Build the fixed-base G1 29-DoF + Dex1 model used by Task 1.

The official Unitree URDF is compiled through MuJoCo, placed at its nominal
standing height, and augmented with conservative position servos, a floor,
lighting, a camera, and a right-gripper reference site.  The generated XML
uses a repository-relative mesh path so that a fresh clone remains portable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

# Repository paths for the official Unitree source model and generated MJCF.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "third_party/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf"
DEFAULT_OUTPUT = ROOT / "models/base/g1_29dof_dex1_fixed.xml"


def mesh_assets(urdf: Path) -> dict[str, bytes]:
    """Helper: return the Unitree mesh files required to compile ``urdf``."""
    return {f"meshes/{p.name}": p.read_bytes() for p in (urdf.parent / "meshes").iterdir() if p.is_file()}


def servo_parameters(name: str) -> tuple[float, float, float]:
    """Helper: return conservative ``(kp, kv, force_limit)`` values for a joint."""
    # Use joint-group-specific gains because fingers, wrists, torso, and legs
    # have different inertia and safe force requirements.
    if "dex1_finger" in name:
        return 300.0, 12.0, 20.0
    if "hip" in name or "knee" in name:
        return 180.0, 18.0, 88.0
    if "ankle" in name:
        return 100.0, 10.0, 35.0
    if "waist" in name:
        return 300.0, 20.0, 50.0
    if "wrist" in name:
        return 60.0, 5.0, 13.4
    return 150.0, 10.0, 25.0


def build(urdf: Path, output: Path) -> Path:
    """Build a fixed-base MJCF from ``urdf`` and return its ``output`` path."""
    urdf, output = urdf.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Compile the official Unitree URDF and all referenced mesh assets.
    # Unitree's URDF already prefixes filenames with meshes/. Replacing the
    # meshdir avoids meshes/meshes/<file>, without modifying third-party code.
    text = urdf.read_text(encoding="utf-8").replace(
        '<compiler meshdir="meshes" discardvisual="false"/>',
        '<compiler meshdir="." discardvisual="false"/>',
    )
    compiled = mujoco.MjModel.from_xml_string(text, mesh_assets(urdf))
    temporary = output.with_suffix(".compiled.tmp.xml")
    try:
        mujoco.mj_saveLastXML(str(temporary), compiled)
        tree = ET.parse(temporary)
    finally:
        temporary.unlink(missing_ok=True)

    # 2. Store a relative mesh path so the generated model works after cloning.
    root = tree.getroot()
    relative_meshdir = os.path.relpath(urdf.parent, output.parent)
    root.find("compiler").set("meshdir", relative_meshdir)

    # 3. Set a stable 500-Hz physics configuration with standard gravity.
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.attrib.update(timestep="0.002", integrator="implicitfast", gravity="0 0 -9.81")

    # 4. Configure contact-point and contact-force visualization in the viewer.
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "scale", contactwidth="0.015", contactheight="0.03", forcewidth="0.01")
    ET.SubElement(visual, "map", force="0.01")

    # 5. Create a visible checkerboard material for the ground plane.
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="ground_texture", type="2d", builtin="checker",
                  rgb1="0.18 0.22 0.25", rgb2="0.08 0.10 0.12", width="256", height="256")
    ET.SubElement(asset, "material", name="ground_material", texture="ground_texture",
                  texrepeat="4 4", texuniform="true", reflectance="0.05")

    # 6. Fix the robot base and place the pelvis at the nominal standing height.
    world = root.find("worldbody")
    # MuJoCo fuses the fixed pelvis and fixed Dex1 base during URDF import.
    # Wrap every imported top-level robot element to place the fused pelvis at
    # the stock G1 standing height, without introducing a free joint.
    imported = list(world)
    fixed_base = ET.Element("body", name="fixed_base", pos="0 0 0.793")
    for element in imported:
        world.remove(element)
        fixed_base.append(element)
    world.append(fixed_base)

    # 7. Add the floor, lighting, and a fixed inspection camera.
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  material="ground_material", friction="0.8 0.02 0.001")
    ET.SubElement(world, "light", name="key_light", pos="1 -1 2.5", dir="-0.4 0.4 -1", directional="true")
    ET.SubElement(world, "camera", name="smoke_camera", pos="2 -2 1.45",
                  mode="targetbody", target="torso_link")

    # 8. Attach a site to the right gripper as the end-effector reference point.
    hand = world.find(".//body[@name='right_wrist_yaw_link']")
    if hand is None:
        raise RuntimeError("Compiled model is missing the right wrist body")
    ET.SubElement(hand, "site", name="right_grasp_site", pos="0.1365 0 0",
                  size="0.008", rgba="1 0.2 0.2 1")

    # 9. Give every actuated joint a bounded position servo for direct control.
    old = root.find("actuator")
    if old is not None:
        root.remove(old)
    actuators = ET.SubElement(root, "actuator")
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if not name:
            continue
        kp, kv, force = servo_parameters(name)
        ET.SubElement(actuators, "position", name=f"servo_{name}", joint=name,
                      kp=f"{kp:g}", kv=f"{kv:g}", ctrllimited="true",
                      ctrlrange=joint.get("range", "-3.14159 3.14159"),
                      forcelimited="true", forcerange=f"{-force:g} {force:g}")

    # 10. Save and immediately reload the MJCF to catch invalid output early.
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(output))
    print(f"Built: {output}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} joints={model.njnt} cameras={model.ncam}")
    return output


def main() -> None:
    """CLI entry point for building the base robot model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.urdf, args.output)


if __name__ == "__main__":
    main()
