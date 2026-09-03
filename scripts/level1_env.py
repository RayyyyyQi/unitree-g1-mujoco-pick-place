#!/usr/bin/env python3
"""Reusable Level-1 environment state, reset, and reproducibility helpers.

The reset restores both robot servo targets and the cube free-joint state.
Optional seeded XY perturbations support bounded evaluation variants while the
default path remains fully deterministic.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from smoke_test import reset as reset_robot_servos

# Shared baseline paths used by reset demos and control-policy scripts.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/task1/baseline.xml"
DEFAULT_CONFIG = ROOT / "configs/level1_scene.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Helper: load ``path`` and return the parsed YAML scene configuration."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def freejoint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    """Helper: return the ``(qpos, qvel)`` offsets of free joint ``name``."""
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise KeyError(f"Missing free joint: {name}")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def reset_episode(model: mujoco.MjModel, data: mujoco.MjData,
                  config: dict | None = None, seed: int = 0,
                  randomize: bool = False) -> None:
    """Reset robot and cube state, optionally using a seeded XY perturbation."""
    # 1. Reset MuJoCo and initialize every robot servo at its home joint value.
    reset_robot_servos(model, data)

    # 2. Resolve the cube's seven qpos and six qvel entries through its free joint.
    cube_qpos, cube_dof = freejoint_addresses(model, "red_cube_freejoint")

    # 3. Restore the configured cube position, or use the MJCF default pose.
    if config is None:
        cube_xyz = model.qpos0[cube_qpos:cube_qpos + 3].copy()
    else:
        cube_xyz = np.asarray(
            config["red_cube"]["position"], dtype=float
        ).copy()

    # 4. Optionally apply a reproducible bounded XY perturbation for evaluation.
    if randomize:
        rng = np.random.default_rng(seed)
        cube_xyz[:2] += rng.uniform([-0.02, -0.02], [0.02, 0.02])

    # 5. Reset cube translation, orientation, velocity, and acceleration.
    data.qpos[cube_qpos:cube_qpos + 3] = cube_xyz
    data.qpos[cube_qpos + 3:cube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cube_dof:cube_dof + 6] = 0.0
    data.qacc[:] = 0.0

    # 6. Recompute all derived positions and sensor quantities immediately.
    mujoco.mj_forward(model, data)


def episode_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray | float]:
    """Helper: return copies of the simulator fields used to verify reset."""
    # Copies prevent later simulation steps from mutating the saved snapshot.
    return {
        "time": float(data.time),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
    }


def max_state_error(first: dict, second: dict) -> dict[str, float]:
    """Helper: return maximum absolute errors between two state snapshots."""
    # A correct deterministic reset should return zero for all four fields.
    return {
        "time": abs(float(first["time"]) - float(second["time"])),
        "qpos": float(np.max(np.abs(first["qpos"] - second["qpos"]))),
        "qvel": float(np.max(np.abs(first["qvel"] - second["qvel"]))),
        "ctrl": float(np.max(np.abs(first["ctrl"] - second["ctrl"]))),
    }


def perturb_episode(model: mujoco.MjModel, data: mujoco.MjData, config: dict) -> None:
    """Modify robot and cube state to create a visible reset-test perturbation."""
    # Change both the elbow joint state and its servo command.
    elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow_joint")
    elbow_qpos = model.jnt_qposadr[elbow]
    data.qpos[elbow_qpos] = 0.45
    actuator = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_right_elbow_joint"
    )
    data.ctrl[actuator] = 0.45

    # Move and rotate the cube above the tray, and assign nonzero velocity.
    cube_qpos, cube_dof = freejoint_addresses(model, "red_cube_freejoint")
    tray = np.asarray(config["tray"]["position"], dtype=float)
    data.qpos[cube_qpos:cube_qpos + 3] = tray + [0.0, 0.0, 0.22]
    data.qpos[cube_qpos + 3:cube_qpos + 7] = [0.9238795, 0.0, 0.0, 0.3826834]
    data.qvel[cube_dof:cube_dof + 6] = [0.05, -0.03, 0.0, 0.0, 0.0, 0.5]

    # Update derived state so the perturbation is visible before stepping.
    mujoco.mj_forward(model, data)
