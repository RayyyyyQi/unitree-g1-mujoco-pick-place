#!/usr/bin/env python3
"""Run safety and functionality checks on the fixed-base robot model.

The test validates loading, finite simulation state, deterministic reset,
minimum-jerk waist/right-arm motion, Dex1 opening and closing, and optional
camera rendering.  It writes machine-readable results for reproducibility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

# Default model/report paths and the joints exercised by the smoke test.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/base/g1_29dof_dex1_fixed.xml"
DEFAULT_REPORT = ROOT / "results/smoke_test/report.json"
CONTROLLED = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
FINGERS = ("right_dex1_finger_joint_1", "right_dex1_finger_joint_2")


@dataclass
class Check:
    """One named smoke-test outcome and its supporting measurements."""
    name: str
    passed: bool
    details: dict[str, object]


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    """Helper: return the MuJoCo ID for ``name`` or raise a readable error."""
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise KeyError(f"Model is missing {name!r}")
    return index


def joint_value(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    """Helper: return the current scalar position of joint ``name``."""
    joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[joint]])


def actuator_id(model: mujoco.MjModel, joint: str) -> int:
    """Helper: return the position-servo actuator ID for ``joint``."""
    return object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{joint}")


def minimum_jerk(alpha: float) -> float:
    """Helper: map normalized time ``alpha`` to a smooth minimum-jerk blend."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return 10 * a**3 - 15 * a**4 + 6 * a**5


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset physics and set every servo command to its reset joint position."""
    mujoco.mj_resetData(model, data)
    # Position-servo commands are initialized from the reset joint positions.
    for actuator in range(model.nu):
        joint = model.actuator_trnid[actuator, 0]
        data.ctrl[actuator] = data.qpos[model.jnt_qposadr[joint]]
    mujoco.mj_forward(model, data)


def finite(data: mujoco.MjData) -> bool:
    """Helper: return whether all relevant simulation values remain finite."""
    return math.isfinite(data.time) and all(
        np.all(np.isfinite(x)) for x in (data.qpos, data.qvel, data.qacc, data.ctrl)
    )


def step_for(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    """Simulate for ``seconds`` and fail immediately on numerical instability."""
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not finite(data):
            raise FloatingPointError(f"Non-finite state at t={data.time:.4f}s")


def move(model: mujoco.MjModel, data: mujoco.MjData,
         targets: dict[str, float], duration: float) -> None:
    """Move selected joints to ``targets`` smoothly over ``duration`` seconds."""
    starts = {name: float(data.ctrl[actuator_id(model, name)]) for name in targets}
    steps = round(duration / model.opt.timestep)
    for step in range(steps):
        blend = minimum_jerk((step + 1) / steps)
        for name, target in targets.items():
            data.ctrl[actuator_id(model, name)] = starts[name] + blend * (target - starts[name])
        mujoco.mj_step(model, data)
        if not finite(data):
            raise FloatingPointError(f"Non-finite state at t={data.time:.4f}s")


def render_camera(model: mujoco.MjModel, data: mujoco.MjData, output: Path) -> tuple[bool, str]:
    """Render the smoke camera and return ``(success, output_or_error)``."""
    try:
        renderer = mujoco.Renderer(model, height=360, width=480)
        renderer.update_scene(data, camera="smoke_camera")
        image = renderer.render()
        renderer.close()
        import imageio.v3 as iio
        output.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output, image)
        return True, str(output)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run(model_path: Path, report_path: Path, render: bool) -> list[Check]:
    """Run all checks, save a JSON report, and return the individual results."""
    # 1. Load the model, create runtime data, and record the initial reset state.
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    checks: list[Check] = []
    reset(model, data)
    initial_qpos, initial_ctrl = data.qpos.copy(), data.ctrl.copy()

    # 2. Verify that the model is fixed-base and every joint has an actuator.
    fixed = not any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    checks.append(Check("model_load_and_fixed_base", fixed and model.nu == model.njnt,
                        {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                         "joints": model.njnt, "fixed_base": fixed}))

    # 3. Run free-space physics and check for drift or non-finite state values.
    step_for(model, data, 2.0)
    speed = float(np.max(np.abs(data.qvel)))
    checks.append(Check("free_space_stability_2s", finite(data) and speed < 1.0,
                        {"time_s": float(data.time), "max_abs_qvel": speed,
                         "contacts": int(data.ncon)}))

    # 4. Exercise the waist and right arm with small minimum-jerk commands.
    # Small targets (about 2-9 degrees) plus a two-second minimum-jerk curve.
    targets = {
        "waist_yaw_joint": 0.05, "waist_roll_joint": 0.03, "waist_pitch_joint": -0.04,
        "right_shoulder_pitch_joint": 0.12, "right_shoulder_roll_joint": -0.08,
        "right_shoulder_yaw_joint": 0.08, "right_elbow_joint": 0.15,
        "right_wrist_roll_joint": 0.08, "right_wrist_pitch_joint": -0.06,
        "right_wrist_yaw_joint": 0.06,
    }
    move(model, data, targets, 2.0)
    step_for(model, data, 0.5)
    readback = {name: joint_value(model, data, name) for name in CONTROLLED}
    errors = {name: abs(readback[name] - target) for name, target in targets.items()}
    checks.append(Check("safe_motion_and_state_readback", finite(data) and max(errors.values()) < 0.03,
                        {"targets_rad": targets, "readback_rad": readback,
                         "max_tracking_error_rad": max(errors.values())}))

    # 5. Open and close both Dex1 fingers and measure their actual travel.
    # Opposite slide axes mean equal q targets move both fingers symmetrically.
    # Positive displacement moves both fingers away from the center (open).
    open_targets = {name: 0.018 for name in FINGERS}
    close_targets = {name: -0.015 for name in FINGERS}
    move(model, data, open_targets, 0.5)
    opened = {name: joint_value(model, data, name) for name in FINGERS}
    move(model, data, close_targets, 0.5)
    closed = {name: joint_value(model, data, name) for name in FINGERS}
    travel = {name: opened[name] - closed[name] for name in FINGERS}
    checks.append(Check("right_dex1_open_close", min(travel.values()) > 0.025,
                        {"opened_m": opened, "closed_m": closed, "travel_m": travel}))

    # 6. Reset again and compare state/control values with the initial snapshot.
    reset(model, data)
    qerr = float(np.max(np.abs(data.qpos - initial_qpos)))
    cerr = float(np.max(np.abs(data.ctrl - initial_ctrl)))
    checks.append(Check("deterministic_reset", qerr < 1e-12 and cerr < 1e-12 and data.time == 0,
                        {"time_s": float(data.time), "max_qpos_error": qerr, "max_ctrl_error": cerr}))

    # 7. Optionally verify off-screen RGB rendering from the fixed camera.
    if render:
        ok, detail = render_camera(model, data, report_path.parent / "smoke_camera.png")
        checks.append(Check("camera_rgb_render", ok, {"result": detail}))

    # 8. Save machine-readable evidence and print a concise pass/fail summary.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": str(model_path.resolve()), "passed": all(c.passed for c in checks),
              "checks": [asdict(c) for c in checks]}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}")
        print("      ", json.dumps(check.details, ensure_ascii=False))
    print(f"Report: {report_path.resolve()}")
    return checks


def main() -> None:
    """CLI entry point that returns a nonzero exit code if any check fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    checks = run(args.model, args.report, args.render)
    raise SystemExit(0 if all(c.passed for c in checks) else 1)


if __name__ == "__main__":
    main()
