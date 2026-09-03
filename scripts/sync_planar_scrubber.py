#!/usr/bin/env python3
"""Plan and replay the staged approach from home to the cube.

The solver first aligns waist yaw and forearm direction, then computes a
minimum-jerk shoulder/elbow path in the selected arm plane.  The complete path
is precomputed so viewer playback can be paused, rewound, and rerun exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
from mujoco import viewer
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from level1_env import DEFAULT_MODEL, reset_episode
from smoke_test import minimum_jerk

# Replay rate and the bounded torso pitch used during the planar approach.
FPS = 60
WAIST_PITCH = 0.25


def unit(v):
    """Helper: return the normalized form of vector ``v``."""
    return v / max(np.linalg.norm(v), 1e-12)


def solve(model, data):
    """Plan the staged approach and return poses, path frames, and metrics."""
    # 1. Reset the episode and resolve the joints/sites used by the planner.
    reset_episode(model, data)
    home = data.qpos.copy()
    hand = model.site("right_grasp_site").id
    cube = model.site("red_cube_grasp_site").id
    elbow_body = model.body("right_elbow_link").id
    wrist_body = model.body("right_wrist_roll_link").id

    names = ("waist_yaw_joint", "right_shoulder_pitch_joint", "right_elbow_joint")
    ids = np.array([model.joint(name).id for name in names])
    adrs = model.jnt_qposadr[ids]
    lo = model.jnt_range[ids, 0] + 1e-5
    hi = model.jnt_range[ids, 1] - 1e-5
    pitch_adr = model.jnt_qposadr[model.joint("waist_pitch_joint").id]

    # 2. Jointly solve waist yaw, shoulder pitch, and elbow at the cube target.
    # Solve the actual planar terminal pose. Shoulder yaw/roll and all wrist joints
    # stay at home; waist yaw is therefore chosen from the real arm plane.
    def final_residual(q):
        """Helper: return terminal hand-to-cube position error for candidate joints."""
        data.qpos[:] = home
        data.qpos[pitch_adr] = WAIST_PITCH
        data.qpos[adrs] = q
        mujoco.mj_forward(model, data)
        return data.site_xpos[hand] - data.site_xpos[cube]

    terminal = least_squares(
        final_residual, [0.35, -0.8, 0.9], bounds=(lo, hi), max_nfev=5000
    ).x
    yaw = terminal[0]

    # With that correct plane selected, elbow aims the forearm while waist yaw and
    # elbow are animated simultaneously.
    elbow_id = ids[2]
    elbow_adr = adrs[2]

    def aim_error(elbow_q):
        """Helper: return squared forearm-to-target angular error for one elbow value."""
        data.qpos[:] = home
        data.qpos[adrs[0]] = yaw
        data.qpos[elbow_adr] = elbow_q
        mujoco.mj_forward(model, data)
        forearm = unit(data.xpos[wrist_body] - data.xpos[elbow_body])
        toward_cube = unit(data.site_xpos[cube] - data.xpos[elbow_body])
        angle = np.arccos(np.clip(np.dot(forearm, toward_cube), -1.0, 1.0))
        return float(angle * angle)

    elbow_aim = minimize_scalar(
        aim_error, bounds=tuple(model.jnt_range[elbow_id]), method="bounded"
    ).x
    aimed = home.copy()
    aimed[adrs[0]] = yaw
    aimed[elbow_adr] = elbow_aim
    data.qpos[:] = aimed
    mujoco.mj_forward(model, data)
    start_hand = data.site_xpos[hand].copy()
    target = data.site_xpos[cube].copy()

    # 3. Track a minimum-jerk Cartesian line using shoulder/elbow IK per frame.
    # Precompute the entire coordinated motion. During playback there is no online
    # optimization or wait-for-feedback step.
    planar_adrs = adrs[1:]
    planar_lo = lo[1:]
    planar_hi = hi[1:]
    previous = np.array([aimed[planar_adrs[0]], aimed[planar_adrs[1]]])
    reach_path = []
    path_errors = []
    for alpha in np.linspace(0.0, 1.0, 91)[1:]:
        blend = minimum_jerk(alpha)
        desired = (1.0 - blend) * start_hand + blend * target
        pitch = blend * WAIST_PITCH

        def frame_residual(q):
            """Helper: return Cartesian tracking and continuity errors for one frame."""
            data.qpos[:] = aimed
            data.qpos[pitch_adr] = pitch
            data.qpos[planar_adrs] = q
            mujoco.mj_forward(model, data)
            return np.r_[data.site_xpos[hand] - desired, 0.002 * (q - previous)]

        previous = least_squares(
            frame_residual,
            previous,
            bounds=(planar_lo, planar_hi),
            max_nfev=500,
        ).x
        qpos = aimed.copy()
        qpos[pitch_adr] = pitch
        qpos[planar_adrs] = previous
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        path_errors.append(np.linalg.norm(data.site_xpos[hand] - desired))
        reach_path.append(qpos)

    # 4. Replace the last sample with the exact solved terminal configuration.
    # Use the exact terminal solution for the last frame.
    final = home.copy()
    final[pitch_adr] = WAIST_PITCH
    final[adrs] = terminal
    reach_path[-1] = final
    data.qpos[:] = final
    mujoco.mj_forward(model, data)
    final_error = np.linalg.norm(data.site_xpos[hand] - data.site_xpos[cube])

    # 5. Record interpretable planning angles and path/terminal errors.
    metrics = {
        "waist_yaw_deg": np.degrees(yaw),
        "waist_pitch_rad": WAIST_PITCH,
        "waist_pitch_deg": np.degrees(WAIST_PITCH),
        "aim_elbow_deg": np.degrees(elbow_aim),
        "final_shoulder_pitch_deg": np.degrees(terminal[1]),
        "final_elbow_deg": np.degrees(terminal[2]),
        "max_precomputed_path_error_cm": 100 * max(path_errors),
        "final_touch_error_mm": 1000 * final_error,
    }
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    return home, aimed, np.asarray(reach_path), metrics


def timeline(home, aimed, reach_path):
    """Convert approach checkpoints into replay frames and phase indices."""
    # Stage 1: briefly hold home, then synchronize waist yaw and elbow aiming.
    frames = [home.copy() for _ in range(round(0.5 * FPS))]
    phases = {"1": len(frames)}
    # Waist yaw and elbow aim happen together in 1.1 seconds.
    for alpha in np.linspace(0.0, 1.0, round(1.1 * FPS) + 1)[1:]:
        blend = minimum_jerk(alpha)
        frames.append((1.0 - blend) * home + blend * aimed)
    frames.extend([aimed.copy() for _ in range(round(0.5 * FPS))])
    phases["2"] = len(frames)
    # Stage 2: replay the synchronized waist-pitch/shoulder/elbow reach path.
    # Waist pitch, shoulder pitch and elbow are already synchronized in this path.
    frames.extend(reach_path)
    frames.extend([reach_path[-1].copy() for _ in range(round(3.0 * FPS))])
    return np.asarray(frames), phases


def main():
    """CLI entry point for planning and interactively replaying the approach."""
    # 1. Load the requested scene and precompute the complete motion timeline.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    home, aimed, reach_path, metrics = solve(model, data)
    frames, phases = timeline(home, aimed, reach_path)

    # 2. Print numerical planning results and the viewer keyboard controls.
    print("Fast precomputed planar preview; no feedback loop; penetration allowed.")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print("Enter/Space=Run/Pause | Backspace/Delete=Reset (paused)")
    print("R or 0=Reset+Run | Left/Right=0.5s | ,/.=frame")
    print("1=aim | 2=reach | free mouse camera | no automatic loop")

    state = {"index": 0, "playing": True, "last_key": 0.0}

    def key_callback(keycode):
        """Handle playback, reset, phase-jump, rewind, and single-frame controls."""
        now = time.monotonic()
        if keycode in (32, 257, 335):  # Space, Enter, keypad Enter
            if state["index"] >= len(frames) - 1:
                state["index"] = 0
            state["playing"] = not state["playing"]
            print("[run]" if state["playing"] else "[pause]")
        elif keycode == 263:
            state["playing"] = False
            state["index"] = max(0, state["index"] - FPS // 2)
        elif keycode == 262:
            state["playing"] = False
            state["index"] = min(len(frames) - 1, state["index"] + FPS // 2)
        elif keycode == 44:
            state["playing"] = False
            state["index"] = max(0, state["index"] - 1)
        elif keycode == 46:
            state["playing"] = False
            state["index"] = min(len(frames) - 1, state["index"] + 1)
        elif keycode in (49, 50):
            state["playing"] = False
            state["index"] = phases[chr(keycode)]
        elif keycode in (259, 261):  # Backspace or Delete: reset, then wait for Run
            state["index"] = 0
            state["playing"] = False
            print("[reset] paused at frame 0; press Enter or Space to run")
        elif keycode in (48, 82) and now - state["last_key"] > 0.4:
            state["index"] = 0
            state["playing"] = True
            state["last_key"] = now
            print("[replay] restarted from frame 0")

    # 3. Open a freely movable camera and replay stored qpos frames at 60 FPS.
    last_index = -1
    with viewer.launch_passive(model, data, key_callback=key_callback) as handle:
        # Start from a useful overview, but keep a true free camera so the user
        # can orbit, pan and zoom throughout playback.
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        handle.cam.lookat[:] = [0.45, 0.0, 0.85]
        handle.cam.distance = 2.35
        handle.cam.azimuth = 145.0
        handle.cam.elevation = -22.0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        while handle.is_running():
            index = state["index"]
            if index != last_index:
                data.qpos[:] = frames[index]
                data.qvel[:] = 0.0
                data.time = index / FPS
                for actuator in range(model.nu):
                    joint = model.actuator_trnid[actuator, 0]
                    data.ctrl[actuator] = data.qpos[model.jnt_qposadr[joint]]
                mujoco.mj_forward(model, data)
                last_index = index
            handle.sync()
            if state["playing"]:
                if state["index"] < len(frames) - 1:
                    state["index"] += 1
                else:
                    state["playing"] = False
                    print("[end] paused")
            time.sleep(1 / FPS)


if __name__ == "__main__":
    main()
