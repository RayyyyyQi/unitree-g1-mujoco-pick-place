#!/usr/bin/env python3
"""Visualize deterministic Level-1 reset after perturbing robot and cube.

The loop alternates between the initial episode, an obvious artificial
post-interaction state, and a complete reset so both object and controller
state restoration can be inspected in the MuJoCo viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
from mujoco import viewer

from level1_env import DEFAULT_MODEL, load_config, perturb_episode, reset_episode


def simulate(model, data, handle, seconds: float) -> bool:
    """Helper: simulate for ``seconds`` and return whether the viewer remains open."""
    # Match wall-clock playback to the MuJoCo timestep while updating the viewer.
    end = time.monotonic() + seconds
    while handle.is_running() and time.monotonic() < end:
        before = time.monotonic()
        mujoco.mj_step(model, data)
        handle.sync()
        time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - before)))
    return handle.is_running()


def main() -> None:
    """CLI entry point for the continuous perturb-and-reset viewer demonstration."""
    # 1. Read the task-model path and create MuJoCo model/data objects.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    # 2. Load the baseline configuration and establish the initial episode state.
    config = load_config()
    reset_episode(model, data, config)

    print("Loop: RESET HOME (4s) -> PERTURB robot/cube (4s) -> full RESET")

    # 3. Open a passive viewer using the fixed task camera and contact markers.
    with viewer.launch_passive(model, data) as handle:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "task_camera")
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        handle.cam.fixedcamid = camera_id
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        # 4. Alternate between home, perturbed, and fully reset episode states.
        while handle.is_running():
            print("[phase] deterministic RESET / home state")
            if not simulate(model, data, handle, 4.0):
                break
            print("[phase] perturb elbow/control and drop cube over tray")
            perturb_episode(model, data, config)
            if not simulate(model, data, handle, 4.0):
                break
            print("[phase] reset complete episode")
            reset_episode(model, data, config)


if __name__ == "__main__":
    main()
