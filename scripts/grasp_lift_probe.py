#!/usr/bin/env python3
"""Probe the physical grasp before running the full transfer policy.

Starting from the planned reach pose, this script opens the Dex1 gripper,
retracts for clearance, descends, closes on the cube, and performs a short
lift.  Contact reporting distinguishes fingertip contact from an unintended
gripper-base collision.
"""

from __future__ import annotations

import argparse
import time

import mujoco
from mujoco import viewer
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from level1_env import DEFAULT_MODEL
from smoke_test import minimum_jerk
from sync_planar_scrubber import solve

# Closed-finger target shared with the complete pick-and-place controller.
CLOSE_TARGET = -0.010


def actuator_targets(model, qref):
    """Helper: convert full ``qref`` into clipped position-servo commands."""
    ctrl = np.zeros(model.nu)
    for actuator in range(model.nu):
        joint = model.actuator_trnid[actuator, 0]
        ctrl[actuator] = qref[model.jnt_qposadr[joint]]
    return np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def interpolate(a, b, alpha):
    """Helper: return a minimum-jerk interpolation from ``a`` to ``b``."""
    blend = minimum_jerk(np.clip(alpha, 0.0, 1.0))
    return (1.0 - blend) * a + blend * b


def finger_cube_contacts(model, data):
    """Helper: return ``(finger_count, base_hit)`` for current cube contacts."""
    cube_body = model.body("red_cube").id
    finger_bodies = {
        model.body("right_dex1_finger_link_1").id,
        model.body("right_dex1_finger_link_2").id,
    }
    hit = set()
    base_hit = False
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {
            model.geom_bodyid[contact.geom1],
            model.geom_bodyid[contact.geom2],
        }
        if cube_body not in bodies:
            continue
        hit.update(bodies & finger_bodies)
        other = next(iter(bodies - {cube_body}), None)
        if other == model.body("right_wrist_yaw_link").id:
            base_hit = True
    return len(hit), base_hit


def main():
    """CLI entry point for planning, executing, and optionally replaying the probe."""
    # 1. Load options, model, and the terminal pose from the approach planner.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--retract", type=float, default=0.0,
                        help="horizontal retreat before descent, meters")
    parser.add_argument("--tray-clearance", type=float, default=0.04,
                        help="desired cube-bottom clearance above tray wall, meters")
    parser.add_argument("--transfer-waist-only", action="store_true",
                        help="after lifting, lock everything except waist yaw and turn toward tray")
    parser.add_argument("--align-tray", action="store_true",
                        help="then align hand above tray using waist yaw/pitch and shoulder pitch")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    planning_data = mujoco.MjData(model)
    _, _, reach_path, _ = solve(model, planning_data)
    checkpoint = reach_path[-1].copy()

    shoulder_adr = model.jnt_qposadr[model.joint("right_shoulder_pitch_joint").id]
    elbow_id = model.joint("right_elbow_joint").id
    elbow_adr = model.jnt_qposadr[elbow_id]
    finger_adrs = np.array([
        model.jnt_qposadr[model.joint("right_dex1_finger_joint_1").id],
        model.jnt_qposadr[model.joint("right_dex1_finger_joint_2").id],
    ])
    hand_site = model.site("right_grasp_site").id
    cube_body = model.body("red_cube").id

    pair_adrs = np.array([shoulder_adr, elbow_adr])
    pair_ids = np.array([
        model.joint("right_shoulder_pitch_joint").id,
        elbow_id,
    ])
    pair_lo = model.jnt_range[pair_ids, 0] + 1e-5
    pair_hi = model.jnt_range[pair_ids, 1] - 1e-5

    # 2. Open the gripper and solve a collision-clear horizontal retreat.
    # Retreat horizontally along the gripper's current approach direction.
    open_checkpoint = checkpoint.copy()
    open_checkpoint[finger_adrs] = 0.0245
    planning_data.qpos[:] = open_checkpoint
    mujoco.mj_forward(model, planning_data)
    original_site = planning_data.site_xpos[hand_site].copy()
    wrist_rotation = planning_data.body("right_wrist_yaw_link").xmat.reshape(3, 3)
    horizontal_approach = wrist_rotation[:, 0].copy()
    horizontal_approach[2] = 0.0
    horizontal_approach /= np.linalg.norm(horizontal_approach)
    retreat_target = original_site - args.retract * horizontal_approach

    def retreat_residual(q):
        """Helper: return hand-position error for the retreat IK candidate ``q``."""
        planning_data.qpos[:] = open_checkpoint
        planning_data.qpos[pair_adrs] = q
        mujoco.mj_forward(model, planning_data)
        return planning_data.site_xpos[hand_site] - retreat_target

    retreated_pair = least_squares(
        retreat_residual,
        open_checkpoint[pair_adrs],
        bounds=(pair_lo, pair_hi),
        max_nfev=3000,
    ).x
    open_checkpoint[pair_adrs] = retreated_pair
    planning_data.qpos[:] = open_checkpoint
    mujoco.mj_forward(model, planning_data)
    old_site = planning_data.site_xpos[hand_site].copy()
    raised_target = old_site + [0.0, 0.0, 0.025]

    def raise_residual(q):
        """Helper: return hand-position error for the raised pregrasp candidate."""
        planning_data.qpos[:] = open_checkpoint
        planning_data.qpos[pair_adrs] = q
        mujoco.mj_forward(model, planning_data)
        return planning_data.site_xpos[hand_site] - raised_target

    raised_pair = least_squares(
        raise_residual,
        open_checkpoint[pair_adrs],
        bounds=(pair_lo, pair_hi),
        max_nfev=3000,
    ).x
    start = open_checkpoint.copy()
    start[pair_adrs] = raised_pair

    # 3. Solve an elbow-only lift that clears the tray after physical grasping.
    # Elbow-only lift target: cube bottom clears the tray wall by the requested
    # margin. Horizontal drift is intentionally allowed in this probe.
    grasp = open_checkpoint.copy()
    planning_data.qpos[:] = grasp
    mujoco.mj_forward(model, planning_data)
    grasp_site = planning_data.site_xpos[hand_site].copy()

    tray_geom = model.geom("tray_wall_front").id
    cube_geom = model.geom("red_cube_geom").id
    tray_top = planning_data.geom_xpos[tray_geom, 2] + model.geom_size[tray_geom, 2]
    cube_half_height = model.geom_size[cube_geom, 2]
    initial_cube_height = planning_data.xpos[cube_body, 2]
    desired_cube_height = tray_top + args.tray_clearance + cube_half_height
    desired_lift = desired_cube_height - initial_cube_height

    def lift_height_error(elbow_q):
        """Helper: return squared lift-height error for one elbow candidate."""
        planning_data.qpos[:] = grasp
        planning_data.qpos[elbow_adr] = elbow_q
        mujoco.mj_forward(model, planning_data)
        return (planning_data.site_xpos[hand_site, 2] - (grasp_site[2] + desired_lift)) ** 2

    lift_elbow = minimize_scalar(
        lift_height_error,
        bounds=tuple(model.jnt_range[elbow_id]),
        method="bounded",
    ).x
    lifted = grasp.copy()
    lifted[elbow_adr] = lift_elbow
    lifted[finger_adrs] = CLOSE_TARGET

    waist_id = model.joint("waist_yaw_joint").id
    waist_adr = model.jnt_qposadr[waist_id]
    tray_site = model.site("tray_target_site").id

    def tray_distance(waist_yaw):
        """Helper: return hand-to-tray distance for a candidate waist yaw."""
        planning_data.qpos[:] = lifted
        planning_data.qpos[waist_adr] = waist_yaw
        mujoco.mj_forward(model, planning_data)
        return np.linalg.norm(
            planning_data.site_xpos[hand_site] - planning_data.site_xpos[tray_site]
        )

    transfer_yaw = minimize_scalar(
        tray_distance,
        bounds=tuple(model.jnt_range[waist_id]),
        method="bounded",
    ).x
    transferred = lifted.copy()
    transferred[waist_adr] = transfer_yaw

    # 4. Optionally refine tray alignment with waist pitch and shoulder roll.
    # After the waist-only closest point, allow only waist yaw, waist pitch and
    # shoulder roll. Shoulder roll is the G1 DOF that removes the observed
    # inward arm bias; elbow, wrist and fingers remain fixed.
    waist_pitch_id = model.joint("waist_pitch_joint").id
    waist_pitch_adr = model.jnt_qposadr[waist_pitch_id]
    shoulder_roll_id = model.joint("right_shoulder_roll_joint").id
    shoulder_roll_adr = model.jnt_qposadr[shoulder_roll_id]
    align_ids = np.array([waist_id, waist_pitch_id, shoulder_roll_id])
    align_adrs = np.array([waist_adr, waist_pitch_adr, shoulder_roll_adr])
    align_lo = model.jnt_range[align_ids, 0] + 1e-5
    align_hi = model.jnt_range[align_ids, 1] - 1e-5
    planning_data.qpos[:] = transferred
    mujoco.mj_forward(model, planning_data)
    align_start = transferred[align_adrs].copy()
    held_height = planning_data.site_xpos[hand_site, 2]

    def align_residual(q):
        """Helper: return tray-alignment, height, and regularization errors."""
        planning_data.qpos[:] = transferred
        planning_data.qpos[align_adrs] = q
        mujoco.mj_forward(model, planning_data)
        hand = planning_data.site_xpos[hand_site]
        tray = planning_data.site_xpos[tray_site]
        return np.r_[
            25.0 * (hand[:2] - tray[:2]),
            1.0 * (hand[2] - held_height),
            0.20 * (q - align_start),
        ]

    aligned_values = least_squares(
        align_residual,
        align_start,
        bounds=(align_lo, align_hi),
        max_nfev=5000,
    ).x
    aligned = transferred.copy()
    aligned[align_adrs] = aligned_values

    # 5. Execute open, descend, close, lift, and optional transfer in physics.
    data = mujoco.MjData(model)
    data.qpos[:] = start
    data.qvel[:] = 0.0
    data.ctrl[:] = actuator_targets(model, start)
    mujoco.mj_forward(model, data)
    initial_cube_z = float(data.xpos[cube_body, 2])
    cube_qadr = model.jnt_qposadr[model.joint("red_cube_freejoint").id]
    initial_cube_quat = data.qpos[cube_qadr + 3:cube_qadr + 7].copy()
    timestep = model.opt.timestep
    duration = 12.5 if args.align_tray else (8.0 if args.transfer_waist_only else 6.0)
    history = []
    bilateral_ever = False
    base_contact_ever = False
    grip_relative = None

    while data.time < duration:
        t = data.time
        if t < 0.5:
            qref = start
        elif t < 2.0:
            qref = interpolate(start, open_checkpoint, (t - 0.5) / 1.5)
        elif t < 3.0:
            qref = open_checkpoint.copy()
            qref[finger_adrs] = interpolate(
                np.full(2, 0.0245), np.full(2, CLOSE_TARGET), (t - 2.0) / 1.0
            )
        elif t < 3.5:
            qref = grasp.copy()
            qref[finger_adrs] = CLOSE_TARGET
        elif t < 5.5:
            closed_grasp = grasp.copy()
            closed_grasp[finger_adrs] = CLOSE_TARGET
            qref = interpolate(closed_grasp, lifted, (t - 3.5) / 2.0)
        elif not args.transfer_waist_only or t < 6.0:
            qref = lifted
        elif t < 7.5:
            qref = interpolate(lifted, transferred, (t - 6.0) / 1.5)
        elif not args.align_tray or t < 8.0:
            qref = transferred
        elif t < 12.0:
            qref = interpolate(transferred, aligned, (t - 8.0) / 4.0)
        else:
            qref = aligned

        data.ctrl[:] = actuator_targets(model, qref)
        mujoco.mj_step(model, data)
        finger_count, base_hit = finger_cube_contacts(model, data)
        bilateral_ever |= finger_count == 2
        base_contact_ever |= base_hit
        if t >= 3.45 and grip_relative is None:
            hand_rotation = data.site_xmat[hand_site].reshape(3, 3)
            grip_relative = hand_rotation.T @ (
                data.xpos[cube_body].copy() - data.site_xpos[hand_site].copy()
            )
        if len(history) == 0 or t - history[-1][0] >= 1 / 60:
            history.append((t, data.qpos.copy()))

    # 6. Evaluate contact quality, lift height, slip, rotation, and success.
    final_cube_z = float(data.xpos[cube_body, 2])
    final_hand_rotation = data.site_xmat[hand_site].reshape(3, 3)
    final_relative = final_hand_rotation.T @ (
        data.xpos[cube_body].copy() - data.site_xpos[hand_site].copy()
    )
    relative_slip = np.linalg.norm(final_relative - grip_relative)
    lift = final_cube_z - initial_cube_z
    final_cube_quat = data.qpos[cube_qadr + 3:cube_qadr + 7].copy()
    quat_alignment = np.clip(abs(np.dot(initial_cube_quat, final_cube_quat)), 0.0, 1.0)
    cube_rotation_deg = np.degrees(2.0 * np.arccos(quat_alignment))
    w, x, y, z = final_cube_quat
    cube_yaw_deg = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    success = bilateral_ever and not base_contact_ever and lift > 0.02 and relative_slip < 0.02
    print("User-proposed slanted grasp + elbow-only lift probe")
    print(f"requested_retract_cm={100*args.retract:.3f}")
    print(f"retreat_IK_error_mm={1000*np.linalg.norm(retreat_residual(retreated_pair)):.3f}")
    print(f"start_raise_IK_error_mm={1000*np.linalg.norm(raise_residual(raised_pair)):.3f}")
    print(f"bilateral_finger_contact={bilateral_ever}")
    print(f"gripper_base_contact={base_contact_ever}")
    print(f"cube_lift_cm={100*lift:.3f}")
    print(f"desired_cube_center_z={desired_cube_height:.3f}")
    print(f"final_cube_center_z={final_cube_z:.3f}")
    print(f"cube_gripper_relative_slip_cm={100*relative_slip:.3f}")
    print(f"cube_total_rotation_deg={cube_rotation_deg:.3f}")
    print(f"cube_final_yaw_deg={cube_yaw_deg:.3f}")
    if args.transfer_waist_only:
        hand_to_tray = data.site_xpos[hand_site] - data.site_xpos[tray_site]
        print(f"transfer_waist_yaw_deg={np.degrees(transfer_yaw):.3f}")
        print(f"final_hand_tray_horizontal_cm={100*np.linalg.norm(hand_to_tray[:2]):.3f}")
        print(f"final_hand_tray_3d_cm={100*np.linalg.norm(hand_to_tray):.3f}")
    if args.align_tray:
        print(f"aligned_waist_yaw_deg={np.degrees(aligned_values[0]):.3f}")
        print(f"aligned_waist_pitch_deg={np.degrees(aligned_values[1]):.3f}")
        print(f"aligned_shoulder_roll_deg={np.degrees(aligned_values[2]):.3f}")
        print(f"aligned_horizontal_error_cm={100*np.linalg.norm(hand_to_tray[:2]):.3f}")
    print(f"success={success}")

    if not args.viewer:
        return

    # 7. Convert sampled physics states into an interactive replay.
    frames = np.asarray([qpos for _, qpos in history])
    replay = mujoco.MjData(model)
    state = {"index": 0, "playing": True}

    def key_callback(keycode):
        """Handle run, pause, reset, replay, rewind, and advance controls."""
        if keycode in (32, 257, 335):
            if state["index"] >= len(frames) - 1:
                state["index"] = 0
            state["playing"] = not state["playing"]
        elif keycode in (259, 261):
            state["index"] = 0
            state["playing"] = False
            print("[reset] press Enter/Space to run")
        elif keycode in (48, 82):
            state["index"] = 0
            state["playing"] = True
        elif keycode == 263:
            state["playing"] = False
            state["index"] = max(0, state["index"] - 30)
        elif keycode == 262:
            state["playing"] = False
            state["index"] = min(len(frames) - 1, state["index"] + 30)

    print("Replay: Enter/Space=Run/Pause | Delete=Reset | R=Reset+Run")
    with viewer.launch_passive(model, replay, key_callback=key_callback) as handle:
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        handle.cam.lookat[:] = [0.47, 0.08, 0.80]
        handle.cam.distance = 1.25
        handle.cam.azimuth = 145.0
        handle.cam.elevation = -20.0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        last_index = -1
        while handle.is_running():
            if state["index"] != last_index:
                replay.qpos[:] = frames[state["index"]]
                replay.qvel[:] = 0.0
                replay.time = state["index"] / 60
                mujoco.mj_forward(model, replay)
                last_index = state["index"]
            handle.sync()
            if state["playing"]:
                if state["index"] < len(frames) - 1:
                    state["index"] += 1
                else:
                    state["playing"] = False
                    print("[end] paused")
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
