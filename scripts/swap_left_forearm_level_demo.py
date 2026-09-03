#!/usr/bin/env python3
"""Run the mirrored cube/tray variant with left-arm clearance control.

The task geometry is a strict left-right mirror of the baseline.  Pickup and
transfer references are recomputed for the new targets, while the left elbow
is adjusted concurrently with torso pitch to keep the left forearm nearly
horizontal and clear of the table.  The left arm is not given a separate
motion phase and remains in its compensated pose through placement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

import full_pick_to_tray_center as controller
from level1_env import reset_episode
from smoke_test import minimum_jerk

# Mirrored-scene defaults and bounded torso/left-arm support parameters.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/task1/swap_cube_tray.xml"
LEFT_SHOULDER_PITCH_DEG = 0.0
LEFT_SHOULDER_ROLL_DEG = 20.0
LEFT_ELBOW_DEG = 100.0
WAIST_YAW_DEG = 0.0
WAIST_PITCH = 0.40
TRANSFER_XY_BIAS = np.array([0.0, 0.0])


def unit(vector):
    """Helper: return the normalized form of ``vector``."""
    return vector / max(np.linalg.norm(vector), 1e-12)


def level_left_forearm(model, pose):
    """Adjust and return ``pose`` with a nearly horizontal left forearm."""
    planning = mujoco.MjData(model)
    elbow_id = model.joint("left_elbow_joint").id
    wrist_id = model.joint("left_wrist_roll_joint").id
    elbow_adr = model.jnt_qposadr[elbow_id]
    low, high = model.jnt_range[elbow_id]

    def vertical_error(elbow_value):
        """Helper: return left-forearm vertical error for one elbow value."""
        planning.qpos[:] = pose
        planning.qpos[elbow_adr] = elbow_value
        mujoco.mj_forward(model, planning)
        vertical = planning.xanchor[wrist_id, 2] - planning.xanchor[elbow_id, 2]
        return float(vertical * vertical + 1e-8 * elbow_value * elbow_value)

    pose[elbow_adr] = minimize_scalar(
        vertical_error,
        bounds=(low + 1e-5, high - 1e-5),
        method="bounded",
        options={"xatol": 1e-10},
    ).x
    return pose


def tune_left_support_servos(model):
    """Increase left-arm gains for accurate clearance compensation."""
    for name in (
        "servo_left_shoulder_pitch_joint",
        "servo_left_shoulder_roll_joint",
        "servo_left_shoulder_yaw_joint",
        "servo_left_elbow_joint",
    ):
        actuator = model.actuator(name).id
        model.actuator_gainprm[actuator, 0] = 1000.0
        model.actuator_biasprm[actuator, 1] = -1000.0
        model.actuator_biasprm[actuator, 2] = -50.0


def solve_swap_pick(model, data):
    """Plan and return the mirrored pickup poses, path, and metrics."""
    # 1. Reset and resolve the right-arm joints and mirrored cube target.
    reset_episode(model, data)
    home = data.qpos.copy()
    hand = model.site("right_grasp_site").id
    cube = model.site("red_cube_grasp_site").id
    elbow_body = model.body("right_elbow_link").id
    wrist_body = model.body("right_wrist_roll_link").id
    names = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    ids = np.array([model.joint(name).id for name in names])
    adrs = model.jnt_qposadr[ids]
    lo = model.jnt_range[ids, 0] + 1e-5
    hi = model.jnt_range[ids, 1] - 1e-5
    yaw_adr = model.jnt_qposadr[model.joint("waist_yaw_joint").id]
    yaw = np.radians(WAIST_YAW_DEG)
    pitch_adr = model.jnt_qposadr[model.joint("waist_pitch_joint").id]

    def terminal_residual(q):
        """Helper: return mirrored grasp-position and orientation IK errors."""
        data.qpos[:] = home
        data.qpos[yaw_adr] = yaw
        data.qpos[pitch_adr] = WAIST_PITCH
        data.qpos[adrs] = q
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[hand].reshape(3, 3)
        return np.r_[
            100.0 * (data.site_xpos[hand] - data.site_xpos[cube]),
            10.0 * rotation[2, 1],  # keep the finger closing axis horizontal
            2.0 * (rotation[2, 2] - 0.95),
            0.0001 * q,
        ]

    terminal = least_squares(
        terminal_residual,
        np.radians([-42.0, -4.0, -2.0, 7.0, 2.0, 42.0, 1.0]),
        bounds=(lo, hi),
        max_nfev=5000,
    ).x
    elbow_id = model.joint("right_elbow_joint").id
    elbow_adr = model.jnt_qposadr[elbow_id]

    def aim_error(elbow_q):
        """Helper: return forearm-to-cube angular error for one elbow value."""
        data.qpos[:] = home
        data.qpos[yaw_adr] = yaw
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
    aimed[yaw_adr] = yaw
    aimed[elbow_adr] = elbow_aim
    data.qpos[:] = aimed
    mujoco.mj_forward(model, data)
    start_hand = data.site_xpos[hand].copy()
    target = data.site_xpos[cube].copy()
    arm_adrs = adrs
    arm_lo = lo
    arm_hi = hi
    previous = aimed[arm_adrs].copy()
    reach_path = []
    path_errors = []

    # 2. Precompute a minimum-jerk Cartesian approach for the mirrored target.
    for alpha in np.linspace(0.0, 1.0, 91)[1:]:
        progress = minimum_jerk(alpha)
        desired = (1.0 - progress) * start_hand + progress * target

        def frame_residual(q):
            """Helper: return tracking, orientation, and continuity errors per frame."""
            data.qpos[:] = aimed
            data.qpos[pitch_adr] = progress * WAIST_PITCH
            data.qpos[arm_adrs] = q
            mujoco.mj_forward(model, data)
            rotation = data.site_xmat[hand].reshape(3, 3)
            interpolated_terminal = (
                (1.0 - progress) * aimed[arm_adrs] + progress * terminal
            )
            return np.r_[
                100.0 * (data.site_xpos[hand] - desired),
                2.0 * progress * rotation[2, 1],
                0.01 * (q - interpolated_terminal),
                0.002 * (q - previous),
            ]

        previous = least_squares(
            frame_residual,
            previous,
            bounds=(arm_lo, arm_hi),
            max_nfev=1000,
        ).x
        qpos = aimed.copy()
        qpos[pitch_adr] = progress * WAIST_PITCH
        qpos[arm_adrs] = previous
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        path_errors.append(np.linalg.norm(data.site_xpos[hand] - desired))
        reach_path.append(qpos)

    final = home.copy()
    final[yaw_adr] = yaw
    final[pitch_adr] = WAIST_PITCH
    final[adrs] = terminal
    reach_path[-1] = final
    data.qpos[:] = final
    mujoco.mj_forward(model, data)
    final_error = np.linalg.norm(data.site_xpos[hand] - data.site_xpos[cube])
    metrics = {
        "waist_yaw_deg": np.degrees(yaw),
        "waist_pitch_deg": np.degrees(WAIST_PITCH),
        "aim_elbow_deg": np.degrees(elbow_aim),
        "final_shoulder_pitch_deg": np.degrees(terminal[0]),
        "final_shoulder_roll_deg": np.degrees(terminal[1]),
        "final_shoulder_yaw_deg": np.degrees(terminal[2]),
        "final_elbow_deg": np.degrees(terminal[3]),
        "max_precomputed_path_error_cm": 100 * max(path_errors),
        "final_touch_error_mm": 1000 * final_error,
    }
    return home, aimed, np.asarray(reach_path), metrics


def cartesian_approach_path(model, start, target, frame_count=91):
    """Return a smooth mirrored approach that avoids sweeping through the cube."""
    planning = mujoco.MjData(model)
    hand = model.site("right_grasp_site").id
    names = (
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    ids = np.array([model.joint(name).id for name in names])
    adrs = model.jnt_qposadr[ids]
    lo = model.jnt_range[ids, 0] + 1e-5
    hi = model.jnt_range[ids, 1] - 1e-5
    planning.qpos[:] = start
    mujoco.mj_forward(model, planning)
    start_position = planning.site_xpos[hand].copy()
    planning.qpos[:] = target
    mujoco.mj_forward(model, planning)
    target_position = planning.site_xpos[hand].copy()
    previous = start[adrs].copy()
    frames = []
    for alpha in np.linspace(0.0, 1.0, frame_count):
        progress = minimum_jerk(alpha)
        desired_position = (
            (1.0 - progress) * start_position + progress * target_position
        )
        desired_joints = (1.0 - progress) * start[adrs] + progress * target[adrs]

        def residual(q):
            """Helper: return Cartesian and joint-continuity errors for one frame."""
            planning.qpos[:] = start
            planning.qpos[adrs] = q
            mujoco.mj_forward(model, planning)
            return np.r_[
                100.0 * (planning.site_xpos[hand] - desired_position),
                0.01 * (q - desired_joints),
                0.002 * (q - previous),
            ]

        previous = least_squares(
            residual, previous, bounds=(lo, hi), max_nfev=1000
        ).x
        frame = start.copy()
        frame[adrs] = previous
        frames.append(frame)
    frames[-1] = target.copy()
    return np.asarray(frames)


def elevated_pregrasp(model, open_checkpoint, clearance=0.12):
    """Return a raised pregrasp that provides clearance for tracking sag."""
    planning = mujoco.MjData(model)
    hand = model.site("right_grasp_site").id
    names = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    ids = np.array([model.joint(name).id for name in names])
    adrs = model.jnt_qposadr[ids]
    lo = model.jnt_range[ids, 0] + 1e-5
    hi = model.jnt_range[ids, 1] - 1e-5
    planning.qpos[:] = open_checkpoint
    mujoco.mj_forward(model, planning)
    start_rotation = planning.site_xmat[hand].reshape(3, 3).copy()
    target_position = planning.site_xpos[hand].copy() + [0.0, 0.0, clearance]
    start_values = open_checkpoint[adrs].copy()

    def residual(q):
        """Helper: return pregrasp position, orientation, and motion errors."""
        planning.qpos[:] = open_checkpoint
        planning.qpos[adrs] = q
        mujoco.mj_forward(model, planning)
        rotation = planning.site_xmat[hand].reshape(3, 3)
        return np.r_[
            100.0 * (planning.site_xpos[hand] - target_position),
            0.5 * (rotation - start_rotation).reshape(-1),
            0.001 * (q - start_values),
        ]

    solution = least_squares(
        residual, start_values, bounds=(lo, hi), max_nfev=5000
    ).x
    raised = open_checkpoint.copy()
    raised[adrs] = solution
    return raised


def main():
    """CLI entry point that applies mirrored overrides to the shared controller."""
    # 1. Load the mirrored model and tune both the working and support arms.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--transport-roll-deg", type=float, default=-90.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    controller.tune_right_arm_servos(model)
    tune_left_support_servos(model)
    original_builder = controller.build_pick_references
    original_solver = controller.solve
    original_transition = controller.run_transition
    original_plan_transfer = controller.plan_transfer
    original_append_step = controller.append_step
    controller.solve = solve_swap_pick
    approach_path = None
    raised_reference = None

    def left_arm_down_references(active_model):
        """Build pickup references while synchronizing left-forearm leveling."""
        nonlocal approach_path, raised_reference
        refs = original_builder(active_model)
        for value in refs.values():
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 1
                and value.shape[0] == active_model.nq
            ):
                level_left_forearm(active_model, value)
        refs["raised"] = elevated_pregrasp(active_model, refs["open_checkpoint"])
        level_left_forearm(active_model, refs["raised"])
        raised_reference = refs["raised"].copy()
        approach_path = cartesian_approach_path(
            active_model, refs["aimed_open"], refs["raised"]
        )
        for frame in approach_path:
            level_left_forearm(active_model, frame)
        return refs

    def path_aware_transition(active_model, data, start, target, seconds, history, last_sample):
        """Use the collision-aware approach path and adjusted transfer timing."""
        if (
            approach_path is not None
            and raised_reference is not None
            and abs(seconds - 1.15) < 1e-9
            and np.allclose(target, raised_reference)
        ):
            begin = data.time
            motion_seconds = 1.7
            end = begin + motion_seconds
            while data.time < end:
                alpha = np.clip((data.time - begin) / motion_seconds, 0.0, 1.0)
                index = min(
                    len(approach_path) - 1,
                    int(round(alpha * (len(approach_path) - 1))),
                )
                controller.append_step(
                    active_model, data, approach_path[index], history, last_sample
                )
            return target.copy()
        if abs(seconds - 2.8) < 1e-9:
            # Faster than the conservative version, while retaining enough
            # settling time for the bounded-torque servos.
            return original_transition(
                active_model, data, start, target, 3.8, history, last_sample
            )
        return original_transition(
            active_model, data, start, target, seconds, history, last_sample
        )

    controller.build_pick_references = left_arm_down_references
    controller.run_transition = path_aware_transition

    left_elbow_id = model.joint("left_elbow_joint").id
    left_wrist_id = model.joint("left_wrist_roll_joint").id
    left_elbow_adr = model.jnt_qposadr[left_elbow_id]
    left_elbow_low, left_elbow_high = model.jnt_range[left_elbow_id]

    def level_feedback_step(active_model, data, qref, history, last_sample):
        """Correct left-elbow command online to counter physical servo lag."""
        # Compensate physical servo lag during the simultaneous waist bend.
        forearm = data.xanchor[left_wrist_id] - data.xanchor[left_elbow_id]
        signed_tilt = np.arctan2(forearm[2], np.linalg.norm(forearm[:2]))
        corrected = qref.copy()
        corrected[left_elbow_adr] = np.clip(
            qref[left_elbow_adr] + 1.2 * signed_tilt,
            left_elbow_low + 1e-5,
            left_elbow_high - 1e-5,
        )
        original_append_step(active_model, data, corrected, history, last_sample)

    controller.append_step = level_feedback_step

    def bounded_plan_transfer(active_model, data, current_ref, desired_z=0.960):
        """Clip hinge state before calling shared transfer IK and restore targets."""
        # Stronger waist/arm motion can leave the simulated hinge a tiny amount
        # beyond its declared range; clip that numerical overshoot before IK.
        for joint_id in range(active_model.njnt):
            if active_model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            address = active_model.jnt_qposadr[joint_id]
            low, high = active_model.jnt_range[joint_id]
            data.qpos[address] = np.clip(data.qpos[address], low + 1e-5, high - 1e-5)
            current_ref[address] = np.clip(
                current_ref[address], low + 1e-5, high - 1e-5
            )
        mujoco.mj_forward(active_model, data)
        tray_site = active_model.site("tray_target_site").id
        original_site_xy = active_model.site_pos[tray_site, :2].copy()
        active_model.site_pos[tray_site, :2] = original_site_xy + TRANSFER_XY_BIAS
        try:
            return original_plan_transfer(active_model, data, current_ref, desired_z)
        finally:
            active_model.site_pos[tray_site, :2] = original_site_xy
            mujoco.mj_forward(active_model, data)

    controller.plan_transfer = bounded_plan_transfer
    # 2. Run the shared physical policy with all mirrored overrides installed.
    try:
        policy_frames, metrics, _ = controller.simulate(
            model,
            transport_roll_deg=args.transport_roll_deg,
            place_descent=True,
            unroll_before_descent=False,
            coordinated_place=True,
        )
    finally:
        controller.build_pick_references = original_builder
        controller.solve = original_solver
        controller.run_transition = original_transition
        controller.plan_transfer = original_plan_transfer
        controller.append_step = original_append_step

    # 3. Replay all frames to measure left-arm clearance and placement quality.
    # No dedicated left-arm phase: compensation occurs inside the same frames
    # as the waist bend/right-arm approach, then remains fixed afterwards.
    frames = policy_frames

    data = mujoco.MjData(model)
    hand_site = model.site("right_grasp_site").id
    cube_body = model.body("red_cube").id
    tray_body = model.body("tray").id
    left_elbow_id = model.joint("left_elbow_joint").id
    left_wrist_id = model.joint("left_wrist_roll_joint").id
    left_elbow_adr = model.jnt_qposadr[left_elbow_id]
    min_hand_cube = np.inf
    left_tray_contacts = 0
    left_environment_contacts = 0
    max_left_forearm_tilt_deg = 0.0
    left_elbow_values = []
    for frame in policy_frames:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        min_hand_cube = min(
            min_hand_cube,
            float(np.linalg.norm(data.site_xpos[hand_site] - data.xpos[cube_body])),
        )
        forearm = data.xanchor[left_wrist_id] - data.xanchor[left_elbow_id]
        tilt = np.degrees(np.arctan2(abs(forearm[2]), np.linalg.norm(forearm[:2])))
        max_left_forearm_tilt_deg = max(max_left_forearm_tilt_deg, float(tilt))
        left_elbow_values.append(float(frame[left_elbow_adr]))
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = model.body(model.geom_bodyid[contact.geom1]).name or "world"
            body2 = model.body(model.geom_bodyid[contact.geom2]).name or "world"
            if (
                (model.geom_bodyid[contact.geom1] == tray_body and body2.startswith("left_"))
                or (model.geom_bodyid[contact.geom2] == tray_body and body1.startswith("left_"))
            ):
                left_tray_contacts += 1
            geom1 = model.geom(contact.geom1).name or "unnamed"
            geom2 = model.geom(contact.geom2).name or "unnamed"
            if (
                body1.startswith("left_")
                and (geom2.startswith("table_") or model.geom_bodyid[contact.geom2] == tray_body)
            ) or (
                body2.startswith("left_")
                and (geom1.startswith("table_") or model.geom_bodyid[contact.geom1] == tray_body)
            ):
                left_environment_contacts += 1

    print("Swap variant with synchronized level-left-forearm compensation")
    print(f"  left_arm_tray_contacts_after_preparation: {left_tray_contacts}")
    print(f"  left_arm_table_or_tray_contacts: {left_environment_contacts}")
    print(f"  left_elbow_range_deg: {np.degrees(min(left_elbow_values)):.2f}..{np.degrees(max(left_elbow_values)):.2f}")
    print(f"  max_left_forearm_tilt_deg: {max_left_forearm_tilt_deg:.3f}")
    print(f"  minimum_right_hand_cube_distance_cm: {100*min_hand_cube:.3f}")
    print(f"  lift_bilateral_contacts: {metrics['lift_bilateral_contacts']}")
    print(f"  transport_bilateral_contacts: {metrics['transport_bilateral_contacts']}")
    print(f"  transport_lift_to_center_slip_cm: {metrics['transport_lift_to_center_slip_cm']:.3f}")
    print(f"  roll_stage_slip_cm: {metrics['roll_stage_slip_cm']:.3f}")
    print(f"  horizontal_error_cm: {metrics['horizontal_error_cm']:.3f}")
    print(f"  planned_cube_center: {metrics['planned_cube_center']}")
    print(f"  transport_center_before_lowering: {metrics['transport_center_before_lowering']}")
    print(f"  final_cube_center: {metrics['final_cube_center']}")
    print(f"  tray_center: {metrics['tray_center']}")
    print(f"  descent_release_trigger: {metrics['descent_release_trigger']}")
    print(f"  post_release_tray_contact: {metrics['post_release_tray_contact']}")
    print(f"  success: {metrics['success']}")
    print("Replay: Space=Run/Pause | Left/Right=Rewind/Advance | R=Replay")
    if args.viewer:
        controller.replay(model, frames)


if __name__ == "__main__":
    main()
