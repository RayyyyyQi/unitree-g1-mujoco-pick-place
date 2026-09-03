#!/usr/bin/env python3
"""Execute the complete physics-based Level-1 pick-and-place policy.

The controller combines staged numerical IK with minimum-jerk position-servo
tracking.  It aligns to the cube, opens and closes the Dex1 gripper, lifts,
uses wrist roll to support the payload, transports the measured cube center
above the tray, descends under feedback, and releases only after contact or a
settled near-contact condition.  Planning and execution use MuJoCo state rather
than teleporting either the robot or the object.
"""

from __future__ import annotations

import argparse
import time

import mujoco
from mujoco import viewer
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from grasp_lift_probe import CLOSE_TARGET, actuator_targets, finger_cube_contacts
from level1_env import DEFAULT_MODEL
from smoke_test import minimum_jerk
from sync_planar_scrubber import solve

# Replay rate and Dex1 fully-open command used throughout the controller.
REPLAY_FPS = 60
OPEN_TARGET = 0.0245


def tune_right_arm_servos(model):
    """Increase right-arm position gains to reduce gravity sag during transport."""
    for name in (
        "servo_right_shoulder_pitch_joint",
        "servo_right_shoulder_roll_joint",
        "servo_right_shoulder_yaw_joint",
        "servo_right_elbow_joint",
    ):
        actuator = model.actuator(name).id
        model.actuator_gainprm[actuator, 0] = 1000.0
        model.actuator_biasprm[actuator, 1] = -1000.0
        model.actuator_biasprm[actuator, 2] = -50.0


def blend(a, b, alpha):
    """Helper: return a minimum-jerk interpolation from ``a`` to ``b``."""
    s = minimum_jerk(np.clip(alpha, 0.0, 1.0))
    return (1.0 - s) * a + s * b


def joint_address(model, name):
    """Helper: return the qpos address of scalar joint ``name``."""
    return model.jnt_qposadr[model.joint(name).id]


def append_step(model, data, qref, history, last_sample):
    """Helper: apply ``qref``, step physics, and sample replay history."""
    data.ctrl[:] = actuator_targets(model, qref)
    mujoco.mj_step(model, data)
    if data.time - last_sample[0] >= 1.0 / REPLAY_FPS:
        history.append(data.qpos.copy())
        last_sample[0] = data.time


def run_hold(model, data, qref, seconds, history, last_sample):
    """Helper: hold ``qref`` for the requested simulated duration."""
    end = data.time + seconds
    while data.time < end:
        append_step(model, data, qref, history, last_sample)


def run_transition(model, data, start, target, seconds, history, last_sample):
    """Helper: execute and return one smooth joint-space transition target."""
    begin = data.time
    end = begin + seconds
    while data.time < end:
        qref = blend(start, target, (data.time - begin) / seconds)
        append_step(model, data, qref, history, last_sample)
    return target.copy()


def build_pick_references(model):
    """Return the planned home, approach, grasp, and lift reference poses."""
    # 1. Reuse the planar planner for the home, aim, and terminal reach poses.
    planning = mujoco.MjData(model)
    home, aimed, reach_path, _ = solve(model, planning)
    checkpoint = reach_path[-1].copy()

    finger_adrs = np.array([
        joint_address(model, "right_dex1_finger_joint_1"),
        joint_address(model, "right_dex1_finger_joint_2"),
    ])
    shoulder_adr = joint_address(model, "right_shoulder_pitch_joint")
    elbow_id = model.joint("right_elbow_joint").id
    elbow_adr = model.jnt_qposadr[elbow_id]
    pair_adrs = np.array([shoulder_adr, elbow_adr])
    pair_ids = np.array([model.joint("right_shoulder_pitch_joint").id, elbow_id])
    pair_lo = model.jnt_range[pair_ids, 0] + 1e-5
    pair_hi = model.jnt_range[pair_ids, 1] - 1e-5
    hand_site = model.site("right_grasp_site").id

    # 2. Open the fingers and retract horizontally before physical descent.
    open_checkpoint = checkpoint.copy()
    open_checkpoint[finger_adrs] = OPEN_TARGET
    planning.qpos[:] = open_checkpoint
    mujoco.mj_forward(model, planning)
    original_site = planning.site_xpos[hand_site].copy()
    approach = planning.body("right_wrist_yaw_link").xmat.reshape(3, 3)[:, 0].copy()
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    retreat_target = original_site - 0.02 * approach

    def retreat_residual(q):
        """Helper: return hand-position error for a retreat IK candidate."""
        planning.qpos[:] = open_checkpoint
        planning.qpos[pair_adrs] = q
        mujoco.mj_forward(model, planning)
        return planning.site_xpos[hand_site] - retreat_target

    retreated = least_squares(
        retreat_residual,
        open_checkpoint[pair_adrs],
        bounds=(pair_lo, pair_hi),
        max_nfev=3000,
    ).x
    open_checkpoint[pair_adrs] = retreated

    # 3. Raise the open gripper to a collision-clear pregrasp waypoint.
    planning.qpos[:] = open_checkpoint
    mujoco.mj_forward(model, planning)
    raised_target = planning.site_xpos[hand_site].copy() + [0.0, 0.0, 0.060]

    def raised_residual(q):
        """Helper: return hand-position error for a raised-pregrasp candidate."""
        planning.qpos[:] = open_checkpoint
        planning.qpos[pair_adrs] = q
        mujoco.mj_forward(model, planning)
        return planning.site_xpos[hand_site] - raised_target

    raised_pair = least_squares(
        raised_residual,
        open_checkpoint[pair_adrs],
        bounds=(pair_lo, pair_hi),
        max_nfev=3000,
    ).x
    raised = open_checkpoint.copy()
    raised[pair_adrs] = raised_pair

    aimed_open = aimed.copy()
    aimed_open[finger_adrs] = OPEN_TARGET
    closed = open_checkpoint.copy()
    closed[finger_adrs] = CLOSE_TARGET

    # 4. Solve a pure-elbow lift with clearance for the later transfer.
    planning.qpos[:] = closed
    mujoco.mj_forward(model, planning)
    grasp_z = planning.site_xpos[hand_site, 2]
    desired_lift = 0.159  # nominal cube center 0.771 -> about 0.930 m

    def lift_error(elbow_q):
        """Helper: return squared lift-height error for one elbow candidate."""
        planning.qpos[:] = closed
        planning.qpos[elbow_adr] = elbow_q
        mujoco.mj_forward(model, planning)
        return (planning.site_xpos[hand_site, 2] - (grasp_z + desired_lift)) ** 2

    lift_elbow = minimize_scalar(
        lift_error,
        bounds=tuple(model.jnt_range[elbow_id]),
        method="bounded",
    ).x
    lifted = closed.copy()
    lifted[elbow_adr] = lift_elbow

    return {
        "home": home,
        "aimed_open": aimed_open,
        "raised": raised,
        "open_checkpoint": open_checkpoint,
        "closed": closed,
        "lifted": lifted,
        "finger_adrs": finger_adrs,
    }


def predict_cube_center(model, planning, qpos, hand_site, relative_local):
    """Helper: predict cube center from hand pose and local grasp offset."""
    planning.qpos[:] = qpos
    mujoco.mj_forward(model, planning)
    rotation = planning.site_xmat[hand_site].reshape(3, 3)
    return planning.site_xpos[hand_site] + rotation @ relative_local


def cube_tray_bottom_contact(model, data):
    """Helper: return whether the cube directly contacts the tray bottom."""
    cube_geom = model.geom("red_cube_geom").id
    tray_geom = model.geom("tray_bottom").id
    for index in range(data.ncon):
        pair = {data.contact[index].geom1, data.contact[index].geom2}
        if pair == {cube_geom, tray_geom}:
            return True
    return False


def plan_transfer(model, data, current_ref, desired_z=0.960):
    """Return waist/final references that center the cube above the tray."""
    # 1. Measure the physical cube-to-hand transform after the lift.
    planning = mujoco.MjData(model)
    hand_site = model.site("right_grasp_site").id
    tray_site = model.site("tray_target_site").id
    cube_body = model.body("red_cube").id
    hand_rotation = data.site_xmat[hand_site].reshape(3, 3).copy()
    relative_local = hand_rotation.T @ (
        data.xpos[cube_body].copy() - data.site_xpos[hand_site].copy()
    )
    actual_start = data.qpos.copy()
    # Keep actuator targets close to the actual post-lift state to avoid a jump.
    for actuator in range(model.nu):
        joint = model.actuator_trnid[actuator, 0]
        adr = model.jnt_qposadr[joint]
        current_ref[adr] = actual_start[adr]
    current_ref[joint_address(model, "right_dex1_finger_joint_1")] = CLOSE_TARGET
    current_ref[joint_address(model, "right_dex1_finger_joint_2")] = CLOSE_TARGET

    waist_id = model.joint("waist_yaw_joint").id
    waist_adr = model.jnt_qposadr[waist_id]
    planning.qpos[:] = current_ref
    mujoco.mj_forward(model, planning)
    tray_xy = planning.site_xpos[tray_site, :2].copy()

    def waist_objective(yaw):
        """Helper: return cube-to-tray XY distance for a candidate waist yaw."""
        candidate = current_ref.copy()
        candidate[waist_adr] = yaw
        center = predict_cube_center(model, planning, candidate, hand_site, relative_local)
        return np.linalg.norm(center[:2] - tray_xy)

    waist_yaw = minimize_scalar(
        waist_objective,
        bounds=tuple(model.jnt_range[waist_id]),
        method="bounded",
    ).x
    waist_ref = current_ref.copy()
    waist_ref[waist_adr] = waist_yaw

    # 2. Use upper-body IK after waist alignment to refine the cube center.
    # Wrist DOFs are
    # included so the gripper can retain its post-lift orientation.
    names = (
        "waist_yaw_joint", "waist_pitch_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    ids = np.array([model.joint(name).id for name in names])
    adrs = model.jnt_qposadr[ids]
    lo = model.jnt_range[ids, 0] + 1e-5
    hi = model.jnt_range[ids, 1] - 1e-5
    start_values = waist_ref[adrs].copy()
    # Aim high enough that gravity/servo tracking error still leaves the cube
    # safely above the tray walls in the realized physics trajectory.
    desired_center = np.r_[tray_xy, desired_z]

    def final_residual(values):
        """Helper: return position, orientation, and motion-cost IK residuals."""
        candidate = waist_ref.copy()
        candidate[adrs] = values
        planning.qpos[:] = candidate
        mujoco.mj_forward(model, planning)
        rotation = planning.site_xmat[hand_site].reshape(3, 3)
        center = planning.site_xpos[hand_site] + rotation @ relative_local
        orientation = (rotation - hand_rotation).reshape(-1)
        normalized_change = (values - start_values) / np.maximum(hi - lo, 1e-6)
        return np.r_[
            30.0 * (center - desired_center),
            2.0 * orientation,
            0.15 * normalized_change,
        ]

    solution = least_squares(
        final_residual,
        start_values,
        bounds=(lo, hi),
        max_nfev=10000,
    ).x
    final_ref = waist_ref.copy()
    final_ref[adrs] = solution
    planned_center = predict_cube_center(
        model, planning, final_ref, hand_site, relative_local
    )
    return waist_ref, final_ref, relative_local, planned_center


def simulate(
    model,
    transport_roll_deg=0.0,
    place_descent=False,
    unroll_before_descent=False,
    coordinated_place=False,
):
    """Run the complete closed-loop policy and return metrics, frames, and checkpoints."""
    # 1. Initialize physics from the planned home reference.
    refs = build_pick_references(model)
    data = mujoco.MjData(model)
    data.qpos[:] = refs["home"]
    data.qvel[:] = 0.0
    data.ctrl[:] = actuator_targets(model, refs["home"])
    mujoco.mj_forward(model, data)
    history = [data.qpos.copy()]
    last_sample = [data.time]
    checkpoints = {}

    # 2. Aim, approach, descend, close the fingers, and lift the cube.
    current = refs["home"].copy()
    run_hold(model, data, current, 0.35, history, last_sample)
    current = run_transition(
        model, data, current, refs["aimed_open"], 0.7, history, last_sample
    )
    run_hold(model, data, current, 0.2, history, last_sample)
    current = run_transition(
        model, data, current, refs["raised"], 1.15, history, last_sample
    )
    run_hold(model, data, current, 0.25, history, last_sample)
    current = run_transition(
        model, data, current, refs["open_checkpoint"], 0.8, history, last_sample
    )
    current = run_transition(
        model, data, current, refs["closed"], 0.6, history, last_sample
    )
    run_hold(model, data, current, 0.3, history, last_sample)
    current = run_transition(
        model, data, current, refs["lifted"], 2.0, history, last_sample
    )
    run_hold(model, data, current, 0.65, history, last_sample)

    hand_site = model.site("right_grasp_site").id
    cube_body = model.body("red_cube").id
    tray_site = model.site("tray_target_site").id
    lift_fingers, lift_base = finger_cube_contacts(model, data)
    lift_cube_z = float(data.xpos[cube_body, 2])
    checkpoints["grasp_lifted"] = data.qpos.copy()
    lift_rotation = data.site_xmat[hand_site].reshape(3, 3).copy()
    lift_relative = lift_rotation.T @ (
        data.xpos[cube_body].copy() - data.site_xpos[hand_site].copy()
    )

    # 3. Roll one finger under the cube to support it during transport.
    wrist_roll_adr = joint_address(model, "right_wrist_roll_joint")
    original_wrist_roll = current[wrist_roll_adr]
    transport_ref = current.copy()
    transport_ref[wrist_roll_adr] = np.clip(
        original_wrist_roll + np.radians(transport_roll_deg),
        model.joint("right_wrist_roll_joint").range[0] + 1e-5,
        model.joint("right_wrist_roll_joint").range[1] - 1e-5,
    )
    current = run_transition(
        model, data, current, transport_ref, 1.5, history, last_sample
    )
    run_hold(model, data, current, 0.35, history, last_sample)
    rolled_rotation = data.site_xmat[hand_site].reshape(3, 3)
    rolled_relative = rolled_rotation.T @ (
        data.xpos[cube_body].copy() - data.site_xpos[hand_site].copy()
    )
    roll_slip = np.linalg.norm(rolled_relative - lift_relative)

    # 4. Turn toward the tray, remeasure slip, and run final transfer IK.
    waist_ref, final_ref, grip_relative, planned_center = plan_transfer(
        model, data, current.copy()
    )
    current = run_transition(
        model, data, current, waist_ref, 1.2, history, last_sample
    )
    run_hold(model, data, current, 0.25, history, last_sample)

    # Re-capture the relationship after waist rotation, then refine final IK.
    waist_ref, final_ref, grip_relative, planned_center = plan_transfer(
        model, data, current.copy()
    )
    current = run_transition(
        model, data, current, final_ref, 2.8, history, last_sample
    )
    run_hold(model, data, current, 0.45, history, last_sample)

    transport_center = data.xpos[cube_body].copy()
    checkpoints["ready_to_lower"] = data.qpos.copy()
    transport_rotation = data.site_xmat[hand_site].reshape(3, 3)
    transport_relative = transport_rotation.T @ (
        transport_center - data.site_xpos[hand_site]
    )
    transport_slip = np.linalg.norm(transport_relative - lift_relative)
    transport_fingers, transport_base = finger_cube_contacts(model, data)

    # 5. Select the requested wrist-unroll and lowering strategy.
    # Use wrist roll for support during transport, restore it above the tray,
    # and only then begin the vertical placement descent.
    if abs(transport_roll_deg) > 1e-6 and unroll_before_descent and not coordinated_place:
        flat_ref = current.copy()
        flat_ref[wrist_roll_adr] = original_wrist_roll
        current = run_transition(
            model, data, current, flat_ref, 3.0, history, last_sample
        )
        run_hold(model, data, current, 1.0, history, last_sample)

    if abs(transport_roll_deg) > 1e-6 and not unroll_before_descent and not coordinated_place:
        # Keep the supporting roll while lowering the cube inside the tray. Only
        # then remove the support by rolling flat, so any settling occurs in-tray.
        _, lowered_ref, grip_relative, planned_center = plan_transfer(
            model, data, current.copy(), desired_z=0.805
        )
        current = run_transition(
            model, data, current, lowered_ref, 3.5, history, last_sample
        )
        run_hold(model, data, current, 0.7, history, last_sample)
        flat_ref = current.copy()
        flat_ref[wrist_roll_adr] = original_wrist_roll
        current = run_transition(
            model, data, current, flat_ref, 3.0, history, last_sample
        )
        run_hold(model, data, current, 1.2, history, last_sample)

    descent_metrics = {}
    if place_descent:
        # 6. Track cube XY while descending and returning wrist roll.
        planning = mujoco.MjData(model)
        shoulder_id = model.joint("right_shoulder_pitch_joint").id
        elbow_id = model.joint("right_elbow_joint").id
        wrist_pitch_id = model.joint("right_wrist_pitch_joint").id
        wrist_roll_id = model.joint("right_wrist_roll_joint").id
        if coordinated_place:
            pair_ids = np.array([
                shoulder_id, elbow_id, wrist_pitch_id, wrist_roll_id,
            ])
        else:
            pair_ids = np.array([shoulder_id, elbow_id])
        pair_adrs = model.jnt_qposadr[pair_ids]
        pair_lo = model.jnt_range[pair_ids, 0] + 1e-5
        pair_hi = model.jnt_range[pair_ids, 1] - 1e-5
        # Keep the existing servo targets continuous. Their small offset from
        # actual qpos is the torque that supports the arm and cube against gravity.
        current[joint_address(model, "right_dex1_finger_joint_1")] = CLOSE_TARGET
        current[joint_address(model, "right_dex1_finger_joint_2")] = CLOSE_TARGET
        descent_rotation = data.site_xmat[hand_site].reshape(3, 3).copy()
        descent_relative = descent_rotation.T @ (
            data.xpos[cube_body].copy() - data.site_xpos[hand_site].copy()
        )
        planning.qpos[:] = current
        mujoco.mj_forward(model, planning)
        forearm_target = (
            planning.xanchor[wrist_pitch_id] - planning.xanchor[elbow_id]
        )
        forearm_target /= np.linalg.norm(forearm_target)
        roll_adr = model.jnt_qposadr[wrist_roll_id]
        roll_start = float(current[roll_adr])
        start_z = float(data.xpos[cube_body, 2])
        tray_bottom_geom = model.geom("tray_bottom").id
        tray_bottom_z = (
            data.geom_xpos[tray_bottom_geom, 2]
            + model.geom_size[tray_bottom_geom, 2]
        )
        cube_half_height = model.geom_size[model.geom("red_cube_geom").id, 2]
        target_z = tray_bottom_z + cube_half_height + 0.002
        tray_xy = data.site_xpos[tray_site, :2].copy()
        descent_start_time = data.time
        descent_duration = 4.7
        max_horizontal_error = 0.0
        max_forearm_angle_error = 0.0
        released_by = "timeout"
        update_counter = 0

        def reference_center(qref):
            """Helper: return predicted cube center for a descent reference."""
            return predict_cube_center(
                model, planning, qref, hand_site, descent_relative
            )

        def reference_forearm(qref):
            """Helper: return normalized forearm direction for a reference pose."""
            planning.qpos[:] = qref
            mujoco.mj_forward(model, planning)
            direction = (
                planning.xanchor[wrist_pitch_id] - planning.xanchor[elbow_id]
            )
            return direction / np.linalg.norm(direction)

        while data.time - descent_start_time < descent_duration + 1.0:
            elapsed = data.time - descent_start_time
            alpha = np.clip(elapsed / descent_duration, 0.0, 1.0)
            handoff_gain = minimum_jerk(np.clip(elapsed / 0.5, 0.0, 1.0))
            blend_value = minimum_jerk(alpha)
            desired_z = (1.0 - blend_value) * start_z + blend_value * target_z
            blend_rate = (30.0 * alpha ** 2 * (1.0 - alpha) ** 2) / descent_duration
            desired_z_velocity = (target_z - start_z) * blend_rate
            actual_center = data.xpos[cube_body].copy()
            max_horizontal_error = max(
                max_horizontal_error,
                float(np.linalg.norm(actual_center[:2] - tray_xy)),
            )
            bottom_gap = actual_center[2] - cube_half_height - tray_bottom_z
            forearm_now = reference_forearm(current)
            forearm_angle = np.arccos(np.clip(
                np.dot(forearm_now, forearm_target), -1.0, 1.0
            ))
            max_forearm_angle_error = max(
                max_forearm_angle_error, float(forearm_angle)
            )
            roll_ready = (
                not coordinated_place
                or abs(current[roll_adr] - original_wrist_roll) < np.radians(4.0)
            )
            if cube_tray_bottom_contact(model, data) and roll_ready:
                released_by = "tray_contact"
                break
            if bottom_gap <= 0.002 and roll_ready:
                released_by = "height_threshold"
                break

            if update_counter % 5 == 0:
                base_center = reference_center(current)
                jacobian = np.zeros((3, len(pair_adrs)))
                forearm_jacobian = np.zeros((3, len(pair_adrs)))
                epsilon = 1e-4
                for column, adr in enumerate(pair_adrs):
                    perturbed = current.copy()
                    perturbed[adr] += epsilon
                    jacobian[:, column] = (
                        reference_center(perturbed) - base_center
                    ) / epsilon
                    if coordinated_place:
                        forearm_jacobian[:, column] = (
                            reference_forearm(perturbed) - forearm_now
                        ) / epsilon
                desired_velocity = np.r_[
                    4.0 * (tray_xy - actual_center[:2]),
                    desired_z_velocity + 2.0 * (desired_z - actual_center[2]),
                ]
                if coordinated_place:
                    desired_roll = (
                        (1.0 - minimum_jerk(alpha)) * roll_start
                        + minimum_jerk(alpha) * original_wrist_roll
                    )
                    roll_row = np.zeros(len(pair_adrs))
                    roll_row[-1] = 1.0
                    system = np.vstack([
                        20.0 * jacobian,
                        3.0 * forearm_jacobian,
                        5.0 * roll_row,
                    ])
                    rhs = np.r_[
                        20.0 * desired_velocity,
                        3.0 * 3.0 * (forearm_target - forearm_now),
                        5.0 * 3.0 * (desired_roll - current[roll_adr]),
                    ]
                    qdot = np.linalg.lstsq(system, rhs, rcond=1e-4)[0]
                else:
                    qdot = np.linalg.lstsq(
                        jacobian, desired_velocity, rcond=1e-4
                    )[0]
                qdot *= handoff_gain
                qdot = np.clip(qdot, -0.45, 0.45)
                current[pair_adrs] = np.clip(
                    current[pair_adrs] + qdot * (5 * model.opt.timestep),
                    pair_lo,
                    pair_hi,
                )
            update_counter += 1
            append_step(model, data, current, history, last_sample)

        # 7. If needed, continue a slower contact-seeking terminal descent.
        # After the wrist is nearly flat, relax
        # the forearm-orientation task and use shoulder, elbow, and wrist pitch
        # to lower the actual cube gently onto the tray bottom.
        if released_by == "timeout":
            final_start = data.time
            final_adrs = pair_adrs[:3] if coordinated_place else pair_adrs
            final_lo = pair_lo[:len(final_adrs)]
            final_hi = pair_hi[:len(final_adrs)]
            final_counter = 0
            final_contact_seen = False
            while data.time - final_start < 2.5:
                actual_center = data.xpos[cube_body].copy()
                bottom_gap = (
                    actual_center[2] - cube_half_height - tray_bottom_z
                )
                max_horizontal_error = max(
                    max_horizontal_error,
                    float(np.linalg.norm(actual_center[:2] - tray_xy)),
                )
                if cube_tray_bottom_contact(model, data):
                    final_contact_seen = True
                if bottom_gap <= 0.002 and final_contact_seen:
                    released_by = "two_mm_settled_contact"
                    break
                if final_counter % 5 == 0:
                    base_center = reference_center(current)
                    final_jacobian = np.zeros((3, len(final_adrs)))
                    epsilon = 1e-4
                    for column, adr in enumerate(final_adrs):
                        perturbed = current.copy()
                        perturbed[adr] += epsilon
                        final_jacobian[:, column] = (
                            reference_center(perturbed) - base_center
                        ) / epsilon
                    final_down_speed = -0.004 if final_contact_seen else -0.012
                    final_velocity = np.r_[
                        3.0 * (tray_xy - actual_center[:2]),
                        final_down_speed,
                    ]
                    final_qdot = np.linalg.lstsq(
                        final_jacobian, final_velocity, rcond=1e-4
                    )[0]
                    final_qdot = np.clip(final_qdot, -0.25, 0.25)
                    current[final_adrs] = np.clip(
                        current[final_adrs]
                        + final_qdot * (5 * model.opt.timestep),
                        final_lo,
                        final_hi,
                    )
                final_counter += 1
                append_step(model, data, current, history, last_sample)
            if released_by == "timeout" and final_contact_seen:
                released_by = "settled_soft_contact"

        release_cube_center_z = float(data.xpos[cube_body, 2])
        release_bottom_gap = (
            release_cube_center_z - cube_half_height - tray_bottom_z
        )
        release_had_tray_contact = cube_tray_bottom_contact(model, data)

        # 8. Open only at the detected low-height/contact condition.
        closed_ref = current.copy()
        open_ref = current.copy()
        open_ref[joint_address(model, "right_dex1_finger_joint_1")] = OPEN_TARGET
        open_ref[joint_address(model, "right_dex1_finger_joint_2")] = OPEN_TARGET
        current = run_transition(
            model, data, closed_ref, open_ref, 0.7, history, last_sample
        )
        run_hold(model, data, current, 1.2, history, last_sample)
        descent_metrics = {
            "descent_release_trigger": released_by,
            "descent_release_bottom_gap_mm": 1000.0 * release_bottom_gap,
            "descent_release_had_tray_contact": release_had_tray_contact,
            "descent_max_horizontal_error_cm": 100 * max_horizontal_error,
            "descent_target_cube_center_z": target_z,
            "post_release_tray_contact": cube_tray_bottom_contact(model, data),
            "descent_max_forearm_angle_error_deg": np.degrees(
                max_forearm_angle_error
            ),
            "descent_final_wrist_roll_error_deg": np.degrees(
                abs(current[roll_adr] - original_wrist_roll)
            ),
        }

    # 9. Compute final placement, contact, slip, and success measurements.
    cube_center = data.xpos[cube_body].copy()
    tray_center = data.site_xpos[tray_site].copy()
    horizontal_error = np.linalg.norm(cube_center[:2] - tray_center[:2])
    final_rotation = data.site_xmat[hand_site].reshape(3, 3)
    final_relative = final_rotation.T @ (
        cube_center - data.site_xpos[hand_site]
    )
    relative_slip = np.linalg.norm(final_relative - grip_relative)
    overall_slip = np.linalg.norm(final_relative - lift_relative)
    finger_count, base_hit = finger_cube_contacts(model, data)
    tray_wall = model.geom("tray_wall_front").id
    tray_top = data.geom_xpos[tray_wall, 2] + model.geom_size[tray_wall, 2]
    tray_bottom = model.geom("tray_bottom").id
    tray_bottom_top = (
        data.geom_xpos[tray_bottom, 2] + model.geom_size[tray_bottom, 2]
    )
    cube_half = model.geom_size[model.geom("red_cube_geom").id, 2]
    bottom_clearance = cube_center[2] - cube_half - tray_top
    if place_descent:
        cube_bottom = cube_center[2] - cube_half
        success = (
            horizontal_error < 0.05
            and lift_fingers == 2
            and not lift_base
            and transport_fingers == 2
            and not transport_base
            and abs(cube_bottom - tray_bottom_top) < 0.012
            and np.linalg.norm(data.qvel[model.jnt_dofadr[model.joint("red_cube_freejoint").id]:][:6]) < 0.15
        )
    elif abs(transport_roll_deg) > 1e-6:
        cube_bottom = cube_center[2] - cube_half
        success = (
            horizontal_error < 0.03
            and cube_bottom >= tray_bottom_top - 0.005
            and cube_center[2] < tray_top + cube_half
            and not base_hit
        )
    else:
        success = (
            horizontal_error < 0.02
            and bottom_clearance > 0.02
            and relative_slip < 0.02
            and finger_count == 2
            and not base_hit
        )
    metrics = {
        "lift_bilateral_contacts": lift_fingers,
        "lift_base_contact": lift_base,
        "lift_cube_center_z": lift_cube_z,
        "planned_cube_center": planned_center,
        "final_cube_center": cube_center,
        "tray_center": tray_center,
        "horizontal_error_cm": 100 * horizontal_error,
        "bottom_clearance_above_tray_wall_cm": 100 * bottom_clearance,
        "cube_gripper_slip_cm": 100 * relative_slip,
        "wrist_roll_transport_deg": transport_roll_deg,
        "roll_stage_slip_cm": 100 * roll_slip,
        "overall_lift_to_final_slip_cm": 100 * overall_slip,
        "transport_center_before_lowering": transport_center,
        "transport_lift_to_center_slip_cm": 100 * transport_slip,
        "transport_bilateral_contacts": transport_fingers,
        "transport_base_contact": transport_base,
        "final_bilateral_contacts": finger_count,
        "final_base_contact": base_hit,
        "success": success,
    }
    metrics.update(descent_metrics)
    return np.asarray(history), metrics, checkpoints


def replay(model, frames):
    """Replay recorded ``frames`` in a controllable MuJoCo viewer."""
    data = mujoco.MjData(model)
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

    with viewer.launch_passive(model, data, key_callback=key_callback) as handle:
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        handle.cam.lookat[:] = [0.52, -0.04, 0.82]
        handle.cam.distance = 1.65
        handle.cam.azimuth = 145.0
        handle.cam.elevation = -22.0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        last_index = -1
        while handle.is_running():
            if state["index"] != last_index:
                data.qpos[:] = frames[state["index"]]
                data.qvel[:] = 0.0
                data.time = state["index"] / REPLAY_FPS
                mujoco.mj_forward(model, data)
                last_index = state["index"]
            handle.sync()
            if state["playing"]:
                if state["index"] < len(frames) - 1:
                    state["index"] += 1
                else:
                    state["playing"] = False
                    print("[end] paused")
            time.sleep(1.0 / REPLAY_FPS)


def main():
    """CLI entry point for executing, reporting, and optionally replaying Task 1."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--transport-wrist-roll-deg", type=float, default=0.0)
    parser.add_argument("--place-descent", action="store_true")
    parser.add_argument("--unroll-before-descent", action="store_true")
    parser.add_argument("--coordinated-place", action="store_true")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    tune_right_arm_servos(model)
    frames, metrics, _ = simulate(
        model,
        args.transport_wrist_roll_deg,
        args.place_descent,
        args.unroll_before_descent,
        args.coordinated_place,
    )
    print("Full home -> grasp -> lift -> cube-center-over-tray physics test")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("Replay: Enter/Space=Run/Pause | Delete=Reset | R=Reset+Run")
    if args.viewer:
        replay(model, frames)


if __name__ == "__main__":
    main()
