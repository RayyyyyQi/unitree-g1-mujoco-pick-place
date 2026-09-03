# Unitree G1 MuJoCo Pick-and-Place

This repository contains my entrance-test submission for a simulated humanoid
manipulation task. A fixed-base Unitree G1 29-DoF robot with a Dex1-1 gripper
must follow the instruction:

> Pick up the red cube and place it in the tray.

The implementation uses the MuJoCo Python API on macOS. It does not use a
learned policy: the task is completed by a staged controller combining
minimum-jerk interpolation, numerical inverse kinematics, position servos,
contact feedback, and an explicit success condition.

## Task and environment

The scene contains the G1 robot, a collision-enabled table, a dynamic red cube,
a tray assembled from collision geoms, visual target sites, and two fixed task
cameras. The lower body is fixed so the evaluation focuses on waist, right-arm,
and gripper manipulation. Episode reset restores robot state, controller state,
cube pose and velocity, and simulation time; optional seeded XY perturbation is
available for reproducible evaluation.

Scene parameters are defined in `configs/`. Portable MJCF models are stored in
`models/`. The Unitree asset repositories are pinned as Git submodules in
`third_party/`.

## Control policy

The controller in `scripts/full_pick_to_tray_center.py` executes the following
stages:

1. Reset to the home configuration.
2. Align waist yaw and aim the right forearm toward the cube.
3. Use numerical IK to generate a smooth shoulder/elbow approach.
4. Open the Dex1 gripper, descend, close both fingers, and verify contact.
5. Lift the cube and rotate the wrist so one finger supports it during motion.
6. Turn toward the tray and solve upper-body IK using the measured cube-to-hand offset.
7. Track the tray center during a coordinated shoulder, elbow, and wrist descent.
8. Release only after tray contact or a settled near-contact condition.
9. Evaluate final placement, contact, tracking error, and object slip.

The mirrored cube/tray variant uses
`scripts/swap_left_forearm_level_demo.py`, which recomputes the pickup and
transfer trajectories and adjusts the left elbow during torso motion to avoid
table/tray contact.

## Results

The baseline completed the full grasp, lift, transport, controlled descent, and
release sequence. It maintained bilateral finger contact during lift and
transport, avoided gripper-base contact, and finished with a horizontal cube-to-
tray center error of approximately 0.58 cm.

The same overall policy was evaluated on several setup variants:

| Setup | Outcome | Final horizontal error |
| --- | --- | ---: |
| Baseline | Success | 0.58 cm |
| Higher table | Success | 1.63 cm |
| Workspace rotated left | Success | 0.16 cm |
| Workspace rotated right | Success | 0.29 cm |
| Cube on pedestal | Success | 0.80 cm |
| Cube/tray mirrored | Success | 1.31 cm |

The variants change scene geometry rather than teleporting the robot or object.
For rotated workspaces, the complete table/cube/tray arrangement is rotated
around the robot so the test emphasizes initial waist alignment rather than
simple reach distance.

## Demonstration videos

- [Baseline - front-left camera](results/level1_pick_place_v2_fast_transport/task_camera_front_left.mp4)
- [Baseline - rear-right camera](results/level1_pick_place_v2_fast_transport/task_camera_rear_right.mp4)

The videos show the constructed simulation environment and how the complete
control policy operates; they are not recordings of the development process.

## Reproduction

```bash
git clone --recurse-submodules <repository-url>
cd unitree-g1-mujoco-pick-place
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Run the automated checks:

```bash
python scripts/smoke_test.py
python -m pytest -q tests/test_level1_reset.py
```

Run the baseline controller headlessly:

```bash
python scripts/full_pick_to_tray_center.py \
  --model models/task1/baseline.xml \
  --transport-wrist-roll-deg 90 \
  --place-descent \
  --coordinated-place
```

On macOS, interactive MuJoCo playback should be launched with the environment's
`mjpython` executable and the same script arguments plus `--viewer`.

## Limitations and lessons learned

- The fixed base removes locomotion and whole-body balance from the scope.
- Dex1 grasp stability is sensitive to contact geometry, friction, and servo lag.
- Open-loop joint targets alone produced object slip during transport; measured
  object offset, wrist support, replanning, and contact-aware release improved reliability.
- The controller is task-specific and is intended as environment/control
  validation before collecting demonstrations or integrating a learned VLA policy.

AI tools were used to assist with implementation, debugging, documentation, and
experiment organization. I reviewed the resulting design and can explain the
simulation model, controller stages, evaluation logic, and limitations.
