"""Regression tests for deterministic and seeded Level-1 episode reset.

These tests verify that reset restores robot, object, controller, and time
state exactly, and that bounded randomization remains repeatable for a fixed
seed.  They run headlessly and do not require the MuJoCo viewer.
"""

from pathlib import Path
import sys

import mujoco

# Import the submitted environment helpers directly from the repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from level1_env import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    episode_state,
    load_config,
    max_state_error,
    perturb_episode,
    reset_episode,
)


def test_episode_reset_restores_robot_object_control_and_time():
    """Verify that reset exactly restores qpos, qvel, controls, and time."""
    # 1. Capture the deterministic initial episode state.
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    data = mujoco.MjData(model)
    config = load_config(DEFAULT_CONFIG)

    reset_episode(model, data, config, seed=0)
    expected = episode_state(model, data)

    # 2. Perturb both robot and cube, then advance physical simulation.
    perturb_episode(model, data, config)
    for _ in range(250):
        mujoco.mj_step(model, data)

    # 3. Reset again and require an exact match with the initial snapshot.
    reset_episode(model, data, config, seed=0)
    actual = episode_state(model, data)

    assert max_state_error(expected, actual) == {
        "time": 0.0,
        "qpos": 0.0,
        "qvel": 0.0,
        "ctrl": 0.0,
    }


def test_same_seed_is_reproducible_with_bounded_randomization():
    """Verify that repeated randomized resets with the same seed are identical."""
    # Generate two independently reset states using the same random seed.
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    data = mujoco.MjData(model)
    config = load_config(DEFAULT_CONFIG)
    reset_episode(model, data, config, seed=7, randomize=True)
    first = episode_state(model, data)
    reset_episode(model, data, config, seed=7, randomize=True)
    second = episode_state(model, data)

    # The full simulator/controller state must be exactly reproducible.
    assert max_state_error(first, second) == {
        "time": 0.0,
        "qpos": 0.0,
        "qvel": 0.0,
        "ctrl": 0.0,
    }
