"""TriFinger push — shared seed / env / run / replay helpers. The grader replays your policy with
this exact code.

Episodes are deterministic (PyBullet DIRECT mode, no visualization, no real-time sleeps): given the
same two-RNG seed (`np.random.seed()` for the initial cube pose, `move_cube.seed()` for the goal) and
the same recorded action sequence, replaying those actions on a freshly constructed env reproduces the
exact same per-step reward and return. That is what lets the sealed verifier score by CLEAN REPLAY of
an untrusted policy's action trace instead of re-running the policy in the trusted process.

Policy contract (what `solver.py` must expose):
    train(dataset_dir: str, out_dir: str, device: str = "cpu") -> None
        Train an offline-RL policy on the `trifinger-cube-push-sim-mixed-v0` dataset cached under
        dataset_dir and save a checkpoint to out_dir (checkpoint every epoch so a run that hits the
        timeout still leaves a usable model).
    class Policy(trifinger_rl_datasets.PolicyBase)
        `__init__(self, action_space, observation_space, episode_length)` loads the checkpoint from
        `os.environ.get("MODEL_DIR", "/app/submission/model")` (the PolicyBase signature is fixed by
        the upstream library, so the checkpoint location is passed via env var, not a constructor arg).
        `get_action(self, observation) -> np.ndarray` must not reference ground truth or retrain.
"""
from __future__ import annotations

import typing

import numpy as np
import gymnasium as gym

import trifinger_simulation.tasks.move_cube as move_cube
from trifinger_rl_datasets import TriFingerDatasetEnv

DATASET_NAME = "trifinger-cube-push-sim-mixed-v0"
ENV_NAME = "trifinger-cube-push-sim-expert-v0"  # only used to pick episode_length/difficulty for the
                                                 # sim env; does NOT trigger a dataset download
OBS_DIM = 97
ACT_DIM = 9


def seed_episode(n: int) -> None:
    """Seal both RNGs an episode depends on: `numpy.random` (initial cube pose,
    trifinger_rl_datasets/sampling_utils.py::sample_initial_cube_pose) and the trifinger_simulation
    module-level RNG (goal pose, trifinger_simulation.tasks.move_cube.sample_goal). Verified
    bit-identical across separate process invocations for a full 750-step episode."""
    np.random.seed(n)
    move_cube.seed(n)


def make_env(data_dir: typing.Optional[str] = None) -> TriFingerDatasetEnv:
    return typing.cast(TriFingerDatasetEnv, gym.make(
        ENV_NAME, disable_env_checker=True, visualization=False,
        flatten_obs=True, image_obs=False, data_dir=data_dir))


# --- run (policy) & replay (trace) ---------------------------------------------------------------

def run_policy_episode(env: TriFingerDatasetEnv, seed_k: int, policy) -> dict:
    """Run one episode with a LIVE policy; RECORD the per-step action trace (untrusted side)."""
    seed_episode(seed_k)
    obs, info = env.reset()
    policy.reset()
    trace, ep_return = [], 0.0
    while True:
        action = np.asarray(policy.get_action(obs), dtype=np.float32)
        trace.append(action.tolist())
        obs, rew, terminated, truncated, info = env.step(action)
        ep_return += float(rew)
        if terminated or truncated:
            break
    return {"actions": trace, "return": ep_return}


def replay_episode(env: TriFingerDatasetEnv, seed_k: int, actions: list) -> dict:
    """Replay a recorded action trace on a FRESH env (trusted side) — no policy code runs.

    every recorded action is validated before it reaches env.step(). The trace is
    plain JSON written by the UNTRUSTED child, and make_env() passes disable_env_checker=True, so
    gymnasium performs no action-space validation at all — a NaN/inf action used to flow straight
    into PyBullet and poison the per-step reward, turning the whole episode's return into NaN.
    Rejecting it here kills that at the source. A bad trace raises, which fails the scoring step and
    scores the submission 0; the sealed side carries further non-finite guards behind this one.
    """
    seed_episode(seed_k)
    obs, info = env.reset()
    ep_return, steps = 0.0, 0
    for raw_action in actions:
        action = np.asarray(raw_action, dtype=np.float32)
        if action.shape != (ACT_DIM,) or not np.isfinite(action).all():
            raise ValueError(
                f"seed {seed_k} step {steps}: action trace is not a finite {ACT_DIM}-vector "
                f"(shape={action.shape})")
        obs, rew, terminated, truncated, info = env.step(action)
        ep_return += float(rew)
        steps += 1
        if terminated or truncated:
            break
    return {"return": ep_return, "steps": steps}
