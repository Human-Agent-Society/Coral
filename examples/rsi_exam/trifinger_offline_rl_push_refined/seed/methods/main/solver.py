"""Starter offline-RL solver for TriFinger cube push — IMPROVE ME.

ARTIFACT-EVAL contract. THIS DIRECTORY (`/app/methods/main/`) IS WHAT GETS GRADED, together with the
checkpoint you save under `/app/submission/model/`. The workflow is split in two:

    train(dataset_dir, out_dir, device="cpu") -> None
        Load the offline dataset (`trifinger-cube-push-sim-mixed-v0`, already downloaded into
        dataset_dir at image-build time) and train a policy, SAVING a checkpoint into out_dir. The
        shipped baseline plain-BC training is a few seconds; checkpoint after every epoch so a run
        that hits the timeout wall still leaves a usable model.

    class Policy(PolicyBase)
        Loads the checkpoint from `os.environ.get("MODEL_DIR", "/app/submission/model")` (see
        trifinger_score.py's module docstring for why the path is env-driven, not a constructor arg)
        and implements `get_action(observation) -> np.ndarray`. No ground truth, no retraining, no
        network calls inside `get_action`.

So the loop is: edit this file -> run it (`python /app/methods/main/solver.py`) to write the
checkpoint to `/app/submission/model/` -> leave it in place. The verifier reloads that checkpoint and
re-runs `Policy.get_action()` in a sealed rollout on hidden episodes; it does NOT re-train.

The shipped method is deliberately WEAK behavior cloning (plain MSE regression obs->action, tiny
64-unit MLP, 1 epoch): it only imitates the mixture of expert/weak-expert/random trajectories in the
dataset, it cannot filter out the bad ones or stitch together good sub-trajectories from mediocre
demonstrations. Headroom: a bigger/longer-trained BC network already helps some, but the dataset mixes
trajectories of very different quality, and a method that can tell them apart -- rather than imitating
all of them uniformly -- is what actually exploits that spread. Which method, and how to identify the
good transitions, is the open part of the task. You may add sibling `.py` modules beside this
file. Keep the two names (`train`, `Policy`) — the verifier imports them directly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from trifinger_rl_datasets import PolicyBase, PolicyConfig

ACT_LOW, ACT_HIGH = -0.397, 0.397
ARCH = "starter_bc_v1"


class BCNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        raw = self.net(obs)
        scale = (ACT_HIGH - ACT_LOW) / 2.0
        mid = (ACT_HIGH + ACT_LOW) / 2.0
        return torch.tanh(raw) * scale + mid


def _load_dataset(dataset_dir: str):
    import gymnasium as gym
    import trifinger_rl_datasets  # noqa: F401  (registers the gym envs)

    env = gym.make("trifinger-cube-push-sim-mixed-v0", disable_env_checker=True, data_dir=dataset_dir)
    return env.unwrapped.get_dataset()


def train(dataset_dir: str, out_dir: str = "/app/submission/model", device: str = "cpu",
          hidden: int = 64, epochs: int = 1, batch_size: int = 4096, lr: float = 1e-3,
          seed: int = 0) -> None:
    """Train the weak BC baseline and SAVE weights + arch config into out_dir."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[train] loading dataset from {dataset_dir} ...")
    t0 = time.time()
    dataset = _load_dataset(dataset_dir)
    obs = dataset["observations"].astype(np.float32)
    act = dataset["actions"].astype(np.float32)
    print(f"[train] loaded {obs.shape[0]} transitions in {time.time() - t0:.1f}s")

    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0) + 1e-6

    model = BCNet(obs.shape[1], act.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    obs_t = torch.from_numpy((obs - obs_mean) / obs_std).to(device)
    act_t = torch.from_numpy(act).to(device)
    n = obs_t.shape[0]

    def _checkpoint() -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(), "obs_mean": obs_mean, "obs_std": obs_std,
            "obs_dim": obs.shape[1], "act_dim": act.shape[1], "hidden": hidden, "arch": ARCH,
        }, out / "model.pt")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, max(1, n // batch_size)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            pred = model(obs_t[idx])
            loss = loss_fn(pred, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        _checkpoint()  # checkpoint EVERY epoch — timeout-safe (overwrite)
        print(f"[train] epoch {epoch}: mse={epoch_loss / n_batches:.6f} "
              f"elapsed={time.time() - t0:.1f}s")

    print(f"[train] saved model -> {out_dir}")


class Policy(PolicyBase):
    """Loads the checkpoint saved by train() and implements PolicyBase.get_action()."""

    def __init__(self, action_space, observation_space, episode_length):
        super().__init__(action_space, observation_space, episode_length)
        model_dir = Path(os.environ.get("MODEL_DIR", "/app/submission/model"))
        ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
        self.model = BCNet(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.obs_mean = torch.from_numpy(ckpt["obs_mean"])
        self.obs_std = torch.from_numpy(ckpt["obs_std"])
        self.action_space = action_space

    @staticmethod
    def get_policy_config() -> PolicyConfig:
        return PolicyConfig(flatten_obs=True, image_obs=False)

    def get_action(self, observation) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(observation, dtype=np.float32))
            obs_n = (obs_t - self.obs_mean) / self.obs_std
            action = self.model(obs_n.unsqueeze(0)).squeeze(0).numpy()
        return np.clip(action, self.action_space.low, self.action_space.high)


if __name__ == "__main__":
    # Runnable: train on the visible mixed dataset and persist the checkpoint the verifier will load.
    data_dir = os.environ.get("DATA_DIR", "/app/data/trifinger_dataset")
    model_dir = os.environ.get("MODEL_DIR", "/app/submission/model")
    device = os.environ.get("DEVICE", "cpu")
    train(data_dir, model_dir, device=device)
    print(f"[main] model ready at {model_dir} — the verifier will load it and run Policy.get_action() "
          f"on a sealed rollout.")
