# Learn a TriFinger cube-pushing policy from a fixed offline dataset

You inherit an **offline reinforcement learning** problem on the TriFinger robot simulator: a
three-fingered robot must push a cube to a target position, and you may only learn from a **fixed,
pre-collected dataset** of past trajectories (no live simulator interaction during training). The
shipped `methods/main/solver.py` is a deliberately weak behavior-cloning baseline. You train a policy
and save a checkpoint; a sealed verifier then **reloads your checkpoint and re-runs your policy** on a
hidden, disjoint batch of episodes you never see (it does NOT re-train), scoring the mean **return**
across those episodes (higher is better).

## Hard Constraints

- **CRITICAL (artifact-eval timeout safety):** your `train()` MUST persist the checkpoint to `out_dir`
  (`/app/submission/model`) **incrementally, not only at the very end**. The grader scores whatever is
  in `/app/submission/model` at the deadline; saving only at the end and hitting the timeout leaves an
  empty submission and scores 0. Populate it early and keep overwriting.
- You may only edit code under `/app/methods/main/`; you may add sibling `.py` modules. **The two
  entrypoints and their signatures must not change** — the verifier imports them directly:
  - `train(dataset_dir: str, out_dir: str, device: str = "cpu") -> None` — train an offline-RL policy
    on the dataset cached in `dataset_dir` and **save everything needed to reload it** into `out_dir`.
    Any layout works as long as your own `Policy.__init__` can read it back — several files, an
    `.npz`, a subdirectory, whatever you like. The verifier only checks that `out_dir` exists and is
    not empty. Naming your main checkpoint `model.pt` (as the shipped starter does) is **suggested**
    for consistency, not required.
  - `class Policy(trifinger_rl_datasets.PolicyBase)` — `__init__(self, action_space, observation_space,
    episode_length)` must load your checkpoint from `os.environ.get("MODEL_DIR",
    "/app/submission/model")` (the base class signature is fixed by the upstream library, so the
    checkpoint path travels through this env var, not a constructor argument);
    `get_action(self, observation) -> np.ndarray` returns the 9-dim torque action for a 97-dim flat
    observation. No ground truth, no re-training, no network calls inside `get_action`.
- **Your policy must be deterministic given the observation stream** — no unseeded randomness inside
  `get_action`. The verifier scores by replaying your recorded action trace on a fresh copy of each
  episode; nondeterminism makes the replayed score diverge from what you saw.
- Train only on the shipped dataset; do not download or fabricate additional trajectories, and do not
  call the live simulator to generate new rollouts during training (this is an *offline*-RL task).
- A crash or an empty checkpoint directory scores 0. Note what non-determinism actually costs you:
  the verifier does **not** run a determinism check, so a non-deterministic policy is not detected or
  penalised as such — instead your local evaluation and the graded value simply stop agreeing, and
  you have no way to tell which one is right. Also note that a crash *inside a single episode* only
  zeroes **that** episode (it contributes 0.0 to the mean), not the whole submission; a crash while
  loading your `Policy` zeroes everything.

**The grading budget, in full — size your policy against it.** You get no per-attempt feedback, so
these numbers are published rather than left for you to guess:

| | grading run | your own session / free `selfcheck.py` |
|---|---|---|
| episodes rolled out | **32** sealed (hidden), then replayed once each | 100 visible (`selfcheck.py`), plus any seeds you pick yourself |
| judged data scale vs. visible | **0.32×** the self-check pool | — |
| wall-clock, whole grading container | **9000 s** | your session budget is 32400 s (9 h) |
| wall-clock, your policy's rollout phase | **3600 s** for all 32 episodes (~112 s/episode) | none |
| wall-clock, the sealed replay afterwards | 1800 s (does not run your code) | — |
| CPU / memory | **4 cores / 4096 MB** | 4 cores / 16384 MB |

Two consequences worth planning around. First, **grading gives your policy less RAM than your own
session does** (4 GB vs 16 GB): a checkpoint you can train comfortably may still be too heavy to
*load and run* at grading time — size the deployed model, not just the training job. Second,
`get_action` is called 750 times per episode × 32 episodes = 24,000 times inside that 3600 s; a
per-call cost above ~140 ms will not finish. If your policy does run out of wall clock, the episodes
that already completed are still scored and the unreached ones count as return 0.0 — a slow policy
degrades, it is not thrown away — but that is a floor, not a plan.

## What You Have

- `/app/data/trifinger_dataset/`: the visible offline dataset `trifinger-cube-push-sim-mixed-v0`
  (~2.9M transitions of `(observation, action, reward, timeout)`, mixed quality — expert, weak, and
  near-random trajectories). Load it with the standard `trifinger_rl_datasets` API: `gym.make(
  "trifinger-cube-push-sim-mixed-v0", data_dir="/app/data/trifinger_dataset").unwrapped.get_dataset()`.
  Observations are 97-dim flat vectors (robot joint state, cube pose+keypoints, goal, previous action);
  actions are 9-dim joint torques in `[-0.397, 0.397]`.
- `/app/methods/main/solver.py`: the weak BC starter (`train()` + `Policy`) — **this directory is what
  gets graded**, together with the checkpoint you save under `/app/submission/model/`. Improve it in
  place or replace the algorithm entirely (e.g. a genuine offline-RL method).
- `/app/trifinger_score.py`: the **exact** seeding / env / run / replay helpers the verifier uses.
  Read it to see precisely how episodes are seeded and how a recorded action trace is replayed. It
  does **not** contain the metric-to-score mapping — that lives only on the sealed side. All you need
  to know about it is that your score rises monotonically with the mean return.
- `/app/selfcheck.py`: a free, unlimited local dry-run (`python /app/selfcheck.py`) that trains your
  solver to a scratch checkpoint and reports the mean return (± standard error) on 100 visible
  episodes drawn from the **same episode distribution** as the hidden sealed batch (independent
  draws from one family, with no seed shared between the two pools). Use it for **relative**
  comparison — "is change A better than change B" — where it is reliable, because both sides are
  measured on the same fixed episodes and the episode-to-episode noise cancels. Do **not** read a
  single visible mean as a point estimate of your sealed score: at intermediate skill levels the
  per-episode spread is wide enough that the two pools' means can differ by ~50 return purely by
  sampling, in either direction. And it stops being informative at all the moment you tune
  against it. You may also evaluate a trained policy on episode seeds of your
  own choosing via `/app/trifinger_score.py` (`seed_episode` + `run_policy_episode`): simulator use
  for *evaluation and model selection* is allowed and encouraged; only *training on simulator
  rollouts* is forbidden.

## What You Submit

Edit `/app/methods/main/solver.py`, keeping the `train` / `Policy` contract:

```python
def train(dataset_dir: str, out_dir: str, device: str = "cpu") -> None:
    ...  # load the dataset, train, save a checkpoint into out_dir (checkpoint every epoch/N steps)

class Policy(PolicyBase):
    def __init__(self, action_space, observation_space, episode_length):
        ...  # load the checkpoint from os.environ.get("MODEL_DIR", "/app/submission/model")
    def get_action(self, observation):
        ...  # return a 9-dim torque action
```

Then **run it to produce the checkpoint**: `python /app/methods/main/solver.py` trains on the visible
dataset and saves the checkpoint to `/app/submission/model/`. Leave both the edited `solver.py` and the
trained checkpoint in place — there is no submit step and no per-attempt feedback; the verifier grades
once at the end. The headroom over plain behavior cloning: the dataset mixes trajectories of very
different quality, and a method that can tell the good transitions from the bad ones can exploit that
spread — imitating everything uniformly, as plain BC does, cannot.

## How It Is Judged

After your run, the verifier copies `methods/main/` and `/app/submission/model/` into a sealed sandbox,
**loads your checkpoint and re-runs your `Policy.get_action()`** in a rollout on a HIDDEN, disjoint
batch of episodes (it does NOT re-train), replays the recorded action trace on a fresh copy of each
episode to independently recompute the return, and scores the **mean return** across the batch (higher
is better). Your score rises monotonically with the mean return, so pushing the return up is always the
goal. A policy that fails to load, or an empty `/app/submission/model/`, scores 0; an episode your
policy crashes in contributes 0.0 to the mean and the rest still count. Actions must be finite 9-dim
vectors — a trace containing NaN or inf is rejected outright and scores 0.
