"""Untrusted child: run the agent's decoder on ONE sealed setting and emit the
predictions as a .npy array (data only). The parent computes the LER against the
hidden ground truth — which lives ONLY in the parent, never in this child's
setting root."""
import importlib.util
import os
import sys

import numpy as np

solver_path, child_root, name, out_path = sys.argv[1:5]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena_harness as ah

# child_root is a sanitized copy of the sealed setting that carries eval_dets
# but NOT eval_truth.npz — load_setting returns eval_dets and no eval_obs.
setting = ah.load_setting(child_root, name)

spec = importlib.util.spec_from_file_location("agent_solver", solver_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pred = mod.decode(setting, setting["eval_dets"])
pred = np.asarray(pred, dtype=bool)
np.save(out_path, pred)
