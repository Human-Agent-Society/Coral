# Switch-budgeted candidate routing

You inherit a weak persistent router over ten aligned candidate tracks for each queried point. Improve the general routing and visibility method to maximize mean per-video Average Jaccard while using at most four semantic-state changes after each point's query frame. The verifier re-runs your submitted method on a disjoint sealed split for final scoring.

## Hard Constraints

- Put your final method in `/app/methods/main/predict.py` and keep this exact entry point:

  ```python
  def predict(
      query_points,
      candidate_tracks,
      occlusion_logits,
      expected_dist_logits,
      candidate_model_id,
      candidate_stage,
  ) -> tuple[numpy.ndarray, numpy.ndarray]:
      ...
  ```

- Return `(state_token, occluded)`. Both arrays must have shape `[Q,T]`; `state_token` must have a non-boolean integer dtype with values in `0..9`, and `occluded` must have boolean dtype.
- A state token is semantic: `state_token = 5 * candidate_model_id + candidate_stage`. It selects the corresponding candidate track at that query and frame.
- For every query, the state-token sequence on frames strictly after the query frame may contain at most four changes. Frames at or before the query frame are not scored and do not count toward this switch budget.
- Use only the six arrays passed to `predict`. Do not read visible labels, case identifiers, index order, filenames, verifier state, process state, credentials, or files outside `/app/methods` from prediction code.
- Implement one reusable method. Do not encode visible answers, specialize to individual sequences or queries, infer hidden identities, or branch on array fingerprints, exact shapes, ordering, filenames, or case counts.
- Treat candidate order as arbitrary and independently permuted for every query. Use `candidate_model_id` and `candidate_stage` as semantic metadata; never assume a fixed candidate-axis position.
- Preserve query equivariance: reordering queries must only reorder outputs, and a query evaluated alone must receive the same prediction as it does in a batch.
- Prediction must be deterministic for identical inputs. Do not access sealed data, another container, Docker, network resources, or the verifier.
- Keep runtime and memory practical on CPU. Invalid output, more than four scored-frame switches for any query, timeout, excess resource use, or an exception fails the submission.

Run the complete visible evaluation and public contract audit with:

    python /app/selfcheck.py --audit-contract

Use `python /app/selfcheck.py --case-limit 3 --audit-contract` only for quick smoke tests; the subset score is not comparable to the complete visible score.

## What You Have

- `/app/methods/main/predict.py`: a weak route that chooses one persistent native-confidence state after each query.
- `/app/data/visible/inputs/`: 53 visible candidate-lattice cases.
- `/app/data/visible/labels/`: visible labels used by `selfcheck.py` only. They are development targets, never prediction inputs.
- `/app/data/visible/index.json`: shapes and integrity commitments for the visible cases.
- `/app/selfcheck.py`: the visible Average Jaccard evaluator, switch-budget validator, and deterministic/query/candidate/singleton audit.

For one case, `Q` is the number of query tracks, `T` the number of frames, and `K=10` the candidate count. Inputs have these shapes:

- `query_points`: float32 `[Q,3]` in `(query_frame, y, x)` order;
- `candidate_tracks`: float16 `[Q,T,K,2]` in `(x,y)` pixel coordinates;
- `occlusion_logits` and `expected_dist_logits`: float16 `[Q,T,K]`;
- `candidate_model_id` and `candidate_stage`: uint8 `[Q,K]`, containing each `(model,stage)` pair from `{0,1} x {0,1,2,3,4}` exactly once per query.

Candidate permutations and opaque case identifiers differ between visible and sealed cases. The candidates differ systematically in quality, so the visible evaluator provides an edit-run-measure signal without prescribing how routing or visibility should be modeled.

## What You Submit

Harbor collects `/app/methods` and `/app/experiment_log.md`. Helper Python files and a compact learned artifact may be placed under `/app/methods/main`; prediction must not require installation, training, labels, network access, or writable caches at grading time.

Keep every graded helper or learned artifact beside `predict.py` under `/app/methods/main`. The staged bundle may contain at most 64 regular files and 64 MiB total; each file may be at most 32 MiB and must end in `.py`, `.json`, `.joblib`, `.npz`, or `.npy`.

Keep a concise experiment record in `/app/experiment_log.md`. Record complete visible scores, the change tested, and whether you kept or reverted it.

## How It Is Judged

The verifier maps every predicted semantic state to its candidate track and computes Average Jaccard in first-query mode at pixel thresholds `1, 2, 4, 8, 16`; higher is better. Frames at or before each query frame are excluded. Scores are computed per video and macro-averaged across videos.

On a disjoint sealed split, the verifier privately permutes candidate and query order, invokes the submitted method without labels or case identity, validates the four-switch budget, and checks deterministic repeat, semantic candidate-permutation invariance, query equivariance, and singleton equivalence. These are the same behaviors exercised by `selfcheck.py --audit-contract`.
