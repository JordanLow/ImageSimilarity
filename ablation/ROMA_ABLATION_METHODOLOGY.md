# RoMa (C2) ablation — methodology decision record

Decided 2026-07-13. Written to be lifted into the paper's methods/limitations
sections, and to explain to a future reader (including future us) why RoMa's
Stage 2 integration looks the way it does rather than something more elaborate.

## The problem

`geometry_filter_roma.py`'s original implementation selected correspondences
via `model.sample(warp, certainty, num=n_matches)`, which performs a
**fixed-count** weighted-random draw from RoMa's dense warp field —
confirmed by reading RoMa's actual `sample()` source: certainty controls
*which* points are drawn, never *how many*. Every candidate pair, match or
not, produced exactly `n_matches` (5000) raw correspondences. Smoke-testing
confirmed the consequence directly: 10/10 arbitrary candidates passed the
Stage 2 geometry filter, including pairs LightGlue found essentially nothing
in.

This breaks the assumption the pipeline's Stage 2 gate relies on for
ASpanFormer and LightGlue — that raw match count is itself evidence of match
likelihood (sparse detectors naturally find fewer candidates on weak pairs).
For RoMa, raw count is a constant, so it carries zero discriminative signal.

Separately and consequently: `vggt_signals.py` (Stage 3) does not use Stage
2's own fundamental-matrix result for scoring at all — it independently
re-estimates a homography from Stage 2's saved keypoints via its own RANSAC
fit, and that homography is what both (a) produces `aspan_2d_inlier_ratio`
(the actual signal `pose_scoring.py` thresholds at 0.65) and (b) warps the
source image before it's fed to VGGT. So the fixed-count bug's real-world
cost is specifically: (1) Stage 2 stops doing its one job (cheaply rejecting
hopeless candidates before the expensive VGGT step), inflating compute, and
(2) more low-evidence candidates reach the homography-fit-and-warp step,
where a spurious fit could feed VGGT a garbled aligned pair.

## Options considered

- **A — certainty-threshold selection.** Replace the fixed-count draw with
  direct thresholding on RoMa's certainty tensor, so raw count varies with
  match quality again. Everything downstream (sidecar schema, Stage 3's
  homography re-derivation, `aspan_2d_inlier_ratio`, `pose_scoring.py`
  thresholds) stays identical across all three matchers.
- **B — A, plus a RoMa-side homography-inlier-ratio gate**, replacing the
  count-based gate with a ratio Stage 2 computes itself. Introduces a second
  free parameter (the ratio cutoff) without a clear benefit once A already
  restores count-informativeness.
- **C — use RoMa's dense warp directly for VGGT alignment**, replacing the
  homography-based alignment with a per-pixel remap, and replacing
  `aspan_2d_inlier_ratio` with a RoMa-native metric (e.g. certainty
  coverage). This is the only option that actually exploits RoMa's real
  advantage over sparse matchers (non-rigid, locally-varying correspondence,
  not collapsible to a single global homography). It requires: a new sidecar
  format (a full dense warp is ~12MB/candidate vs. KB-scale sparse sidecars —
  a real storage/upload cost at shard scale), new alignment code with
  boundary-handling edge cases inside the *shared* `vggt_signals.py`, and —
  the expensive part — a properly-derived threshold for the new metric,
  comparable in scope to the existing Shard1/Shard2 threshold-derivation
  effort already documented for the ASpanFormer baseline. Realistically a
  multi-day research task on its own, not a quick patch, and rushing it would
  likely produce a *less* defensible threshold than A's minimal change.

## Decision: Option A

Chosen for feasibility under a tight timeline without sacrificing rigor:
it fixes the actual bug (count carrying no signal) with a small, fully
contained change (one function in `geometry_filter_roma.py`), touches no
shared/validated code, and needs no newly-invented, unvalidated threshold —
it reuses the existing, already-accepted gating mechanism, just fed honest
counts. The only new parameter is the certainty cutoff itself, for which we
use RoMa's own author-recommended `sample_thresh` default rather than an
invented value.

## Explicitly accepted limitation

Option A still collapses RoMa's dense correspondence field down to a sparse
point set for a single global homography fit at Stage 3 (the same treatment
ASpanFormer/LightGlue get). It does **not** exploit RoMa's actual comparative
advantage — dense, potentially non-rigid correspondence — because doing so
properly (Option C) requires a dedicated new alignment representation and its
own rigorously-derived decision threshold, which is out of scope here. This
was a deliberate scoping choice, not an oversight: it keeps the Stage 3/4
scoring methodology byte-identical across every ablation arm, preserving the
"vary only the independent variable" property that makes the ablation a
controlled comparison in the first place. A dedicated dense-alignment
treatment of RoMa is legitimate, worthwhile future work.

## Suggested report language

**Methods:** "RoMa's default correspondence sampling draws a fixed number of
points regardless of match quality, which would make Stage 2's evidence-based
filtering step uninformative. We instead select correspondences via
confidence thresholding (using RoMa's recommended certainty cutoff), which
restores the property — used elsewhere in the pipeline — that raw
correspondence counts reflect match likelihood, while leaving the downstream
geometric verification and decision pipeline (Stage 3/4) unchanged across all
compared matchers, so that the ablation isolates the effect of the Stage 2
matcher alone."

**Limitations / future work:** "Our RoMa integration normalizes its dense
output into the same sparse-correspondence, homography-based representation
used for ASpanFormer and LightGlue, to keep the scoring methodology identical
across ablation arms. This does not exploit RoMa's capacity for dense,
non-rigid correspondence estimation, which a single global homography cannot
represent. A dedicated treatment — using RoMa's dense warp directly for
alignment and deriving a RoMa-native decision threshold with the same rigor
applied to the baseline's thresholds — is a natural direction for future
work, but requires its own threshold-derivation study and was out of scope
under our timeline."
