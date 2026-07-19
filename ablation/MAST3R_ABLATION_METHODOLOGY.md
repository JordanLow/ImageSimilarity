# B10 ablation — MASt3R as an alternative Stage-3 pose head

Decided 2026-07-14. Companion to `ROMA_ABLATION_METHODOLOGY.md` (that one covers the
Stage 2 matcher swap; this one covers the Stage 3 pose-model swap named row B10 in
`AAAI27_Sprint_Strategy_Report_Rev4.html` §6.2). Written to be lifted into the paper's
methods section and to explain the design to a future reader.

## What's actually being tested

The pipeline's forensic signal is not literal camera-pose accuracy. After the Stage 2
homography aligns a candidate pair, a true match (same flat photograph, re-captured)
should reduce to near-zero apparent camera motion; a same-scene-different-shot pair has
genuine 3D structure a 2D homography cannot fully explain away, so a 3D-aware model given
the aligned pair reports ("hallucinates") a residual pose shift proportional to that
leftover inconsistency. VGGT's `pose_component_score` is this hallucinated-residual
magnitude, not a validated pose measurement. B10 must reproduce the *same* emergent setup
with MASt3R substituted for VGGT — not give MASt3R a from-scratch two-view pose task, which
would be a different experiment answering a different question.

**Consequence:** MASt3R must be fed the same homography-aligned pair VGGT is fed, not the
raw image pair. This was an explicit correction mid-design — the first draft of this plan
proposed running MASt3R on both the aligned and raw pair "for robustness," which would
have quietly changed what's being measured. A raw-pair run remains useful as an
implementation sanity check (confirming the pose-extraction code works on an unrelated
pair) but must never be reported as part of the B10 result.

## The substitution (single-variable isolation)

| Stage | Baseline B4 (VGGT) | B10 (MASt3R) |
|---|---|---|
| Stage 2 candidates + keypoints | ASpanFormer sidecar | Unchanged — identical sidecars/candidates as B4 |
| Alignment homography + Filter 2 | `_estimate_homography_from_sidecar()` → inlier_ratio ≥ 0.65 | Unchanged |
| Pair fed to the pose model | Aligned pair from `prepare_vggt_inputs()` | Same aligned pair |
| Pose model | VGGT-Omega → `pose_enc` → `pose_component_score` | MASt3R → `fast_reciprocal_NNs` on the aligned pair → essential-matrix RANSAC → `recoverPose` → `mast3r_pose_score` |
| Filter 3 | `pose_component_score` ≤ 2.13 | `mast3r_pose_score` ≤ threshold, derived fresh on Shard 1 dev only |

Holding Stage 2 and Filter 2 fixed and swapping only the pose model mirrors how B8/B9
already isolate classical pose sources against B4 — B10 extends the same spectrum with the
strongest neural alternative.

## `mast3r_pose_score` construction (original design — see Amendment below for the
## retrofit that supersedes the "rotation as primary" assumption)

- **Primary: rotation angle** (geodesic distance from identity, `cv2.Rodrigues` on the
  recovered R). Scale-free and well-posed from an essential matrix; also already
  established (Table B Finding B, B5 = B4 exactly) as carrying the dominant share of the
  discriminative signal in this pipeline.
- **Secondary: translation-direction consistency.** Essential-matrix decomposition only
  yields translation direction, not magnitude — stated openly as a limitation of this
  route rather than glossed over.
- **Recorded for completeness, not the primary score: PnP-derived scaled translation.**
  MASt3R's `pts3d_in_other_view` output is already expressed in the aligned source's frame
  (confirmed from the actual `visloc.py` reference implementation and GitHub issue #103's
  documented indexing gotcha); pairing it with the matched 2D pixels in the target via
  `solvePnP` gives properly-scaled translation, unlike the essential-matrix route. Reported
  as a secondary column / agreement check against the rotation signal, not run through its
  own full bootstrap+McNemar treatment unless it disagrees with the primary result in a way
  worth chasing.

**This turned out to be wrong about which signal is primary** — see the Amendment section
below. Left as-written above for an honest record of the original (reasonable, but
empirically incorrect) design assumption: rotation was expected to dominate by analogy
with VGGT's own pose-component weighting (Finding B), but MASt3R's *specific*
rotation-recovery route (essential-matrix decomposition) turned out to have a failure mode
VGGT's pose encoder doesn't share.

## Threshold derivation — do not repeat Fatal F1

`mast3r_pose_score` is not on the same scale as VGGT's `pose_component_score`; 2.13 is
meaningless for it. A new threshold must be derived on **Shard 1 (dev) only**, via the same
sensitivity-curve methodology already used for the existing thresholds, then frozen before
touching Shard 2 (validation) or Shard 3 (test). This is the exact discipline the strategy
report's Fatal F1 (threshold leakage) finding demands elsewhere in the paper — deriving and
reporting a new threshold on the same data would reintroduce the identical bug.

## What justifies this in the report

1. Table B row B10: P/R/F1 with 95% bootstrap CI + PR-AUC on Shard 1 (dev, threshold
   derivation only) and Shard 2 (validation, the reported comparison), same audited
   candidate set as every other row.
2. McNemar vs. B4, Holm-corrected, same format as B2/B5/B6/B8/B11.
3. PnP-vs-essential-matrix agreement check (secondary/diagnostic).
4. Failure-mode analysis against MASt3R's documented primary failure mode — wrong matches
   on repetitive/symmetric regions (confirmed from the MASt3R-SfM paper) — a concrete,
   checkable hypothesis given magazine pages share mastheads/borders/layout conventions
   across issues.
5. Training-domain symmetry note: MASt3R's 14 training datasets (Habitat, ARKitScenes,
   MegaDepth, Static Scenes 3D, BlendedMVS, ScanNet++, CO3Dv2, Waymo, MapFree, WildRgb,
   VirtualKitti, Unreal4K, TartanAir, plus an internal set) are exclusively real-3D-parallax
   scenes — none are document/flat-surface/archival-photo content, the same
   picture-of-a-picture domain gap already acknowledged for VGGT (§6.5). Stated explicitly
   so the comparison reads as fair to both models.

## Other MASt3R facts on record (from API/idiosyncrasy research, prior turn)

- License: CC BY-NC-SA 4.0, non-commercial research only (confirmed from the repo) — fine
  for an ablation/comparison, worth one sentence if the release package ever touches
  MASt3R code, since NC terms don't mix cleanly with the CC BY 4.0 NCR dataset release.
- No official documented answer exists (as of this research) for the exact two-view
  pose-recovery indexing convention — GitHub issues #101 and #103 on `naver/mast3r` are
  both open and unanswered. Our indexing choice (`matches_im1` 2D pixels ↔
  `pts3d_in_other_view` 3D points, both already in the aligned-source's frame) is a
  deliberate, documented decision made where even upstream maintainers haven't clarified
  intended usage — worth stating plainly rather than presenting as obviously correct.
- Doppelgangers++ (the strategy report's named nearest-neighbor, §3.2) uses MASt3R
  differently than we do — it feeds MASt3R's intermediate decoder features into a trained
  classifier head, not an explicit derived pose. Worth stating precisely when
  distinguishing our approach.

## Status

Implemented and run in full 2026-07-14: `ablation/mast3r_signals.py` +
`ablation/stage3_mast3r_ablation_colab.ipynb`. Results and the amendment below reflect
the actual completed Shard 1 + Shard 2 run.

## Amendment 2026-07-14 — essential-matrix degeneracy found; formula search retrofitted

**Finding.** 63 of 1,280 scored candidates (4.9%) return a rotation angle of *exactly*
180.000° from `cv2.recoverPose` — the mathematical ceiling of axis-angle rotation, not a
real measurement. This is not noise: these 63 rows have *higher* match counts (mean 1,991
vs 1,558) and *higher* ASpanFormer inlier ratios (mean 0.96 vs 0.65) than the rest of the
data — the best-aligned pairs, not the worst. Cross-referenced against ground truth: 10.6%
of true-copy pairs (61/577) hit this degeneracy, versus 0.3% of impostor pairs (2/701) — a
35× skew toward exactly the class of pair this ablation needs to score correctly.

**Mechanism.** This is a known failure mode of two-view essential-matrix geometry: when
the true camera baseline is near zero — precisely what a correctly Stage-2-aligned true
copy looks like — the R/t decomposition becomes poorly constrained, and `recoverPose`'s
cheirality-based disambiguation can flip to a spurious 180° "twisted pair" solution instead
of the correct near-identity rotation. The original design (above) assumed rotation would
carry the dominant signal by analogy with VGGT's own weighting; that analogy doesn't hold
here because VGGT's pose encoder isn't derived via essential-matrix decomposition and
doesn't share this specific singularity.

**Retrofit.** Rather than hand-selecting a replacement formula, `translation_only` (score
= `mast3r_pnp_translation_scaled` alone, structurally immune to the rotation-angle ceiling
since it never uses that signal) was added as one more candidate to the *same* disciplined
Shard-1-dev-only empirical sweep the methodology already used — `cell-threshold-derivation`
in the notebook now includes it. It wins on Shard 1 dev (F1=0.923 vs the original
6-candidate search's best of F1=0.908), and — more importantly, since its threshold was
frozen on Shard 1 only — it *also* significantly beats the original formula on the
held-out Shard 2 (McNemar 13 vs 3, p=0.0213), evidence this isn't Shard-1-dev overfitting.
Reported results now use this retrofitted formula.

**A residual methodological caveat, disclosed rather than hidden.** The *threshold value*
for `translation_only` was fit blind to Shard 2 (no leakage in that sense), but the
*decision to add it as a candidate at all* was made after observing that the original
formula lost significantly to B4 on Shard 2 and diagnosing why — a milder, formula-level
form of hindsight bias than direct threshold leakage, but real. **A fully blind Shard 3
run (planned for the final week before the report) will resolve this**: rerunning the
identical frozen `translation_only` rule (no re-tuning) against Shard 3 ground truth, which
was never touched during either the original search or this retrofit, is the clean
confirmatory test. Until that run, this ablation's B10 result should be read as
"best current estimate, pending blind confirmation," not a fully leakage-free result —
consistent with how this project has treated every other threshold-derivation step.

**Why retrofit rather than report the original (buggy) result and stop there.** This
ablation study serves two roles: (1) a rigorous, leakage-free comparison for the paper, and
(2) genuinely finding the best configuration for each pipeline stage, since any stage found
significantly better by an alternative gets swapped in. Reporting a result known to be
distorted by a diagnosed, fixable geometric artifact would serve neither role well — it
would misrepresent MASt3R's real ceiling on this task. The blind Shard 3 run is what makes
extracting this potential compatible with academic rigor: retrofit now for the best
current estimate, confirm blind before it's final.
