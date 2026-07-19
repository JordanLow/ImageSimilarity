# Statistics methodology — canonical source of truth for the paper's significance testing

Decided 2026-07-16. Companion to `MAST3R_ABLATION_METHODOLOGY.md` / `ROMA_ABLATION_METHODOLOGY.md`
(those cover *what* B10/RoMa test; this covers *how every ablation's numbers get computed and
compared*, for every stage). Written to be pointed at from any future session — human or
Claude — working on this paper's statistics, so the same rigor holds for the rest of the
project regardless of who's in the seat.

## The incident that made this necessary

A reviewer flagged that the paper's quoted B11 headline row (P=98.0/R=95.6/F1=96.8) and
`statistics_results.json`'s B11 McNemar entry described different classifiers. Investigation
confirmed it was real: `ablation/statistics.py` was **independently retraining its own copy**
of the B11 logistic-regression classifier from scratch, purely to get per-pair predictions for
the McNemar test — a separate training run (different CV seed, and a real bug: `LogisticRegressionCV(cv=5)`
passes a bare int, which per scikit-learn's `check_cv()` silently defaults to unshuffled folds
and never receives `random_state` at all) from whatever `_local/learned_classifier.py` did to
produce the headline numbers. Two independently-trained models, same name, different actual
decision boundaries.

Worse: no per-pair predictions or fitted model had ever been persisted anywhere for B11, by
either script — the exact model that produced 98.0/95.6/96.8 was unrecoverable. And the script
that produced it lived in `_local/`, which is entirely gitignored — the paper-critical script
was, by construction, invisible to version control. A wider audit then found this wasn't an
isolated bug: **three independent McNemar implementations existed** (`ablation/statistics.py`,
`ablation/eval_stage2.py`, and the MASt3R notebook), each with real methodological drift —
different B4 baseline source files, different large-sample approximation policy, different
Holm-correction scope, different join keys. This document and the pipeline it describes are
the fix: one canonical implementation, one canonical baseline source, one place correction
happens, and provenance recorded well enough that "retrain to get predictions" never has to
happen again.

## Standing principles

These apply to every ablation, every stage, no exceptions:

1. **Provenance always recorded.** Every reported statistic must trace to exactly which
   trained model/variant/run produced it. A number without a traceable artifact behind it
   isn't citable.
2. **Never silently retrain when reuse is required.** If a headline number comes from a
   specific trained/fitted artifact, any downstream test (significance, CI, etc.) must use
   that exact artifact's own predictions — not an independently-retrained "equivalent" one,
   however identical the nominal configuration looks. This is precisely how the B11 incident
   happened.
3. **Persist per-pair predictions, not just aggregate confusion matrices.** Aggregate
   tp/fp/tn/fn counts can't support a paired significance test later without retraining. Any
   artifact meant to feed McNemar (or any other paired test) must save per-example decisions.
4. **One canonical statistics implementation, not per-ablation-family duplicates.** Every
   ablation family imports `ablation/significance.py` rather than reimplementing McNemar/Holm.
5. **Family-wise corrections get fully recomputed, never patched in isolation.** Holm ranks
   all p-values in a family together — changing one row's raw p-value means every other row's
   Holm-adjusted value must be recomputed too, even if its own outcome doesn't change.
6. **Determinism is a requirement, not a nice-to-have.** Every stochastic step (CV folds,
   regularization search, threshold selection, bootstrap resampling) needs a fixed, recorded
   seed, and the whole pipeline should be re-runnable from raw manifests to final numbers as
   one traceable process.
7. **Same methodology across every ablation, regardless of stage.** The same statistical
   procedure (same test, same correction, same alpha) is used whether the row is Stage 1, 2,
   or 3. No family gets bespoke treatment. (Stage 1's retrieval work is the one deliberate
   exception to the *test* — see "Stage 1 is out of scope" below — but not to the discipline.)
8. **New ablations plug into this pipeline; they don't reimplement it.** Adding a row means:
   produce per-pair predictions, wire the script into `ablation/significance.py` for its
   McNemar test, tag the result with the right `family`, and add a row to the registry below.
   No new script should compute its own p-value from scratch.

## Stage 1 is out of scope for this document

Stage 1 (retrieval: SSCD/CLIP/DINOv3-raw/JOCCH) uses a wholly different evaluation
methodology — human review, Recall@K/mAP, precision/recall — not McNemar/Holm, because it
isn't a paired binary-classifier comparison. It has its own canonical report,
`_local/ablation_stage1_report.html` (see project memory). Nothing below applies to it.
Everything below governs Stage 2 (keypoint matching) and Stage 3 (pose scoring), which *are*
paired binary-classifier comparisons against the B4 baseline.

## Architecture

| File | Role |
|---|---|
| `ablation/significance.py` | Canonical `mcnemar_exact(b, c)` (always exact two-sided binomial — see below) and `holm_correct(tests, family_key)`. The only place these formulas are implemented. Every other script imports from here. |
| `ablation/learned_classifier.py` | B11's training script (moved from `_local/`, git-tracked). Fixed CV bug (explicit `StratifiedKFold(shuffle=True, random_state=...)`, never a bare int passed to `cv=`). Persists per-pair predictions to `D:/DINO OUTPUTS/b11_predictions/{variant}.jsonl` and the fitted pipeline to `D:/DINO OUTPUTS/b11_models/{variant}.joblib` for every feature-group variant — this is what makes B11 reproducible without retraining. |
| `ablation/statistics.py` | Computes Stage 3 raw (uncorrected) McNemar tests for B1/B2/B5/B6/B8 (both shards) and B11 (Shard 2, loaded from `learned_classifier.py`'s persisted predictions — never retrained here). Tags every test `family="stage3"`. Does **not** apply Holm correction. |
| `ablation/eval_stage2.py` | Computes Stage 2 raw McNemar tests for C1/LightGlue (=B14) and C2/RoMa, both shards, tagged `family="stage2"`. Does **not** apply Holm correction. The old pooled-cross-shard "combined" McNemar test was removed — it double-counted against per-stage-family pooling. |
| `ablation/stage3_mast3r_ablation_colab.ipynb` (cell `cell-mcnemar` onward) | Computes B10/MASt3R's raw Stage 3 McNemar test using the same `significance.py` (copied to the Colab runtime, matching the pattern every other notebook in this project already uses) and the same canonical B4 source, joined on `(source_id, target_id)`. Writes `mast3r_stage3_results.json` to Drive for local sync. |
| `ablation/aggregate_significance.py` | The **only** place Holm-Bonferroni correction happens. Reads every upstream raw-test source, pools by `family`, runs `holm_correct()` once per family, writes the final `D:/DINO OUTPUTS/statistics_final.json` / `statistics_final.md` — the file the paper actually cites. Flags a family PROVISIONAL if an expected row hasn't been synced yet, rather than silently correcting over a smaller family without saying so. |
| `ablation/ablation_utils.py` | Canonical ground-truth (`load_ground_truth`) and manifest (`load_aspan_all`, `load_judge_manifest`) loaders. Every script imports from here — `ablation/eval_retrieval.py` was the last holdout with its own duplicate GT parser; it now imports `load_ground_truth` too. |

## McNemar policy: always exact binomial

Every prior implementation switched to a continuity-corrected chi-squared approximation once
`b + c >= 25` (two of the three did; the MASt3R notebook's did not, which was itself part of
the inconsistency). Decision: **drop the approximation branch entirely, always compute the
exact two-sided binomial test.** Every discordant-pair count observed across this project so
far (max seen: `b=41, c=21`, well under a thousand) is computationally trivial to compute
exactly — there's no performance reason to approximate, and exact removes an entire axis of
inconsistency rather than picking a side of it.

## Holm-Bonferroni scope: per-stage family, pooled across shards

**Decision:** two families, corrected independently:
- **Stage 3 family** = {B1, B2, B5, B6, B8, B11, B10} × {Shard1, Shard2}, wherever each has
  data — every row from both shards, corrected together as one family.
- **Stage 2 family** = {C1/LightGlue (B14), C2/RoMa} × {Shard1, Shard2} — corrected together,
  separately from Stage 3.

Rationale: a family should be "the set of comparisons a reader would mentally group together
when asking whether any of them are false positives from multiple testing" — that's naturally
per-pipeline-stage in this paper's structure (Stage 2's matcher choice and Stage 3's pose-model
choice are separate design decisions with separate B4 baselines being challenged), and pooling
across shards within a stage is correct because Shard 1 and Shard 2 results for the *same* row
are not independent hypotheses — they're dev/validation halves of one claim about that row.
Per-shard-only correction (the old `statistics.py` behavior) under-corrects by not accounting
for the other shard's tests of the same row; whole-paper-single-family correction would
over-correct by penalizing Stage 2 rows for how many Stage 3 rows happen to exist, which is an
arbitrary coupling.

**Consequence:** no single script sees every family member at once (Stage 3's B10 is computed
in a Colab notebook, B1–B11 locally). `aggregate_significance.py` exists specifically to
solve this — it's the one point in the whole pipeline where a family's full membership is
assembled and corrected. Until B10's notebook output is synced locally, the Stage 3 family is
explicitly marked PROVISIONAL in `aggregate_significance.py`'s output rather than silently
computed over 6 rows instead of 7.

## B4 baseline source: `aspan_all_manifest_shard{N}.jsonl`

**Decision:** keep the existing `statistics.py` behavior — B4's per-pair predictions come from
`aspan_all_manifest_shard{N}.jsonl` (VGGT-judged rows) via `pose_scoring.score_row()` with its
default thresholds (`INLIER_RATIO_THRESHOLD=0.65`, `POSE_COMPONENT_THRESHOLD=2.13`), not
`Shard{N} Judge Manifest.jsonl`. The two files were empirically identical for the labeled
subset at decision time, but only one should be treated as canonical going forward — every
script (`statistics.py`, `eval_stage2.py`, the MASt3R notebook) now reads the same file via
the same `score_row()` call, so B4 can never silently fork again. Import
`INLIER_RATIO_THRESHOLD`/`POSE_COMPONENT_THRESHOLD` as constants from `pose_scoring.py` rather
than re-typing the numbers — this is what closes the loophole that let the MASt3R notebook's
hardcoded `inlier_ratio >= 0.65 and pose_score <= 2.13` silently drift from the real source if
`pose_scoring.py`'s constants ever changed.

## Registry of ablation rows

| Row | Name | Stage | Family | Script/notebook | Output artifact | Status |
|---|---|---|---|---|---|---|
| B1 | Keypoint-count floor sweep | 3 | stage3 | `ablation/statistics.py` (`pairs_b1`) | `statistics_results.json` | Done, both shards |
| B2 | Inlier-ratio-only filter | 3 | stage3 | `ablation/statistics.py` (`pairs_b_variants`) | `statistics_results.json` | Done, both shards |
| B4 | Full system / paper baseline | 3 | — (the reference every other row is tested against, not tested itself) | `pose_scoring.py` via `aspan_all_manifest_shard{N}.jsonl` | n/a | Canonical baseline |
| B5 | Pose component: rotation+xy only | 3 | stage3 | `ablation/statistics.py` (`pairs_b_variants`) | `statistics_results.json` | Done, both shards |
| B6 | Pose component: fov+z only | 3 | stage3 | `ablation/statistics.py` (`pairs_b_variants`) | `statistics_results.json` | Done, both shards |
| B8 | Classical pose (homography decomposition) | 3 | stage3 | `ablation/statistics.py` (`pairs_b8` via `_local/classical_pose.py`) | `statistics_results.json` | Done, both shards |
| B9 | Essential-matrix pose derived directly from ASpanFormer's own keypoint correspondences (confirmed precise definition 2026-07-17 from the strategy report §6.2 — distinct from B8's homography-decomposition route; not yet implemented in this pipeline) | 3 | stage3 (once implemented) | — | — | **Not yet built** — add here when it is |
| B10 | MASt3R pose head | 3 | stage3 | `ablation/stage3_mast3r_ablation_colab.ipynb` + `_local/recompute_b10.py` | `mast3r_stage3_results.json` (`D:/DINO OUTPUTS/`) | **Synced 2026-07-17** — `_local/recompute_b10.py` independently re-derived the retrofitted numbers from the raw judged manifest (exact match to the prior session's figures) and wrote the missing sync artifact; `aggregate_significance.py`'s Stage 3 family is now complete, not PROVISIONAL. Still provisional in the "blind Shard 3 confirmation" sense per `MAST3R_ABLATION_METHODOLOGY.md`'s Amendment — that part remains open. See `_local/ablation_stage3_report.html`. |
| B11 | Learned classifier (logistic regression, `vggt_aggr` feature set) | 3 | stage3 | `ablation/learned_classifier.py` → `ablation/statistics.py` (loads persisted predictions) | `b11_predictions/lr_vggt_aggr.jsonl`, `b11_models/lr_vggt_aggr.joblib`, then `statistics_results.json` | **Retrained under the fixed CV procedure 2026-07-16** — new numbers: P=0.961/R=0.968/F1=0.964 (threshold=0.55, C=2.7826), replacing the paper's previous 98.0/95.6/96.8 |
| B13 | Referenced in the strategy report by name only; definition not yet located in this codebase | ? | ? | — | — | **Not yet built** — do not guess its definition; confirm against the strategy report before implementing |
| B14 / C1 | LightGlue keypoint matcher | 2 | stage2 | `ablation/eval_stage2.py` | `stage2_geometry_results.json` | Shard 1 done (F1=0.910 vs B4's 0.913, n.s.); Shard 2 pending — see project memory |
| C2 | RoMa keypoint matcher | 2 | stage2 | `ablation/eval_stage2.py` | `stage2_geometry_results.json` | **On hold** pending professor feasibility call (RoMa ~20-30x heavier) — see `ROMA_ABLATION_METHODOLOGY.md` |

Other feature-group variants B11 was swept across (`geom_only`, `vggt_comps`, `all_feats`,
`mlp_all_feats`) are persisted the same way but are not headline/paper rows — see
`learned_classifier.py`'s `FEATURE_GROUPS`.

## Adding a new ablation row

1. Produce per-pair `(source_id, target_id) -> {y_true, y_pred}` predictions from a
   deterministic, seeded process. If it involves fitting a model, persist the fitted
   artifact and its predictions (see B11's pattern) — never plan to retrain later to get a
   significance test.
2. Compute its McNemar test against B4 via `ablation.significance.mcnemar_from_predictions`
   (or `mcnemar_exact` if you already have `b`/`c`). Tag the result dict with `row`, `shard`,
   and `family` (`"stage2"` or `"stage3"` — do not invent a third family without updating this
   document and `aggregate_significance.py`'s `family_expected` map).
3. Feed the raw test into whichever upstream loader `aggregate_significance.py` already has
   for your family, or add a new `load_*` function there following the existing pattern
   (load raw tests, tag family, return a list — no Holm correction inside the loader).
4. Add a row to the registry table above.
5. Do not compute or report a Holm-adjusted p-value from your own script. Run
   `ablation/aggregate_significance.py` and cite its output.

## Open items (as of 2026-07-17)

- **B10/MASt3R**: **sync closed 2026-07-17** — `_local/recompute_b10.py` produced
  `D:/DINO OUTPUTS/mast3r_stage3_results.json` from the raw judged manifest, independently
  reproducing the retrofitted numbers exactly; the Stage 3 family is now complete in
  `aggregate_significance.py`'s output. Still open: the blind Shard 3 confirmation run noted
  in `MAST3R_ABLATION_METHODOLOGY.md`'s Amendment (blocked on Shard 3 labeling, not on
  anything in this pipeline).
- **B14/C1 Shard 2** and **C2/RoMa** (on hold): not yet in `stage2_geometry_results.json`, so
  the Stage 2 family is currently thin. Re-run `aggregate_significance.py` once available.
- **B9, B13**: named in the strategy report but not yet implemented anywhere in this
  pipeline. Confirm their actual definitions against the strategy report before building —
  do not infer them from the B-number sequence.
