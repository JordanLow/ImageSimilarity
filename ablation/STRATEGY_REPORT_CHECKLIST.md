# Strategy Report Alignment Checklist

Written 2026-07-17, on explicit user request, so "are we aligned with the strategy report"
can be answered in one pass, across sessions, without re-deriving it from scratch each time.
Source document: `C:\Users\Jorda\Downloads\AAAI27_Sprint_Strategy_Report_Rev4.html` (§6-§8).
Companion to `REPRODUCIBILITY_STATUS.md` (tracks fix history for issues already found) and
`STATISTICS_METHODOLOGY.md` (tracks *how* the statistics are computed) — this document
tracks whether every piece of work the strategy report actually asks for exists yet.

**Update this file in place** whenever an item's status changes. Don't let it go stale while
the underlying work moves on — same discipline `REPRODUCIBILITY_STATUS.md` documents for
itself.

**Status legend**: ✅ Done · 🟡 Partial · ⛔ Not started · 🚫 Resolved/Won't-do (deliberate,
not a gap) · ⏸ Deferred/Priority (not abandoned — explicitly flagged to revisit) · 🔒 Blocked
(exact unblocking condition stated)

## §6.0 — Blocking preconditions

| # | Item | Status | Where tracked / why |
|---|---|---|---|
| 1 | Commit `pose_scoring.py` (inlier + pose thresholds), reproduces P/R/F1 91.3/94.1 | ✅ | Repo root `pose_scoring.py`, documented acceptance test in its own docstring |
| 2 | Fix the evaluation statistic (`compare_vggt_to_truth.py`) | ✅ | That exact file was deleted in the 2026-07-07 "big refactor" commit; its job is now split across `pose_scoring.py` (decision layer) + `ablation/significance.py`/`aggregate_significance.py` (canonical stats) — see `STATISTICS_METHODOLOGY.md`'s incident log |
| 3 | Determinism (seeds, `cv2.setRNGSeed` before RANSAC, frozen BatchNorm, pinned VGGT commit) | ✅ | `RANSAC_SEED=0` + `cv2.setRNGSeed()` in `geometry_filter.py`/`vggt_signals.py`/every ablation matcher script; `vggt_signals.py` pins a literal commit hash; model `.eval()` freezes BN (no explicit named handling beyond that, but functionally equivalent) |
| 4 | Rerun ASpanFormer with `--breakpoint-value 0` on labeled pairs | 🚫 | **Will not be run — explicit user decision, feasibility concerns (2026-07-17).** Every manifest on disk (`aspan_all_manifest_shard{1,2}.jsonl` + both `_reconstructed` variants) stays at breakpoint=50. Deliberate scope call, not an oversight — don't re-raise this without a new reason to revisit. |

## §6.1 — Table A (retrieval baselines, 10 required rows)

Canonical report: `_local/ablation_stage1_report.html`.

| # | Row | Status | Where tracked / why |
|---|---|---|---|
| 1 | SSCD | ✅ | Stage 1 report, `_local/ModelComboSSCD.py` |
| 2 | DINOv2 global features | ⛔ | **Never built as its own row.** `ablation/retrieval_ablation_colab.ipynb`'s own markdown explicitly treats DINOv3-raw as satisfying the strategy report's combined "DINOv2/DINOv3 global features" line — DINOv2 itself has zero implementation anywhere in the repo/Downloads/D:\ (confirmed via full-text search) |
| 3 | DINOv3 global features ("DINOv3-raw") | ✅ | Stage 1 report, `_local/ModelComboDINOv3Raw.py` |
| 4 | CLIP or SigLIP-2 | 🟡 | CLIP half done (Stage 1 report, `_local/ModelComboCLIP.py`); SigLIP never built — zero matches anywhere |
| 5 | "The old JOCCH ViT ensemble" (existing weights) | 🟡 | `_local/ModelCombo.py` + `JOCCH.pt` — code ready, retrieval not yet run (Stage 1 report §11 Remaining Work) |
| 6 | "Your DINOv3 fusion" (production model) | ✅ | Stage 1 report's "DINO (production)" row, `_local/ModelComboDINO.py` |
| 7-8 | The two VGGT-feature retrieval variants (existing weights) | ⛔ | **Orphaned code exists, never run.** `_local/ModelComboVGGT11.py` (single-layer, layer 11) and `_local/ModelComboVGGTCombined.py`/`ModelComboVGGTCombinedNoGrad.py` (multi-layer, `[4,11,17,23]` fused) — architecturally match the description, added 2026-06-24, but never imported/instantiated/trained anywhere outside `_local/`. No "(existing weights)" checkpoint found for either. |
| 9 | Isolating row: DINOv3-alone + SSL head | ⛔ | Named only in the strategy report's own W9/schedule tables ("A9"). No script, checkpoint, or manifest anywhere. |
| 10 | Isolating row: fusion-without-SSL | ⛔ | Same as #9 ("A10") — planning-table mention only |

## §6.2 — Table B (verification ablations)

Split across two reports per this project's own Stage1/2/3 taxonomy (not the strategy
report's flat numbering) — see [[project_ablation_taxonomy_and_glossary]].

| Row | Variant | Status | Where tracked |
|---|---|---|---|
| B1 | Keypoint-count floor sweep | ✅ | `_local/ablation_stage3_report.html` |
| B2 | Inlier-ratio-only | ✅ | `_local/ablation_stage3_report.html` |
| B4 | Full system (baseline) | ✅ | Reference row, all reports |
| B5 / B6 | Pose components: rotation+xy vs. fov+z | ✅ | `_local/ablation_stage3_report.html` |
| B8 | Homography decomposition (classical pose) | ✅ | `_local/ablation_stage3_report.html` |
| B9 | Essential-matrix pose **from ASpanFormer correspondences directly** (distinct from B8's homography-decomposition route) | ⛔ | Not implemented anywhere. Definition confirmed precisely from the strategy report this session — previously only vaguely tracked as "classical pose, alternate source." |
| B10 | MASt3R relative pose | ✅ | `_local/ablation_stage3_report.html` — see `MAST3R_ABLATION_METHODOLOGY.md` for the full degeneracy-bug/retrofit story |
| B11 | Learned classifier (LR/MLP) | ✅ | `_local/ablation_stage3_report.html` |
| B12 / B13 | Alignment off / raw-vs-filtered alignment keypoints | ⛔ | **Not implemented. Per user decision: deferred to a future stage-less "overall" report**, not Stage 2 or Stage 3 — isolates the alignment step, which sits between the two and doesn't cleanly belong to either. |
| B14 | Matcher swap: LightGlue | 🟡 | `_local/ablation_stage2_report.html` — Shard 1 done, Shard 2 pending |
| B15 | Matcher swap: RoMa (must-run) | ⏸ | On hold pending professor feasibility call — `ROMA_ABLATION_METHODOLOGY.md` |

## §6.3 — Table C (sensitivity curves)

`_local/sensitivity_curves.py` implements all 4 panels, already run once for Shard1+2 (no
Shard 3 — not labeled yet). **Per user decision: curves belong in the report of the stage
they pertain to, not one combined document.**

| Panel | Curve | Belongs in | Status |
|---|---|---|---|
| (i) | Inlier-ratio F1 sweep, 0.65 marked | Stage 3 | ✅ In `_local/ablation_stage3_report.html` |
| (ii) | Pose-score ROC/PR, 2.13 marked | Stage 3 | ✅ In `_local/ablation_stage3_report.html` |
| (iii) | 2D (inlier, pose) F1 heatmap | Stage 3 | ✅ In `_local/ablation_stage3_report.html`, Shard1+2 only |
| (iv) | Retrieval top-K (5→15) recall curve | Stage 1 | ⛔ **Not yet added to the Stage 1 report** — tracked here as pending, do next time that report is touched |

Shard 3 replot for any panel: 🔒 blocked on Shard 3 labeling.

## §6.4 — End-to-end recall/survival audit

**Per user decision: goes into a future stage-less "overall" report, not any Stage N
report.**

| Item | Status | Notes |
|---|---|---|
| Full 4-stage survival table (retrieval top-10 → keypoint → inlier → pose), over all 2,636 verified pairs | 🟡 | Real infrastructure exists (`_local/survival_audit.py`, output at `D:/DINO OUTPUTS/survival_audit.json`/`.md`) — genuinely distinct from the Stage 1 report's retrieval-only §7/§8 analysis — but covers only 577/2,636 pairs (Shard1+2 only) and uses retrieval **top-15**, not the strategy report's top-10. Ready-to-extend infrastructure, not a from-scratch build, whenever the overall report happens. |

## §6.5 — Pose-signal validity controls

| Item | Status | Notes |
|---|---|---|
| (a) Synthetic battery — ~200 exact-copy pairs, controlled crop/rotation/halftone/perspective transforms, confirm pose scores stay sub-threshold | ⛔ | Not built. Proves pose-judge robustness to the *kind* of degradation real scans show, with ground truth known by construction — distinct from validating against whatever degradation already exists in the labeled set. |
| (b) Per-component distributions (violin + ROC-AUC per pose component, true-copy vs. same-scene) | ✅ | `_local/pose_signal_analysis.py`, output at `D:/DINO OUTPUTS/pose_signal_analysis.png`/`.json`, Shard1+2. Empirically backs the pose-component weighting design (previously "asserted from anecdote" per the strategy report's own words). |
| (c) Crop-stage QC audit — YOLOv11 page→photo-crop detector P/R on a 100-page audited sample | ⛔ | Only a 30-image *training* set (`D:\croptraining\`) and unaudited inference output on Jinchaji exist — nobody's checked predictions against ground truth. The rebuttal's ">98% detection, manually checked" claim remains unformalized. |

## §6.6 — Human evidence (annotation reliability, trace-type taxonomy, historian time-saved study)

**Per user decision: deferred this cycle for feasibility (human-labeling/user-study
logistics), flagged as a priority to revisit — not abandoned.**

| Item | Status | Notes |
|---|---|---|
| Codebook + independent PI/intern dual-labeling + Cohen's kappa + adjudicated gold labels | ⏸ | Nothing exists — no codebook document, no kappa computation, no dual-labeling infra. Current labels (`match_manifest_shard{1,2}.csv`) are single-rater only. |
| Trace-type taxonomy (halftone/crop/rotation/retouch/recaption tagging of the ~577 confirmed positives, per-type pass rates) | ⏸ | Nothing exists |
| Historian time-saved study (4-6 historians, within-subjects, pipeline-verified vs. plain retrieval) | ⏸ | Nothing exists — no recruiting, no materials, no results |

**Flagged, not investigated**: `Downloads/Kill_Gate_Guide_v2.md` and `Plan_Amendments_v2.md`
(2026-07-13) describe a superficially similar but apparently distinct workstream ("Kill
Gate," N/R/O relation labeling, co-author "Deqian," a git branch not present locally). User
should confirm whether this supersedes, parallels, or predates Rev4 whenever convenient.

## §6.7 — Cross-domain transfer (Jinchaji)

**Per user decision: same treatment as §6.6 — deferred, flagged as priority, not
abandoned.**

| Item | Status | Notes |
|---|---|---|
| Frozen-pipeline run on JOCCH's Jinchaji dataset (2,146 photos, 230 annotated pairs) | ⏸ | Zero mentions of "Jinchaji" anywhere on disk. **Do not confuse with the repo's own "JOCCH"** — that's an unrelated model-ensemble ablation (`_local/ModelCombo.py` + `JOCCH.pt` weights, tracked in Stage 1's Remaining Work), not this cross-domain dataset. |

## §6.8 — Statistical reporting (applies to every table)

| Item | Status | Notes |
|---|---|---|
| Bootstrap 95% CIs (10,000 resamples) on every P/R/F1/PR-AUC cell | ✅ | `ablation/statistics.py`, `eval_retrieval.py`, `eval_stage2.py` all implement this consistently |
| McNemar exact test vs. full pipeline, Holm-corrected across the table | ✅ | `ablation/significance.py` (canonical exact-binomial McNemar) + `aggregate_significance.py` (the one place Holm runs) |
| 3 seeds for the final SSL model; 1 seed for ablation variants, stated | 🟡 | Ablation variants do state their single seed (e.g. `RANDOM_STATE=42` in `learned_classifier.py`); 3-seed final-model retrain not confirmed/tracked here — out of this checklist's Stage-2/3 scope, belongs with the production-model training track |
| 5 cv2-seed repeats on Shard 1 (geometric-stage variance ≈0) | 🔒 | **Blocked, not just "not done."** `geometry_filter.py` caches raw ASpanFormer keypoints (`raw_mkpts0/1_resized`) per pair *before* RANSAC, so re-fitting with 5 different `cv2.setRNGSeed()` values would be a cheap CPU-only script *if* Shard 1 sidecars existed locally. Checked 2026-07-17: the only cached `.npz` sidecars anywhere on disk are for **Shard 6** (`D:/DINO OUTPUTS/DINO_Output_Shard6/aspan_output/aspan_sidecars/`) — itself explicitly out-of-scope for this paper ("Shards 4–6 are out of scope this cycle"). No Shard 1 sidecars found locally; the canonical manifest's own `sidecar_path` values are Colab-ephemeral (`/content/...`), suggesting they were never synced anywhere permanent. **Unblocks via**: locating Shard 1 sidecars on Google Drive, or a fresh GPU ASpanFormer rerun on Shard 1 to regenerate them. |
| Disclose transductive SSL setup + Shard-3-excluded retrain control | ❓ | Not checked this session — out of Stage 2/3 scope, belongs with the production-model training track |

## §6.9 — Workload/budget

Informational only (Colab Pro+ cost planning) — not a tracked checklist item.

## §7 — Rigorous ablation protocol (methodology rules, not deliverables)

Cross-referenced against actual project practice rather than tracked as checklist rows:

- **7.1** Declare split roles (Shard1=dev, Shard2=validation, Shard3=test, frozen) — ✅
  followed throughout (`STATISTICS_METHODOLOGY.md`, every ablation report's methodology
  section).
- **7.2** One factor at a time from cached artifacts — ✅ the entire B1/B2/B5/B6/B8/B11
  design (`pose_scoring.score_row()`'s CLI-flag architecture).
- **7.3** Every claimed mechanism gets an isolating row — 🟡 followed for pose-aware
  (B2/B8/B5-B6/B11); **not yet followed for the retrieval-ensemble claim** (Table A's
  missing isolating rows #9/#10 above).
- **7.4** Paired statistics, not point estimates — ✅ bootstrap CIs + McNemar everywhere.
- **7.5** Honest denominators — ✅ the "never let a new-pair-only stat read as overall
  performance" rule in the Stage 1 report is this rule in direct action.
- **7.6** Named denominators for the three-stage-funnel metrics — ✅ Stage 1's Table A vs.
  Coverage Gap distinction, Stage 2/3's "new-pair diagnostic vs. overall performance" splits,
  all explicitly follow this recipe.

## §8 — VLM module

**Explicitly, deliberately out of scope** — the strategy report itself states this was
dropped from the current paper submission by team decision (2026-07-06), deferred whole to a
future cycle. Not a gap; don't track it as one.

## Pointers

- Technical statistics methodology: `ablation/STATISTICS_METHODOLOGY.md`
- Reproducibility fix history: `ablation/REPRODUCIBILITY_STATUS.md`
- Stage 1 report: `_local/ablation_stage1_report.html`
- Stage 2 report: `_local/ablation_stage2_report.html`
- Stage 3 report: `_local/ablation_stage3_report.html`
- MASt3R (B10) design rationale: `ablation/MAST3R_ABLATION_METHODOLOGY.md`
- RoMa (C2/B15) hold status: `ablation/ROMA_ABLATION_METHODOLOGY.md`
- Pipeline stage taxonomy: `project_ablation_taxonomy_and_glossary` memory
