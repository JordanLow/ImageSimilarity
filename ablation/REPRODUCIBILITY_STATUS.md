# Reproducibility & Alignment Status — rebuttal-ready dossier

Written 2026-07-16, on explicit request, so a future session (or the user directly) can
answer "did you fix the reproducibility/alignment issues?" precisely, with evidence, in one
pass — no re-investigation. Update this file in place whenever a row's status changes;
don't let it go stale while the underlying work moves on.

**One-line answer, if asked cold:** every reproducibility/alignment issue that was actually
diagnosed this cycle is closed, with committed code and re-run evidence. Several planned
ablation rows are still incomplete — but that's disclosed, itemized incompleteness (§3
below), not a hidden gap, which is the actual rigor bar this whole effort was about meeting.

## 1. The triggering incident and the systemic audit it led to

| # | Concern | Root cause | Fix (files) | Evidence | Status |
|---|---|---|---|---|---|
| A | External reviewer note: paper's quoted B11 row (P=98.0/R=95.6/F1=96.8) didn't match `statistics_results.json`'s B11 McNemar entry — different underlying model | `ablation/statistics.py` independently **retrained a second copy** of B11 to get McNemar predictions (different CV seed, plus a real bug: `LogisticRegressionCV(cv=5)` — bare int, silently unshuffled per sklearn's `check_cv()`, ignores `random_state`) | `ablation/learned_classifier.py` (moved from ungitted `_local/`, now git-tracked): fixed CV bug (explicit `StratifiedKFold(shuffle=True, random_state=...)`), persists per-pair predictions (`D:/DINO OUTPUTS/b11_predictions/*.jsonl`) + fitted models (`b11_models/*.joblib`). `statistics.py` now *loads* these, never retrains. | Re-run 2026-07-16: `python ablation/learned_classifier.py` → new headline `vggt_aggr` numbers P=0.961/R=0.968/F1=0.964 (threshold=0.55, C=2.7826), replacing the unrecoverable 98.0/95.6/96.8. `python ablation/statistics.py` confirmed loading (not retraining) these: `[B11] Loading persisted predictions ... Loaded 636 Shard2-test predictions` | **Closed** — B11 is now reproducible from committed code, no retraining step |
| B | Audit found this wasn't isolated: **3 independent, divergent** McNemar/Holm implementations existed (`statistics.py`, `eval_stage2.py`, MASt3R notebook) — different B4 baseline source files, different large-sample chi-squared-vs-exact policy, different Holm scope/pooling | Each ablation family's script grew its own significance-testing code instead of sharing one | `ablation/significance.py` (canonical exact-binomial McNemar + family-scoped `holm_correct`), imported by all three surfaces. `ablation/aggregate_significance.py` — the **one place** Holm correction now happens, pooled per-stage-family across shards. Canonical B4 source unified to `aspan_all_manifest_shard{N}.jsonl` via `pose_scoring.score_row()` everywhere (imported constants, not re-typed thresholds) | `python ablation/aggregate_significance.py` re-run 2026-07-16: correctly pools 11 raw Stage 3 tests (B1/B2/B5/B6/B8/B11 × 2 shards), correctly reports the family **PROVISIONAL** (not silently "complete") pending B10 sync. **Re-run again 2026-07-17** after B10's sync closed (§3): 13 raw tests, 7/7 rows, family now **complete** | **Closed**, including the B10 completeness gap as of 2026-07-17 |
| C | `ablation/eval_retrieval.py` had its own duplicate ground-truth CSV parser, diverging from the canonical loader | Not wired into `ablation_utils` | Switched to `ablation_utils.load_ground_truth` | `py_compile` + smoke-tested | **Closed** |
| D | `requirements.txt` pinned no scikit-learn/joblib at all, unpinned numpy — a "fixed" rerun wasn't environment-reproducible | Never pinned | Pinned `numpy==2.4.4`, `scikit-learn==1.9.0`, `joblib==1.5.3` to the actually-verified-working versions | `python -c "import sklearn; print(sklearn.__version__)"` → 1.9.0 | **Closed** |
| E | No durable, git-tracked methodology document explaining how the paper's statistics are computed — the structural reason this could drift silently in the first place | N/A (a documentation gap, not a code bug) | `ablation/STATISTICS_METHODOLOGY.md` — incident writeup, standing principles, McNemar/Holm policy + rationale, canonical B4 source, full ablation-row registry with per-row status | File exists, git-tracked; referenced by `statistics.py`/`eval_stage2.py`/`aggregate_significance.py`'s own docstrings | **Closed** (content needs updating as rows in §3 complete — that's expected maintenance, not a gap) |
| F | Stage 1 (retrieval) had **no quantitative Table A** (Recall@K/mAP) at all — "setup done, not run" for SSCD/CLIP/DINOv3-raw/JOCCH | `eval_retrieval.py` existed but had never been run against real data | Ran it for DINO/SSCD/CLIP/DINOv3-raw, Shard 1; added as new §2 of `_local/ablation_stage1_report.html` | `D:/DINO OUTPUTS/table_a_shard1_results.json` — DINOv3-raw R@1=98.3%/mAP=98.3%, SSCD 96.2%/94.9%, CLIP 84.6%/86.4% | **Closed** for the 4 models with data; JOCCH explicitly flagged pending (§3), not silently absent |
| G | Naming collision: `_local/ablation_stage1_report.html` (old, pre-overhaul) had the same "Stage 1" name as the current retrieval report, and its McNemar numbers were computed by the pre-fix, buggy pipeline (its B11-LR row shows the old P=0.957/R=0.972/F1=0.965, not the corrected 0.961/0.968/0.964) | Historical naming predates this project's Stage1/2/3 taxonomy — "Stage 1" originally meant "first CPU-only ablation round," not "pipeline stage 1" | Renamed to `_local/preliminary_ablations_report.html`, retitled, marked superseded/do-not-cite in-document, kept as the placeholder for the eventual Stage 3 report rewrite | File renamed and retitled 2026-07-16 | **Closed** (disambiguated + flagged); its actual pose-scoring content is *not yet rewritten* with the new pipeline — that's future work, not a currently-misreported number, since it's now explicitly marked do-not-cite |
| H | User asked directly whether Stage 1's new Table A met the same rigor bar as Stage 2/3 — it didn't: no bootstrap CIs, no persisted per-query predictions (the same failure class as B11's original bug: aggregate-only, unauditable) | Oversight when Table A was first added | `eval_retrieval.py` extended with `bootstrap_ci()` (10,000 resamples over query sources, seed 42 — same convention as `statistics.py`'s `bootstrap_ci`) and `--per-query-dir` persistence (one `recall_at_{k}`/`ap` JSONL per model) | Re-run 2026-07-16: `D:/DINO OUTPUTS/table_a_per_query_shard1/{DINO,SSCD,CLIP,DINOv3Raw}.jsonl` (292 rows each); CIs added to the report — found DINOv3-raw > SSCD > CLIP is statistically distinguishable at R@1/mAP (non-overlapping CIs), SSCD-vs-CLIP at R@10+ is not | **Closed** |
| I (pre-dates this session) | Strategy report's Fatal F2: "published numbers irreproducible from repo" — the committed judge implemented a *different* classifier than the one that produced the logged results | Off-repo notebook heuristic never committed | `pose_scoring.py` (repo root) — `INLIER_RATIO_THRESHOLD=0.65`, `POSE_COMPONENT_THRESHOLD=2.13` as named constants, with a documented acceptance test in its own docstring | This session's `statistics.py` re-run independently reproduces the acceptance test exactly: Shard 1 P=0.867/R=0.963/F1=0.913, Shard 2 P=0.902/R=0.984/F1=0.941 — bit-for-bit match to `pose_scoring.py`'s docstring | **Confirmed closed** (fixed before this session; re-verified as a byproduct of this session's `statistics.py` re-runs) |

## 2. If a reviewer/rebuttal specifically asks...

- **"Is B11 reproducible?"** Yes — `python ablation/learned_classifier.py` regenerates the
  fitted model, its predictions, and metadata (sklearn/numpy version, seed, selected C,
  tuned threshold) deterministically. The number quoted in the paper should be P=0.961/R=0.968/F1=0.964,
  not the earlier 98.0/95.6/96.8 (that run's exact artifact no longer exists and its CV
  procedure had a real bug — disclosed, not hidden, in `ablation/STATISTICS_METHODOLOGY.md`).
- **"Are your significance tests consistent across ablations?"** Yes — one exact-binomial
  McNemar implementation (`ablation/significance.py`), one Holm correction step
  (`ablation/aggregate_significance.py`), pooled per-stage-family across shards, documented
  and justified in `ablation/STATISTICS_METHODOLOGY.md`.
- **"Can I reproduce your headline pipeline numbers (P=86.7/96.3/91.3 etc.) from the repo?"**
  Yes — `pose_scoring.py`'s named constants + acceptance test, independently re-confirmed
  2026-07-16.
- **"Is your retrieval-stage comparison (Table A) statistically supported, not just point
  estimates?"** Yes as of 2026-07-16 — bootstrap 95% CIs on every cell, per-query predictions
  persisted for independent audit.
- **"Is everything finished?"** No, and here's exactly what's left (§3) — presented as
  planned, tracked work, not a discovered gap.

## 3. What's still open (disclosed, not hidden)

| Item | Why it's open | Where it's tracked |
|---|---|---|
| ~~B10/MASt3R notebook output not yet synced locally~~ **Closed 2026-07-17** | `_local/recompute_b10.py` independently re-derived B10's retrofitted numbers from the raw judged manifest (exact match to prior figures) and wrote `mast3r_stage3_results.json` | `ablation/aggregate_significance.py`'s Stage 3 family is now complete (7/7 rows), not PROVISIONAL — see `_local/ablation_stage3_report.html` §1/§3. Blind Shard 3 confirmation (a separate, still-open item — see `MAST3R_ABLATION_METHODOLOGY.md`'s Amendment) remains blocked on Shard 3 labeling. |
| B14/LightGlue Shard 2, C2/RoMa | RoMa on hold pending professor feasibility call (~20-30x heavier than ASpanFormer) | `ablation/ROMA_ABLATION_METHODOLOGY.md`; Stage 2 family in `aggregate_significance.py` is thin until these land |
| B9, B13 | Named in the strategy report by row number only; definitions never confirmed against it, never implemented | `ablation/STATISTICS_METHODOLOGY.md`'s registry — explicitly says "don't guess, confirm against the strategy report" |
| JOCCH retrieval | No manifest exists locally at all, only `JOCCH.pt` weights | `_local/ablation_stage1_report.html` §11 Remaining Work |
| Shard 2 for SSCD/CLIP/DINOv3-raw | Only Shard 1 has been run | Same §10 |
| DINO's Table A row uses a derived rank | No manifest with DINO's true per-source Shard 1 rank was found locally; rank was derived from a reconstructed ASpanFormer match-count proxy per explicit 2026-07-16 sign-off. R@10+ unaffected (membership-only); R@1/R@5 approximate | `_local/ablation_stage1_report.html` §2's warn-finding callout |
| Stage 3 report itself (`_local/preliminary_ablations_report.html`'s content) | Old pose-scoring tables need full regeneration from `aggregate_significance.py`'s output, same treatment Stage 1 just got | Flagged in that file's own superseded-notice |

## 4. Pointers

- Full technical methodology: `ablation/STATISTICS_METHODOLOGY.md`
- Whole-strategy-report alignment checklist (what's done/deferred/blocked, and why, across
  every §6-§8 item, not just this dossier's own incident list): `ablation/STRATEGY_REPORT_CHECKLIST.md`
- Canonical Stage 1 report: `_local/ablation_stage1_report.html`
- Canonical Stage 2 report: `_local/ablation_stage2_report.html`
- Canonical Stage 3 report: `_local/ablation_stage3_report.html`
- Superseded/placeholder report: `_local/preliminary_ablations_report.html`
- MASt3R (B10) design rationale: `ablation/MAST3R_ABLATION_METHODOLOGY.md`
- RoMa (C2) hold status: `ablation/ROMA_ABLATION_METHODOLOGY.md`
