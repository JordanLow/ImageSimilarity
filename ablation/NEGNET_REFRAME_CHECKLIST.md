# NEG-Net Reframe Checklist — what's done, what's needed, what adapts

Written 2026-07-19, on explicit user request, after Deqian's formulation note (NCR-Match as
latent equivalence-relation recovery) and the resulting NEG-Net Tier-0 scaffolding
(commit `b2e4481`) landed. Companion to `STRATEGY_REPORT_CHECKLIST.md` (tracks the strategy
report's own asks — Table A/B/C, §6.x) and `STATISTICS_METHODOLOGY.md` (tracks how the
statistics work) — this document tracks the *reframe*: which of this project's existing
ablation/experiment work needs to change, be re-reported, or is newly required, now that the
project has a formal graph-structured decision layer on top of the frozen pipeline. Full
formulation context: see the Claude memory notes `project_negnet_formulation`,
`project_negnet_shard_crossover`, `project_negnet_ablation_plan` (not duplicated here in
full — this file is the actionable checklist, those are the reference explanations).

**Per-item fields, all requested explicitly by the user**: Overview, Progress, Effort,
Justification (is it required / what does it block), Provenance (whose call this was — my own
judgment, the strategy report, something the user told me to remember, standard ML/stats
practice, or Deqian's correspondence).

**Status legend**: ✅ Done · 🟡 Partial · ⛔ Not started · 🔒 Blocked (unblocking condition
stated) · ⏸ Deferred/Priority (not abandoned) · 🚫 Deliberately out of scope.

---

## 0. Blocking prerequisites — nothing else in this checklist is trustworthy without these

### 0.1 Leakage prevention (closed — approach changed, not just completed)
- **Status**: ✅ Closed, 2026-07-20. NEG-Net leakage prevention is now node-exposure-based, not
  shard-reassignment-based — see `NCR/negnet_training_exposure/README.md` for the current
  mechanism (record every node a training graph touches; a later evaluation graph checks
  itself against that registry and drops conflicts). An earlier family-level resharding
  approach was built, used, and then fully superseded and removed once this simpler mechanism
  was adopted — no trace of it remains in `NCR/`. Nothing here is blocking; this item is done.

### 0.2 Run `dump_prehead_features.py`
- **Status**: ⛔ Not started — no cache file exists yet.
- **Overview**: One-pass GPU dump of the 3024-d pre-fusion-head backbone concatenation, the
  node feature NEG-Net actually trains on.
- **Effort**: Low mechanically (script exists, runs once), but real wall-clock GPU cost — one
  forward pass per image across the corpus (~42,773 images per the formulation note's own
  count).
- **Justification**: **Required, hard-blocking.** Nothing in Section 2 below can train without
  this.
- **Provenance**: Formulation note §5.1 + `dump_prehead_features.py`'s own design (already
  built, this item is just "run it").
- **Open question, not yet decided (surfaced 2026-07-19 alongside 0.3 below)**: whether this
  should keep dumping the full 3-backbone concat regardless of Stage 1's canonical retrieval
  model, or narrow to raw-DINOv3-only once 0.3 actually lands. Not resolved now — these are
  different objects (retrieval candidate generation vs. NEG-Net node features) that happen to
  both involve DINOv3.
- **Discrepancy surfaced 2026-07-19 (separate correspondence, Q1 architecture answer)**: per
  Deqian's collaborator, the note's actual DEFAULT node feature is the cheap 1000-d post-head
  embedding (`retrieve.py`'s existing cache, zero new compute) — the 3024-d pre-head concat
  this script produces is described as an *optional* export, not the default. The shipped code
  has no path for the cheap default at all (`negnet.py` hardcodes `NODE_DIM = 3024`,
  `train_negnet.py` requires this script's cache with no fallback) — so this item stays the
  only working path today regardless of which one the note calls "default." Full detail:
  [[project_negnet_formulation]]'s Q1-answer section. Worth a direct question to Deqian rather
  than resolved here.

### 0.3 Canonical Stage 1 backbone switch — DINOv3-raw supersedes DINO-with-trained-head
- **Status**: 📝 Decided, deferred. Not blocking 0.1/0.2/Section 2 — explicitly bundled into
  this same NEG-Net arc rather than actioned immediately.
- **Overview**: Going forward, raw DINOv3 (no ensemble, no learned fusion head) is the
  canonical Stage 1 retrieval backbone, superseding `ModelComboDINO` (the 3-backbone
  ensemble-with-trained-head used for every prior ablation this project has run). Full detail:
  [[project_canonical_stage1_backbone_switch]].
- **Effort**: Not estimated — explicitly deferred until "an entire pass over this again" happens
  as part of this arc, not scoped as standalone work right now.
- **Justification**: Not required for 0.1/0.2 or Section 2 to proceed. Required eventually for
  terminology/baseline consistency — every existing B4-relative number
  (B1/B2/B5/B6/B8/B10/B11, LightGlue, RoMa, MASt3R) was computed against candidates the OLD
  canonical backbone generated, and becomes a legacy/historical reference point once the
  switch actually happens, not the current baseline.
- **Provenance**: User decision, 2026-07-19, explicit: "I just need you to remember this and
  not cross wires with the previous experiments" — a remember-now-act-later instruction, not
  a request to update anything today.

---

## 1. Existing per-stage ablation work — reframing status

### 1.1 Stage 1 (Retrieval) ablations — SSCD, CLIP, DINOv3-raw, JOCCH
- **Status**: 🟡 Already in the right currency. Reported via Recall@K/mAP
  (`eval_retrieval.py`, see `project_stage1_ablation_report_canonical`) — this is already
  candidate-generation currency, not final-decision currency.
- **Overview**: No conceptual change needed — Stage 1 swaps were already measuring `R`'s
  quality directly, which is exactly what the reframe asks for.
- **Effort**: Low — mostly a documentation/cross-reference task (note explicitly in the Stage
  1 report that this already satisfies the reframe), not new computation. JOCCH row itself is
  still pending for unrelated reasons (see `project_jocch_training_pipeline`, pinned).
- **Justification**: Not blocking anything; a consistency/documentation nicety.
- **Provenance**: My judgment, cross-checking `project_negnet_ablation_plan` §C against what
  the Stage 1 report already does.

### 1.2 Stage 2 (Keypoint Matching) ablations — LightGlue (B14), RoMa (on hold)
- **Status**: 🟡 Underlying results done/partial (LightGlue Shard 1: F1 0.910 vs B4 0.913,
  n.s.; Shard 2 pending; RoMa on hold pending feasibility). Currently reported via
  pose_scoring-derived final P/R/F1 (matcher-vs-matcher under the B4 decision rule) — the
  reframe asks for this to ALSO be presented via funnel-survivor stats.
- **Overview**: Add a candidate-generation-currency view (how many candidates survive Stage 2,
  independent of the final threshold decision) alongside — not instead of — the existing
  McNemar-vs-B4 comparison, which remains a valid comparison in its own right.
- **Effort**: Medium — needs new analysis code (survivor counts per stage from already-computed
  manifests), but no new GPU/matcher compute for rows already run.
- **Justification**: Not blocking NEG-Net training. Needed for a *complete*, formulation-
  aligned write-up, not for the pipeline to function. Real open question, not yet decided:
  whether Stage 3 needs the same treatment (see 1.3) — worth resolving both at once rather
  than piecemeal.
- **Provenance**: Deqian's correspondence, §C of `project_negnet_ablation_plan`. Also flagged
  in that same correspondence: `geometry_filter_roma.py` already exists — explicit ask to
  either include RoMa or write down why LightGlue is the representative Stage 2 alternative.

### 1.3 Stage 3 (Inlier + Pose Scoring) ablations — B1/B2/B5/B6/B8/B10/B11
- **Status**: 🟡 Underlying results done (`project_stage3_ablation_report_canonical` — only
  B2/B6 significant vs B4 post-Holm; B9/B13 not built). Same currency question as Stage 2,
  but fuzzier: Stage 3 IS close to the final decision itself (it produces the pose signal the
  decision rule thresholds), so a funnel-survivor-only view may be less meaningful here than
  for Stage 2.
- **Overview**: Whether/how to add a candidate-generation-currency view for Stage 3 specifically
  is a genuinely open judgment call, not a settled requirement.
- **Effort**: Medium if pursued, same shape as 1.2.
- **Justification**: Not blocking anything. **Explicitly flagged here as needing a decision
  from you (or Deqian) rather than me assuming** — the correspondence didn't distinguish
  Stage 2 from Stage 3 on this point, and I think the distinction matters.
- **Provenance**: My judgment (the nuance/open question), building on Deqian's correspondence
  (the general reframe ask).

### 1.4 CPU-bound alternative-compute rows — B8 (classical pose), B11 (learned classifier)
- **Status**: ✅ Done (`project_stage3_ablation_report_canonical`) — B8 zero-learned-parameter
  classical homography decomposition, B11 sklearn LR/MLP over 17 stored signals, both
  evaluated vs B4.
- **Overview**: These are Stage-3-internal alternative decision rules (swap the *hand rule*
  for a classical or shallow-learned one), not swaps of the evidence-generating models — they
  sit conceptually adjacent to NEG-Net (also "a learned decision rule over the same evidence")
  rather than to the Stage 1/2/3 pipeline-swap category.
- **Effort**: N/A — already done. Worth a light cross-reference note in the eventual NEG-Net
  write-up (B11 in particular is a natural point of comparison — a much simpler learned model
  over the same evidence family NEG-Net uses, minus the graph structure).
- **Justification**: Not blocking. A framing/cross-reference opportunity, not new work.
- **Provenance**: My judgment — the correspondence doesn't mention B8/B11 directly, but the
  attribution ladder (2.1 below) already effectively re-poses this exact question
  (pair-MLP vs +message-passing) in the new framework, making B8/B11 relevant prior art to
  cite, not redundant work.

### 1.5 Miscellaneous supporting work — sensitivity curves, survival audit, pose-signal
    analysis, reversed-pair audit, B10/MASt3R sync, reproducibility dossier
- **Status**: ✅ Done, various dates (see `STRATEGY_REPORT_CHECKLIST.md`,
  `REPRODUCIBILITY_STATUS.md` for the individual items).
- **Overview**: General-purpose infrastructure/audits, not stage-swap ablations — mostly
  unaffected by the reframe as *work*, but two of them turn out to be formally load-bearing
  under the new formulation specifically (see Section 4 below) rather than just supplementary
  evidence.
- **Effort**: N/A — already done.
- **Justification**: Not blocking. No action needed here beyond the cross-references noted in
  Section 4.
- **Provenance**: My judgment, reviewing what already exists against the new formulation's
  estimands.

---

## 2. NEG-Net must-have ablations (Deqian's correspondence, section A)

### 2.1 Attribution ladder — B4 → pair-MLP → +message-passing → +consistency losses
- **Status**: 🔒 Blocked on §0 (needs clean shards + node features). Code-ready: CLI flags
  (`--mp-rounds`, `--lambda-tri`/`--lambda-nest`) already wired in `train_negnet.py`.
- **Overview**: The main table for the new mechanism — isolates whether gains come from graph
  structure or just more parameters.
- **Effort**: Low once §0 clears — this is "run 4 configs," no new code.
- **Justification**: **Required.** This is the central empirical claim for NEG-Net's existence.
- **Provenance**: Deqian's correspondence, explicitly "already wired," item A1.

### 2.2 Prior-shift on/off comparison on Shard 2
- **Status**: 🔒 Blocked on §0. Code-ready: `losses.py`'s `prior_shift_logits` exists.
- **Overview**: Show calibration/decision metrics measurably degrade without the correction,
  empirically justifying a design choice currently just asserted.
- **Effort**: Low once §0 clears.
- **Justification**: Required for rigor (justifies a real design choice with data), not
  blocking anything else.
- **Provenance**: Deqian's correspondence, item A2.

### 2.3 Label-efficiency curve (25/50/75/100% of labeled families)
- **Status**: ⛔ Not started — needs the Q2 partition-eval module (see 4.3), which doesn't
  exist yet, in addition to §0.
- **Overview**: Train on increasing fractions of labeled *families* (not edges), plot F1 — the
  "works with few expert labels" claim.
- **Effort**: Medium-high — real new code (family-level subsampling + the eval module it
  depends on), not a config toggle.
- **Justification**: Required per the correspondence; also matters for the humanities framing
  of the project (expert-labeling cost is a real constraint).
- **Provenance**: Deqian's correspondence, item A3, explicitly tied to "the note's eval module
  spec."

### 2.4 Statistics.py hookup — NEG-Net vs B4, same McNemar/Holm machinery
- **Status**: ⛔ Not started (a near-identical pattern was built and then deliberately reverted
  earlier this session for a different, premature purpose — see `project_stage2_batching`'s
  history — the *pattern* is proven reusable, not wasted, even though that specific script was
  reverted).
- **Overview**: Extend `ablation/significance.py` + `ablation/aggregate_significance.py` to
  compare NEG-Net vs B4 on the same pair set, own Holm family (not pooled with the existing
  `stage2`/`stage3` families — this is a different kind of comparison, learned-decision-layer
  vs hand-rule).
- **Effort**: Low-medium — the canonical machinery already exists and is well-proven; this is
  "write one more evaluation script following the `eval_stage2.py` pattern."
- **Justification**: Required — this project's whole statistical-rigor standard
  (`STATISTICS_METHODOLOGY.md`) applies here as much as anywhere else; nothing about NEG-Net
  should get a lower evidentiary bar than B1-B14/MASt3R did.
- **Provenance**: Deqian's correspondence, item A4, explicitly naming the existing machinery.

---

## 3. NEG-Net nice-to-have ablations (Deqian's correspondence, section B)

### 3.1 Evidence-group ablations (counts-only / +inlier / +pose / +global-sim / +homography)
- **Status**: 🔒 Blocked on §0. **Effort**: Low-medium, mechanical feature-group dropping.
- **Justification**: Nice-to-have — empirically answers a question ("should inlier ratio be
  an input") this project previously only settled by argument. Not blocking.
- **Provenance**: Deqian's correspondence, item B5.

### 3.2 Node-feature variants (3024-d pre-head vs 1000-d post-head vs raw DINOv3)
- **Status**: 🔒 Blocked on §0.2 only. **Effort**: Low once the prehead dump exists — swap
  which cache loads.
- **Justification**: Nice-to-have — closes the loop with this project's existing fusion-head/
  DINOv3-raw ablation work. Not blocking.
- **Provenance**: Deqian's correspondence, item B6.

### 3.3 Closure edges (E_aug) on/off; single-head (N) vs dual-head (N+S)
- **Status**: 🔒 Blocked on §0. **Effort**: Low-medium, architecture/data toggles.
- **Justification**: Nice-to-have — ablates the graph-structure-specific design choices
  themselves. Not blocking.
- **Provenance**: Deqian's correspondence, item B7.

---

## 4. Target-method gaps — named in the formulation note, not in the Tier-0 commit

### 4.1 `nested_cc.py` — the Fenchel-Young loss's decoder
- **Status**: ⛔ Not started. `losses.py`'s `fy_loss()` literally raises `NotImplementedError`
  pending this.
- **Overview**: Greedy correlation-clustering + Kernighan-Lin local moves, scene layer then
  exposure layer within scene clusters — the note's actual *default* method (§3), not an
  optional extra. Tier-0 (BCE + soft hinges, already built) is explicitly the fallback.
- **Effort**: High — a genuine new algorithm implementation (structured decoder over `Y(V)`),
  not a config change or a thin wrapper around existing code.
- **Justification**: Required *if* the project intends to claim the note's default method, not
  just the Tier-0 fallback. Not blocking Section 2's must-have ablations (those run fine on
  Tier-0). Worth an explicit decision: is Tier-0 the shipped method for this cycle, with FY/CC
  as future work, or is this required before NEG-Net results are reported at all?
- **Provenance**: Formulation note §3 (the object itself) + my judgment (the scoping question,
  not yet posed to or answered by Deqian in what's been shared with me).

### 4.2 Conformal accept/review/reject decision layer
- **Status**: ⛔ Not started.
- **Overview**: The actual risk-controlled deployment policy (formulation note §3, eq. 7) —
  three-way decision rule with a one-sided FNR guarantee via conformal thresholds, after the
  prior-shift correction (2.2) is already applied.
- **Effort**: Medium-high — conformal calibration is a well-scoped, standard technique, but a
  real implementation task, not existing anywhere in this repo yet.
- **Justification**: Required for the note's full deployment story (this is literally what
  "risk-controlled deployment layer" in the note's own scope line means) — but not blocking
  Section 2's must-have ablations, which evaluate raw probabilities/F1, not the deployed
  decision policy.
- **Provenance**: Formulation note §3, table in §5.2 ("conformal script (in repo) + wiring" —
  named as a target, not present).

### 4.3 Q2 partition-eval module (B³ metric, pairwise P/R vs exposure families)
- **Status**: ⛔ Not started. Blocks 2.3 (label-efficiency curve) directly.
- **Overview**: Evaluates the *partition* NEG-Net recovers (which nodes cluster into which
  exposure/scene groups) against ground truth, not just per-edge accuracy — the note's Q2
  estimand.
- **Effort**: Medium — B³ (bcubed precision/recall/F1) is a standard, well-defined clustering
  metric with known implementations to reference; the family-level ground truth already
  exists implicitly via the labeled edges' transitive closure (same machinery `graph_assembly.py`
  already uses for the shard audit).
- **Justification**: Required — directly blocks 2.3, and is the only way to evaluate the part
  of NEG-Net's output (Q2/Q3) that pointwise F1 structurally can't measure.
- **Provenance**: Formulation note §5.2 table ("partition eval module") + Deqian's
  correspondence naming it as part of item A3's scope.

---

### 4.4 Negative-bin audit — splitting same-scene-different-shot from truly-unrelated
- **Status**: ⛔ Not started as a tool/process; already anticipated as a placeholder in shipped
  code. `train_negnet.py:105-113` sets `self.label_s = self.label_n.copy()` with an explicit
  comment: "pre-audit, Positive ⇒ same scene; Negative is only a scene-negative once the audit
  separates S\N from U... until the audit lands, scene BCE trains on positives vs negatives
  as-is (weak labels)."
- **Overview**: Re-review the current Negative-labeled bin in `match_manifest*.csv` to (a)
  split same-scene-different-shot pairs (the "man steps forward from the cannon" case) out
  from truly-unrelated pairs, producing real `S` (same-scene) labels instead of today's
  `label_s = label_n` placeholder, and (b) catch any missed positives along the way. This is
  what turns NEG-Net's output ternary (true-match / same-scene-different-shot / unrelated)
  instead of binary — `θ_N`/`θ_S` combine into exactly these three classes.
- **Effort**: Medium-high — human/manual re-review work, likely similar scale to the earlier
  VGGT/pose human-review passes done for other ablations (see
  [[project_dinov3raw_jocch_geometry_pipeline]]'s 3TP/3FP/92TN-scale review), not a script.
- **Justification**: Blocks real scene (`S`) supervision and the ternary output specifically —
  does NOT block a first Tier-0 run in binary/`N`-only mode, which trains fine today on the
  weak-label placeholder. Cross-references 3.3 (single- vs dual-head ablation) — that
  comparison isn't meaningful with real S-labels until this lands.
- **Provenance**: Deqian's correspondence (2026-07-19, Q1 architecture answer, "negative
  audit"/ternary-output language) — independently confirmed already anticipated in
  `train_negnet.py`'s own code comment, not a gap newly discovered here.

---

## 5. Formal connections to already-tracked strategy-report items (re-prioritization candidates)

### 5.1 §6.4 end-to-end survival audit ↔ Q3's "decision recall" factor
- **Status**: 🟡 exists (`_local/survival_audit.py`, 577/2,636 pairs, top-15 not top-10),
  previously routed to a future stage-less "overall" report per prior user decision.
- **Overview**: The formulation note's Q3 recall factorization (§4, eq. 8) names its second
  factor — `P(δ≠reject | ij∈E, N_ij=1)` — as "what the survival analysis measures (97.2% at
  present)." **This means the existing survival audit isn't just supplementary evidence
  anymore — it's formally one half of a named estimand in Deqian's framework.** Worth
  checking whether the note's cited 97.2% actually matches this repo's own survival-audit
  output, or is a different/independent number — not yet verified either way.
- **Effort**: Low to check the number match; the audit itself already exists (scope-expansion
  to full 2,636 pairs / top-10, if wanted, is separate, larger work already scoped in the
  original checklist.
- **Justification**: Not required to change anything immediately, but this elevates the
  existing "future overall report" item from "nice supplementary analysis" to "a named piece
  of the formal estimand" — worth flagging to Deqian/collaborators rather than leaving it
  purely deferred without comment.
- **Provenance**: My judgment, connecting the formulation note's own text (§4) to
  `STRATEGY_REPORT_CHECKLIST.md`'s existing §6.4 entry — not stated explicitly by Deqian in
  the correspondence shared with me.

### 5.2 §6.6/§6.7 human evidence + Jinchaji cross-domain ↔ Q3's "funnel coverage" factor
- **Status**: ⏸ Deferred/Priority (prior user decision, feasibility-driven).
- **Overview**: The note's Q3 factorization's FIRST factor — `P(ij∈E | N_ij=1)`, funnel
  coverage — is explicitly named as "unmeasurable from `E_lab` by construction" and "estimated
  externally: stratified relabeling of the rejected set, and the Jinchaji pre-model pairs as a
  zero-shot audit of R." **This is precisely §6.6 (human relabeling) and §6.7 (Jinchaji)** —
  the note gives these previously-deferred items a formal name and a precise role (the only
  available estimator for an otherwise structurally unmeasurable quantity), rather than being
  generically "nice human-evidence work."
- **Effort**: Unchanged from the original checklist entries (real logistics/scope, not
  re-estimated here).
- **Justification**: Doesn't change the feasibility constraints that led to deferring these —
  but does strengthen the case for revisiting them sooner rather than later, since they're now
  the *only* named path to estimating Q3's first factor at all, not one option among several.
- **Provenance**: My judgment, same connection-drawing as 5.1 — formally grounded in the
  note's §4 text, not something Deqian's correspondence stated as a re-prioritization
  explicitly.

---

## 6. Deliberately out of scope — do not reopen

### 6.1 Keypoint floor (`breakpoint_value`, default 50) as an ablation axis
- **Status**: 🚫 Deliberately out of scope.
- **Justification**: Per the formulation, it's part of the frozen candidate-generation
  function `R` — its cost is measured by the external reject-audit (5.2/6.6), not by
  resweeping the threshold. Reopening it would re-litigate "why 50" instead of closing it.
  The existing B1 sweep stays as a hand-baseline curve, unchanged.
- **Provenance**: Deqian's correspondence, item D, stated explicitly and firmly.

### 6.2 Retraining NEG-Net once per Stage 2/3 pipeline variant
- **Status**: 🚫 Scoped down — retrain for at most ONE alternative matcher, not all variants.
- **Justification**: Swapping Stage 2/3 models changes `g_ij`'s distribution (the
  `Standardizer` is fit to a specific evidence distribution), which would force a full
  NEG-Net retrain per variant — expensive relative to timeline. Keep pipeline swaps evaluated
  against the B4 hand baseline only (as already done all session); pick at most one
  alternative (likely LightGlue, given its results already exist) if a NEG-Net-side
  comparison is wanted at all.
- **Provenance**: Deqian's correspondence, item C2 — **explicitly flagged there as a scoping
  question awaiting confirmation, not yet a fully closed decision** — surface this again
  before committing engineering time to any variant retrain.

### 6.3 Stage 2 GPU batching
- **Status**: 🚫 Abandoned, unrelated to this reframe (see `project_stage2_batching`) — listed
  here only so it doesn't get confused with an item on this list. Not touched by, and not
  relevant to, the NEG-Net reframe.
- **Provenance**: User decision, 2026-07-18, orthogonal engineering-optimization effort.

---

## Suggested sequencing

1. §0 (shard audit + prehead dump) — nothing else is trustworthy without these.
2. §2 (must-have ablations) — the core NEG-Net results.
3. §4.3 (Q2 eval module) — unblocks 2.3, and is needed regardless for reporting Q2/Q3 at all.
4. §1.2/1.3 (Stage 2/3 reframe) and §3 (nice-to-haves) — in parallel, lower urgency.
5. §4.1/§4.2 (nested_cc.py, conformal layer) — only once a decision is made on whether Tier-0
   is this cycle's shipped method or these are required first (see 4.1's justification).
6. §5 items are re-prioritization flags, not new work queued on their own — resurface with
   Deqian/collaborators rather than scheduling unilaterally.
