# NEG-Net Tier-0 — Training Process Reference

Written 2026-07-20 while the first real Tier-0 run (`ablation/negnet_tier0_shard1_colab.ipynb`,
raw-DINOv3, Shard 1) was being tested, so there's a single reference for what's running, why,
and whose call each design point was. Every claim below traces to code (cited `file:line`),
the git history, or a memory record — nothing here is invented. Where attribution is genuinely
uncertain, that's stated explicitly rather than guessed.

## 1. Purpose

The existing pipeline (`retrieve.py` → `geometry_filter.py` → `vggt_signals.py` →
`pose_scoring.py`) scores every candidate image pair **independently** — retrieve, verify
geometry, threshold two numbers, Positive/Negative. Deqian's formulation note reframes this as
recovering two latent, nested equivalence relations over the whole corpus — `N` (same
photographic exposure) and `S` (same scene, with `N ≤ S`) — from noisy pairwise evidence. That
reframing is what lets a model exploit structure (transitivity, nesting) the pointwise approach
cannot see by construction. NEG-Net is the learned decision layer built on top of the frozen
4-stage pipeline to do this. Tier-0 is explicitly the **fallback/baseline tier** of that model,
not the note's actual target method (§9).

Source: `negnet.py:1` docstring ("the learned decision model for the NCR-Match graph, note
§5.2"); memory `project_negnet_formulation`.

## 2. I/O contract

**Inputs to a training run** (`train_negnet.py`):
- `--graph` — a per-shard graph JSONL from `graph_assembly.py`: `{"type":"node", "node_id",
  "side", "path"}` and `{"type":"edge", "source_id", "target_id", "origin", "missing_evidence",
  "evidence", "label"}` rows.
- `--folds` — a JSON fold assignment (`graph_assembly.py folds`), family-masked so no
  Positive-connected component straddles train/val.
- `--features` — one or more `.npz` node-feature caches (`source_paths`/`target_paths`/
  `source_features`/`target_features`, same layout `retrieve.py` itself writes).
- Optional `--eval-graph`/`--eval-features` — a second, held-out shard's graph for frozen
  cross-shard evaluation.

**Outputs per run**, written to `--save-dir`:
- `negnet_best.pt` — model weights at the epoch with the best validation F1.
- `standardizer.json` — the frozen per-dim edge-feature mean/std (fit on the training fold
  only — see §6).
- `run_config.json` — every CLI arg used, plus the measured training positive rate and best
  validation metrics. This is what makes a saved checkpoint self-describing later.

Source: `train_negnet.py:178-269`.

## 3. Edge feature vector

Built by `extract_edge_features()` from the frozen evidence dict `g_ij` (exactly
`vggt_signals.py`'s output fields, copied verbatim by `graph_assembly.py`'s `EVIDENCE_KEYS`).

| # | Feature | Transform |
|---|---|---|
| 1 | `aspan_2d_raw_match_count` | log1p |
| 2 | `aspan_2d_homography_inliers` | log1p |
| 3 | `aspan_2d_inlier_ratio` | identity |
| 4 | `aspan_2d_mean_reprojection_error` | log1p |
| 5 | `aspan_2d_median_reprojection_error` | log1p |
| 6 | `aspan_2d_aligned_overlap_fraction` | identity |
| 7 | `alignment_keypoint_count` | log1p |
| 8 | `global_similarity` | identity |
| 9 | `pose_rotation_deg` | identity |
| 10 | `pose_translation_xy_l2` | identity |
| 11 | `pose_translation_z_abs` | identity |
| 12 | `pose_fov_l2` | identity |
| 13 | `pose_zoom_depth_fraction` | identity |
| 14 | `pose_component_score` | identity |
| 15-18 | `pose_component_terms.{rotation,translation_xy,translation_z,fov}` | identity |

Plus `alignment_homography` (3×3), normalized so `H[2,2]=1`, first 8 entries kept (9th is
redundant post-normalization) → **8 more raw dims**.

`EDGE_DIM = 18 + 8 = 26` raw values. Every dim also gets a missing-value bit (a "closure" edge
added purely by label transitivity has no evidence at all — every bit is missing, not
imputed-and-hidden). Final model input:
**`EDGE_INPUT_DIM = 2×26 + 1 = 53`** (values + missing-indicators + one edge-level
all-missing flag), verified by directly running `python negnet.py` (prints
`edge_input_dim=53`). Standardization: mean/std fit on the training fold's present values only,
frozen, applied everywhere else; missing entries become exactly 0 post-standardization (=
training mean, no signal), never fit on val/eval data.

Source: `negnet.py:38-132`.

## 4. Node features

LayerNorm + linear projection to the hidden width (`negnet.py:149-150`). The actual feature
**source** has a real discrepancy worth knowing about:

- **The note's stated default** (per a 2026-07-19 collaborator correspondence, secondhand —
  see §9) is the cheap, already-cached **1000-d post-fusion-head** embedding — zero new
  compute, `retrieve.py`'s existing cache.
- **The shipped code never implemented that path.** `dump_prehead_features.py` instead dumps
  the **3024-d pre-fusion-head concat** (1024 DINOv3-L + 1000 EfficientNetV2 + 1000 Swin), and
  `train_negnet.py` required a cache from that script specifically, with no fallback. This was
  originally guessed to be a deliberate choice (decoupling from the fusion-head ablation) — that
  guess was later confirmed wrong by the correspondence; it's simply the only path that got
  built.
- **For this run specifically (raw-DINOv3)**, the discrepancy is moot: `ModelComboDINOv3Raw`
  has no fusion head at all, so "pre-head" and "post-head" are the same 1024-d output —
  `retrieve.py`'s own `feature_cache.npz` is used directly, no dump script needed (the user's
  correction, this session — see §9). `NODE_DIM` is now inferred at load time from the actual
  cache shape (`train_negnet.py`'s `infer_node_dim()`) instead of a hardcoded `3024`, so this
  works without a code branch per backbone.

Source: `negnet.py:9,145-150`; memory `project_negnet_formulation` ("§0.2… Discrepancy surfaced
2026-07-19"); `train_negnet.py` (this session's edit, `infer_node_dim`).

## 5. Architecture

`NEGNet` (`negnet.py:144-184`):
1. **Node encoder**: LayerNorm(`node_dim`) → Linear(`node_dim → hidden`).
2. **Edge encoder**: 2-layer MLP, `edge_input_dim(53) → hidden → hidden`. At `--mp-rounds 0`
   this MLP's output goes **straight to the readout** — i.e. this alone reproduces the
   "pair-MLP" baseline row of the attribution ladder.
3. **Message passing**, `L` rounds (default 3): each round, edges update via a residual MLP
   over `[e, h_src, h_dst]`; nodes update via a residual MLP over `[h, mean-aggregated incident
   edge messages]`.
4. **Readout**: MLP over `[e, h_src, h_dst]` post-message-passing.
5. **Two linear heads**: `theta_S` (same scene) and `theta_N` (same exposure). The `N ≤ S`
   nesting constraint is enforced by the loss (§6), not the architecture.

Verified parameter counts (`python negnet.py`, re-run 2026-07-20 while writing this doc):
`mp_rounds=0` → **1,057,442** params; `mp_rounds=3` → **2,042,018** params.

## 6. Loss (`tier0_loss`, `losses.py:71-87`)

Four terms, summed:
1. **`bce_n`** — BCE on `theta_N` over labeled edges, weighted by `pos_weight` (class
   imbalance correction from the training fold).
2. **`bce_s`** — BCE on `theta_S`, same mechanism, but over `mask_s`. **`mask_s` currently
   equals `mask_n`** — a documented weak-label placeholder (`train_negnet.py:109-114`'s own
   comment): pre-audit, a Negative label conflates "different scene" with "same scene, different
   shot," so scene supervision trains on the same positive/negative split as exposure
   supervision until a "negative audit" (re-review of the Negative bin) lands. Not blocking for
   this run — it just means `theta_S` isn't yet learning what it's ultimately meant to.
3. **`triangle`** (`losses.py:43-56`) — for each 3-cycle of edges, penalizes
   `relu(sg[p_ik·p_kj] − p_ij)²`. The premise product is gradient-**detached**, so the model can
   only fix a violation by raising the weak edge, not by dragging the confident ones down.
   Normalized over active (violated) constraints only.
4. **`nesting`** (`losses.py:59-68`) — `relu(p_N − sg[p_S])²`, same detached-premise /
   active-only pattern, enforcing `N ≤ S`.

`--lambda-tri`/`--lambda-nest` weight terms 3-4 (default 1.0 each) — this is exactly the third
rung of the attribution ladder.

For cross-shard evaluation, `prior_shift_logits()` (`losses.py:99-112`) applies a standard
log-odds correction for the gap between training and deployment positive rates, before any
threshold.

## 7. Training procedure (`train_negnet.py`)

- **Splits**: family-masked K-fold (`graph_assembly.py folds`) — whole Positive-connected
  components go to one fold, so no leakage across train/val within a shard.
- **Standardization**: fit on the training fold only, frozen, saved alongside the checkpoint.
- **Optimizer**: Adam, `lr=1e-3`, `weight_decay=1e-4`.
- **Batching**: full-batch — the whole shard graph in one forward/backward pass (small enough:
  hundreds to low-thousands of edges).
- **Epochs**: 300 default; evaluated every 20 epochs; checkpoints on best validation F1.
- **Seed**: `0` default (`torch.manual_seed`/`np.random.seed`).
- **Attribution ladder** — exactly 3 configs, per the docstring:

  | Config | Flags | What it isolates |
  |---|---|---|
  | `mp0` | `--mp-rounds 0` | pair-MLP baseline |
  | `mp3` | `--mp-rounds 3` (default) | + message passing |
  | `mp3_noloss` | `--mp-rounds 3 --lambda-tri 0 --lambda-nest 0` | + message passing, consistency losses ablated |

## 8. This specific run's setup

- **Backbone**: raw DINOv3 (`ModelComboDINOv3Raw`, no fusion head, 1024-d) for both Stage 1
  retrieval and NEG-Net's node features — the canonical Stage 1 backbone switch, bundled into
  this run.
- **Scope**: Shard 1 only (Shard 2 raw-DINOv3 run shelved, no held-out cross-shard eval yet —
  in-shard folds only).
- **Train-once, freeze-forever**: this run is intended to be the only one. All 3 attribution-
  ladder configs are trained and frozen permanently (not winner-take-all) — future NEG-Net use
  is meant to reuse these weights, not retrain.
- **Two smoke-test gates** (distinct purposes):
  - *Step 0.5* — the entire pipeline (retrieval → geometry → VGGT → pose → graph assembly →
    training), run fresh on ~5 fixed fake-subset image pairs, before Stage 1 touches the real
    corpus at all. Catches plumbing/shape bugs cheaply.
  - *Step 6.5* — a short, disposable training pass (30 epochs, `mp_rounds=3`, discarded, not
    synced to Drive) on the **real** Shard-1 graph/features, right before the frozen 3-config
    run. Catches issues specific to the real data's actual scale/class-balance/missingness that
    the tiny fake-image test can't reach.
- **Leakage safety**: enforced by `NCR/negnet_training_exposure/` — every node this run's
  training graph touches gets recorded in a permanent registry; any future evaluation graph
  (a Shard 2 run, etc.) must check itself against it and drop conflicts before evaluating. This
  is independent of shard identity entirely (an earlier family-reassignment approach to this
  problem was tried and fully superseded/removed — see `NCR/negnet_training_exposure/README.md`
  for why record-and-drop replaced proactive resharding).

Source: this session's plan file and notebook.

## 9. Decision attribution

| Decision | Attributed to | Note |
|---|---|---|
| Overall reframe (latent `N`/`S` equivalence relations, `Y(V)`, need for group consistency) | **Deqian's formulation note** | `project_negnet_formulation` memory; repeated "Provenance: Deqian's correspondence" citations across `NEGNET_REFRAME_CHECKLIST.md` |
| Tier-0's qualitative shape (edge-features→MLP→message-passing→two heads; triangle+nesting terms; prior-shift) | **Deqian's note**, implemented by **an earlier Claude Code session** | Commit `b2e4481`, 2026-07-18 — verified via `git log --format=fuller`: Author/Committer both `Claude <noreply@anthropic.com>`, a `Claude-Session` trailer with a different session ID than this conversation, single non-merge commit on `origin/master` |
| Architecture confirmed to match shipped code (~2M params, edge encoder + ~3 MP rounds + θ_N/θ_S heads) | **Deqian, via a collaborator's correspondence** relaying "Deqian's + a meeting's conclusions" | Memory's own caveat: "not Deqian's own words verbatim, treat as high-confidence secondhand, not primary source" |
| Specific numeric hyperparameters (`hidden=256`, `mp_rounds=3`, `lr=1e-3`, `weight_decay=1e-4`, `epochs=300`, both λ=1.0, `seed=0`) | **The implementing Claude session's own choice** | The same correspondence explicitly states: *"Layer widths aren't pinned by the note — an experiment-plan choice, Deqian to sign off, not a formulation constraint."* No record found that sign-off actually happened. Unvalidated — no sweep/ablation artifact exists for these values anywhere in the repo. |
| Running Tier-0 (BCE+triangle) first, before the note's actual default (Fenchel-Young loss + nested correlation-clustering decoder) | **A collaborator's own suggestion**, explicitly not Deqian's spec | Correspondence: *"mine, not the note's"* — plausible, consistent with what's built, but not literally requested |
| Node-feature default: cheap post-head vs. shipped pre-head concat | **Unreconciled discrepancy** | Note's stated default is the cheap path; shipped code only built the pre-head path. Flagged in memory as "worth a direct question to Deqian rather than an assumption" — not resolved |
| Shard 1/2 family-disjoint reassignment (core protocol: node-embedding models need family-, not edge-, level shard disjointness) | **Deqian's correspondence** (core protocol); **the user** (freeze-vs-drop application, project-wide scope, timing) | `project_negnet_shard_crossover` memory, `NEGNET_REFRAME_CHECKLIST.md` §0.1's own Provenance line |
| Bipartite `V = V_A ⊔ V_M` assumption is obsolete (targets ⊂ sources) | **The user**, confirmed with their professor | `project_negnet_formulation` memory |
| Ternary output requiring a "negative audit" (splitting same-scene-different-shot out of the Negative bin) | **Deqian's correspondence**, 2026-07-19 | Code already anticipated the seam (`train_negnet.py`'s `label_s = label_n.copy()` placeholder, pre-dates the correspondence) — "the correspondence supplies the motivation/scope, the code already has the seam" |
| Raw-DINOv3's own retrieval-time output used directly as the node feature (no separate dump script) | **The user**, this session | Direct correction: *"Since we're using rawDINO, we can simply just use the outputted features of each image directly instead of dumping an intermediate layer"* |
| `NODE_DIM` inferred dynamically instead of hardcoded | **The user's decision** this session, implemented by **this session** | AskUserQuestion answer: "Infer dynamically (Recommended)" |
| Ground-truth-wide (not graph-scoped) shard-crossover audit *(historical — this whole approach was later fully superseded by node-exposure-based leakage prevention; see `NCR/negnet_training_exposure/README.md`, §8 above)* | **The user's** mid-turn correction, implemented by **this session** | *"the shard check should be done over the entire ground truth, not just any one model's surfaced pairs"* |
| Train-once/freeze-forever; train all 3 attribution-ladder configs, not winner-take-all | **The user**, this session | *"this should only be run once and the weights frozen... these weights should work for all use cases of NEG-Net"*; *"Let's train multiple configs, but save them all and freeze them with the config labeled"* |
| Real-data smoke test (Step 6.5) before the frozen run | **The user**, this session | Explicit request while approving the notebook |
| Same-side magazine-magazine pairs carry full evidence (not uniformly evidence-free) | **Not pinned down** | Presented in memory as an amendment to the note's original text; no record of whether Deqian or the user first flagged it |

## 10. Open questions / caveats

- **Hyperparameters are unvalidated** (§9) — no sweep exists for `hidden`/`mp_rounds`/the
  lambdas; `lr`/`weight_decay` read as standard Adam defaults, not tuned for this data.
- **`theta_S` (scene) supervision is a weak-label placeholder** until the negative audit
  separates "same scene, different exposure" from "truly unrelated" in the current Negative
  bin — not blocking this run, but means the scene head isn't learning its real target yet.
- **Tier-0 is explicitly the fallback tier**, not the note's default method. The actual default
  — a perturbed Fenchel-Young loss decoded via nested correlation clustering onto `Y(V)`
  directly — has no implementation (`losses.py`'s `fy_loss()` raises `NotImplementedError`;
  `nested_cc.py` doesn't exist). There is currently no decoder that *guarantees* output
  satisfies the equivalence-relation constraints at inference — consistency is only encouraged
  during training via soft penalties.
- **Node-feature default discrepancy (§4)** is unresolved for the general case (though moot for
  this specific raw-DINOv3 run, which has no head to strip).
- **Deqian's stated fallback if Tier-0 underdelivers**: "a rigorous linear regression over the
  same evidence features" — materially the same thing as this project's existing B11 ablation
  (a learned classifier over the flat evidence vector, no graph). Not evaluated here.
