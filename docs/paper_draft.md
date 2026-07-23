# Same Negative or Same Scene? Auditable Candidate Generation and Risk-Controlled Matching for Archival Image Search

*Paper draft v0.1, 2026-07-23. Master content draft; LaTeX (AAAI-27 kit) to be produced from this file. Every number below is traceable to a compiled report (cited inline as [S1] = dinoraw_baseline_stage1_report, [S2] = stage2, [S3] = stage3, [NN] = negnet_tier0_report, all 2026-07-23 versions). Placeholders and open decisions are marked ⚠TODO with an owner.*

**Authors**: Lin Du 杜琳, Jordan Low, Deqian Kong 孔德乾, Prabhu [全名/单位 ⚠TODO:Lin]

---

## Abstract

Which archival negative does a printed photograph reproduce? For historians, answering this across a large archive is foundational provenance work, and it is hard: reproductions are cropped, halftone-screened, and retouched, while archives are dense with near-duplicate exposures shot moments apart at the same scene. Mistaking one of those exposures for the printed one does not merely miss a match; it manufactures false evidence, and the same confusion underlies image reuse out of context today.

We formulate reproduction matching as estimation of a nested latent partition, in which images group into exposures and exposures group into scenes, observed only through a fixed candidate-generation operator and per-pair geometric evidence. The formulation makes discovery bias definitional rather than anecdotal: end-to-end recall factorizes into funnel coverage, estimated by stratified audits of rejected candidates, and decision recall, measured against expert adjudication.

We instantiate this on the North China Railway Archive (華北交通アーカイブ) and the magazine Hokushi (北支, 1939 to 1942), and release NCR-Match, an expert-adjudicated dataset spanning 42,773 archival photographs, 2,856 printed reproductions, and more than 1,200 candidate pairs labeled same exposure, same scene, or unrelated ⚠(label-taxonomy wording pending the negative audit; see §4.3). We propose NEG-Net, a relational model tailored to this structure: it scores candidate pairs jointly over the corpus rather than independently, and trains on only a few hundred expert adjudications. NEG-Net agrees with expert adjudication on [XX]% of held-out pairs, [+2] points over an otherwise identical model that scores pairs independently, and renders judgments on transitively-implied pairs that pointwise pipelines structurally cannot evaluate; its calibrated three-way output bounds the false-rejection rate at [α] while reducing pairs requiring expert review by [XX]%. Zero-shot transfer to the Sha Fei archive and Jinchaji Pictorial yields [XX].

*(TL;DR, ≤250 chars: see docs/abstract_draft_AAAI27.md; abstract claims must stay consistent with §6 — in particular no claim of beating the hand rule on matched evidence.)*

---

## 1. Introduction

Para 1 — the historian's problem (civic framing kept per 0722 review): provenance of printed reproductions; near-duplicate exposures; misattribution manufactures false evidence; same confusion underlies decontextualized image reuse today.

Para 2 — why pipelines are not enough: hand-tuned filter cascades work but cannot say *what* they estimate, *what* they miss, or *how sure* they are. Three gaps: (i) no explicit estimand; (ii) discovery bias unquantified; (iii) thresholds unjustifiable and corpus-specific.

Para 3 — our reframe: matching as recovery of a nested latent partition (exposures within scenes) over a fixed candidate graph; discovery bias becomes a defined, measured quantity; decisions become risk-controlled.

Para 4 — contributions (each maps to a section):
1. **Formulation** (§3): nested partition estimation with a fixed candidate-generation operator; end-to-end recall factorizes into funnel coverage × decision recall.
2. **NCR-Match dataset** (§4): 42,773 archival photographs × 2,856 reproductions, 1,200+ expert-adjudicated candidate pairs with family structure and leakage-controlled splits; released on Hugging Face.
3. **NEG-Net** (§5): a relational decision layer trained on a few hundred adjudications; message passing improves over pairwise scoring (Holm-corrected p = 0.0089) and extends judgment to transitively-implied pairs that pointwise rules structurally cannot evaluate (30/30 vs 0/30 on Shard 2).
4. **Honest accounting** (§6): on matched evidence the corpus-tuned hand rule remains the stronger pointwise judge (Holm-robust); we report this openly and locate the learned layer's value in coverage, label efficiency, calibration, and transfer ⚠(transfer result pending — JOCCH run decides the strength of this claim; owner: Jordan).

## 2. Related Work ⚠TODO:Jordan+Deqian

- Near-duplicate / copy detection (SSCD lineage); instance retrieval; self-supervised ViTs (DINO family).
- Local feature matching and geometric verification (ASpanFormer, LightGlue, RoMa lineage); pose estimation (VGGT lineage). *(Names appear here and in §6, not in the abstract, per 0722 review.)*
- Correlation clustering / structured prediction over graphs; transitivity constraints; Fenchel–Young losses.
- Conformal prediction and selective classification (accept/review/reject).
- Computational humanities: photo-archive scholarship, JOCCH-line work, image reuse and provenance studies.

## 3. Problem Formulation

Prose version of the formulation note (NCR-Match: Problem Formulation and Code Correspondence, v0.1):

- Corpus V = archival photographs ⊔ printed reproductions. Latent maps: every image reproduces exactly one **exposure**; every exposure belongs to one **scene**. This induces two nested equivalence relations (same-exposure N ⊆ same-scene S); transitivity and nesting are not modeling assumptions but the definition of the hypothesis space.
- **Observation model**: a fixed, given candidate-generation operator R (retrieval top-K + correspondence floor) produces candidate set E; each candidate pair carries a frozen geometric evidence vector g_ij (match counts, inlier ratio, reprojection statistics, overlap, homography, pose components, global similarity). R's quality enters the estimands, not the model.
- **Supervision**: expert adjudication on candidate pairs only; therefore every supervised quantity is conditional on E. **Discovery bias is definitional**: recall_e2e = P(pair ∈ E | true match) × P(accepted | ∈ E, true match). The first factor (funnel coverage) is estimated by stratified relabeling of rejected candidates and a pre-model zero-shot audit; the second (decision recall) against adjudication.
- **Three estimands, strictly ordered** (note §4; this ladder organizes §6): Q1 = pairwise relation on candidate pairs (directly supervised → §6.1–6.2); Q2 = the partition restricted to nodes the funnel touches (partially identified; grows with graph density → §6.2b); Q3 = matches over all archive×reproduction pairs (end-to-end; requires the funnel-coverage factor → §6.6).
- **Scene granularity** (note Remark 1, pre-empting the "scenes drift over long chains" objection): the scene relation is defined at annotation-protocol granularity (same physical setting, same session); transitivity is invoked only at the diameter of observed clusters, and the adjudication codebook fixes the granularity operationally.
- **Decision layer** (separate from estimation): three-way accept / review / reject chosen to minimize expert review volume subject to a one-sided bound on rejected true matches (FNR ≤ α) ⚠TODO:conformal implementation (owner: Jordan, with Deqian spec).

⚠TODO:Deqian — one-paragraph formal statement (equations for N ≤ S, the factorization, and the loss) lifted from the note; keep §3 under one page.

## 4. The NCR-Match Dataset

### 4.1 Source corpora
- North China Railway Archive (華北交通アーカイブ; CODH/ROIS digitization): 42,773 photographs.
- Magazine *Hokushi* (北支, 1939–1942): 2,856 printed reproductions (page crops).
- Transfer corpus (held fully out; §6.7): Sha Fei archive (Harvard-Yenching Library) + *Jinchaji Pictorial* reproductions.

### 4.2 Candidate generation and evidence
Retrieval: frozen off-the-shelf DINOv3 backbone, top-10 per source over the full corpus (427,730 candidate rows) [S1]. Correspondence floor: ASpanFormer keypoint matching, ≥50 filtered keypoints. Surviving pairs receive the full evidence vector (geometry + VGGT pose signals). Per-shard funnel (example, Shard 2): 71,190 candidates → 725 evidenced pairs [NN §3].

Two structural properties worth stating as dataset features: (i) reproductions also serve as queries by design, so retrieved reproduction-to-reproduction pairs carry full geometric evidence — same-side relations are partly observed directly, not only through the partition coupling; (ii) positive adjudications propagate for free through transitive closure, so the labeled positives force additional same-side coordinates without any extra expert review (pre-swap pooled count: 577 positives forced 195 closure coordinates; recompute on the frozen post-swap graphs ⚠TODO:Jordan).

### 4.3 Labels and the taxonomy ⚠OPEN DECISION (group; raised by Jordan 07-23)
- **What exists now**: binary expert adjudication (same exposure vs not) on 1,281+ candidate pairs (Shard 1: 643; Shard 2: 638+; counts to be finalized after ground-truth join freeze), plus transitively-implied positive (closure) pairs.
- **What the formulation defines**: ternary classes (same exposure / same scene, different exposure / unrelated). The same-scene class is produced by **expert adjudication in the negative audit** (re-binning adjudicated non-matches), never inferred automatically from filter survival — surviving the funnel does not imply same scene (two similar houses can pass; [Jordan's example]).
- **Two resolutions, pick one before the dataset-release claim is final**:
  - (a) PI completes the negative audit on the labeled pairs (~600 non-matches to re-bin) → abstract's ternary sentence stands. Owner: Lin Du; feasibility window ⚠TODO.
  - (b) Audit not complete by full-paper deadline → dataset sentence becomes "labeled same exposure or not, with scene-level structure defined for future annotation"; θ_S head reported as formulation structure only. Abstract edited accordingly.
- Either way, **the paper claims no automatic scene classifier**; scene is a label class and a structural prior (nesting penalty), not a deployed output.

### 4.4 Splits, families, and leakage control
- Family = connected component of confirmed positives; family-masked 5-fold CV within the training shard.
- Shards are disjoint subgraphs at node level. All shards retrieve against a shared reproduction pool, so node overlap is structural, not accidental: enforced by (i) family-level consolidation of labeled shards, and (ii) a **node-exposure registry**: every node the frozen model was trained over is recorded; any future evaluation graph deletes those nodes before scoring. Exercised for real on Shard 2: 49 nodes removed → 278 edges (36.7%) pruned; final evaluation graph 664 nodes / 479 edges / 266 positives [NN §3].
- Pre-registered one-way rule for late-surfacing overlaps (Shard 3 and beyond): intersections are excluded from the later shard's test set; frozen shards are never retroactively modified; counts reported.

### 4.5 Release
Hugging Face dataset: images (or links per rights status ⚠TODO:Lin — CODH licensing check), evidence vectors, adjudications, family structure, split registries, and the audit trail. Dataset card documents provenance and the historical context statement (§8).

## 5. NEG-Net

### 5.1 Inputs
- Node features: frozen DINOv3 retrieval embeddings (1024-d), LayerNorm + projection.
- Edge features: 26 evidence statistics (log-transformed counts, inlier ratio, reprojection stats, overlap, 8-d normalized homography, pose components, global similarity) + per-field missing indicators + edge-level missing-evidence flag = 53-d. Standardization fitted on training folds only, then frozen [NN §1].

### 5.2 Architecture
2-layer edge-encoder MLP (width 256) — identical to the pair-MLP baseline; L rounds of residual node/edge message passing (L=3 default; L=0 = pair-MLP attribution row); readout; two heads θ_S (scene), θ_N (exposure). 0.54M params (L=0) / 1.53M (L=3) [NN §1].

### 5.3 Training (Tier-0) and calibration
BCE anchors + triangle penalty (detached premises, active-constraint normalization) + nesting penalty (θ_N ≤ θ_S); class-prior-shift logit correction applied before any thresholding. Tier-0 is the flag-gated fallback; the perturbed Fenchel–Young objective over the nested correlation-clustering decoder is future work and stated as such ⚠TODO:Deqian(scope decision — implement or defer to camera-ready/future work).

### 5.4 Decision layer ⚠TODO:Jordan
Conformal accept/review/reject on prior-shifted probabilities; report risk–coverage, recall at review budget, one-sided FNR bound. Until built, §6 reports fixed-threshold (0.5) results and the abstract's [α]/review-reduction slots stay open.

## 6. Experiments

House rules: every comparison runs McNemar (exact binomial) with Holm correction, families scoped per STATISTICS_METHODOLOGY.md; frozen checkpoints, zero test-time retraining; two evaluation planes never mix denominators: **funnel plane** (who proposes candidates; retrieval/coverage currency) and **decision plane** (who accepts them; P/R/F1 on one frozen graph).

### 6.1 Main result — decision layer on the identical Shard-2 graph [NN §3–4]

Populations: full graph 479 edges = 449 evidenced candidates + 30 closure edges (transitively implied; zero direct evidence; **B4 structurally cannot judge them** — pointwise rules have nothing to threshold).

| Population | Model | P | R | F1 | PR-AUC |
|---|---|---|---|---|---|
| Full 479 | Hand rule B4 | 0.935 | 0.872 | 0.903 | 0.943 |
| Full 479 | pair-MLP (mp0) | 0.870 | 0.977 | 0.920 | 0.969 |
| Full 479 | **NEG-Net (mp3)** | 0.909 | 0.977 | **0.942** | 0.979 |
| Full 479 | mp3, no consistency losses | 0.899 | 0.974 | 0.935 | 0.983 |
| Candidate-only 449 | **Hand rule B4** | 0.935 | 0.983 | **0.959** | 0.994 |
| Candidate-only 449 | pair-MLP (mp0) | 0.855 | 0.975 | 0.911 | 0.962 |
| Candidate-only 449 | NEG-Net (mp3) | 0.898 | 0.975 | 0.935 | 0.973 |
| Candidate-only 449 | mp3, no consistency losses | 0.888 | 0.970 | 0.927 | 0.981 |

Closure-edge split: B4 0/30; all NEG-Net configs 30/30 (all closure edges positive-labeled).

Significance [NN §4]: full-479 B4-vs-NEG-Net differences do **not** survive Holm; candidate-only B4 wins over all configs **do** survive (p_holm ≤ 0.036); attribution ladder: message passing helps (p_holm = 0.0089), consistency losses not yet confirmed (p_holm = 0.34).

**How the paper states this (agreed framing)**: NEG-Net's aggregate advantage is a *coverage* result delivered by the graph formulation (closure pairs become decidable at all), not a pointwise decision-quality win; on matched evidence the corpus-tuned hand rule remains the stronger pointwise judge under Tier-0 training. We report both populations and both verdicts. *(The candid version of this framing is a feature of the paper, not a concession: it is exactly what the two-factor estimand predicts reviewers should ask.)*

### 6.2 In-distribution training and cross-shard generalization [NN §2–3]
Shard-1 fold: mp0 0.953 / mp3 0.958 / mp3_noloss 0.963 F1; cross-shard drop of 2–3 F1 points with recall near-ceiling (0.974–0.977); precision is where message passing pays (fewer false positives on unfamiliar data).

### 6.2b Partition-level (Q2) evaluation ⚠TODO:Jordan (design pre-registered in the note, Remark 2)
Closure-prediction experiment on Shard 1 alone: hold out the closure-implied coordinates, train on labeled candidate edges only, and measure whether the model recovers the held-out implications; plus pairwise P/R against exposure families. This upgrades §6.1's 30/30 closure observation from a by-product into a pre-registered Q2 result, and is the natural home for the "coverage, not pointwise accuracy" claim.

### 6.3 Funnel-plane ablations — Stage 1 (retrieval) [S1]
Table A (Shard 1, 292 queries, bootstrap CIs): DINO-production row circular by construction (its retrieval seeded the ground truth); challengers: DINOv3-raw R@1 98.3 / mAP 98.3, SSCD 96.2 / 94.9, CLIP 84.6 / 86.4; ranking statistically separated at R@1/mAP. Coverage: DINOv3-raw misses 2/325 known positives; union of three challengers covers 325/325. Implied full-pipeline recall if swapped in: DINOv3-raw 96.0% vs production 96.3%. ⚠TODO:Jordan — add Shard-2 columns where available [S1 has partial Shard-2 extension].

### 6.4 Funnel-plane ablations — Stage 2 (correspondence) [S2]
LightGlue vs ASpanFormer on the identical candidate set: McNemar n.s. (32 discordant pairs, raw p = 0.60); report as "representative alternatives statistically indistinguishable at this scale"; RoMa harness exists ⚠TODO:Jordan(include or state exclusion rationale).

### 6.5 Decision-rule ladder on the raw-DINO funnel [S3]
B1 (keypoint floor only) … B2 (inlier only) … B4-equiv (full rule): Shard 1 0.911 → Shard 2 0.935 F1 on the 725-pair population; B5 ≈ B4; B6 = B2; B11 (LR re-derivation) ⚠cite exact numbers from [S3] when tabling. These justify the evidence set and the hand rule's strength as a baseline.

### 6.6 Discovery-bias audit ⚠TODO:Jordan(run)+Lin(adjudicate)
Stratified relabeling of funnel-rejected candidates + pre-model hand-found pairs as a zero-shot audit of R → the funnel-coverage factor with a confidence interval. This section operationalizes §3's factorization; without it the formulation claim is words. Priority: high, small expert workload by design.

### 6.7 Zero-shot transfer — the deciding experiment ⚠IN PROGRESS:Jordan(+Lin review)
Frozen NEG-Net checkpoints AND frozen hand thresholds (0.65/2.13), both applied unchanged to the Sha Fei + Jinchaji Pictorial corpus (2,107 photographs, 1,803 reproductions). This is where "learned decision layer" vs "corpus-tuned thresholds" genuinely differentiates: the hand rule's constants were fitted to the NCR archive's print characteristics; the transfer test asks which decision layer survives a corpus change. Both outcomes are reportable; a NEG-Net win rewrites §6.1's framing upward, a loss bounds the learned claim to coverage + calibration + label efficiency.

**Preliminary (descriptive only, no golden labels yet)** [jinchaji_negnet_vs_b4_report, 2026-07-24]: 978 evidenced candidate pairs from fresh top-10 retrieval; B4 accepts 69.5%; NEG-Net consistently more conservative (57.9–62.9% positive rate); agreement 81.0–84.6%, falling as relational sophistication rises; disagreements skew 3–4x toward B4-accept/NEG-Net-reject. Direction cannot be interpreted without labels; one disagreement pair has weak-tier independent support for NEG-Net.

**Golden-label protocol (before any correctness claims)**: (i) blinded review — verdict and probability fields stripped from the review export, shuffled order; (ii) disagreement strata reviewed exhaustively, agreement strata randomly sampled with inverse-probability weighting for population P/R/F1 with CIs; (iii) only matches.csv's directly-certified tier (235 rows) trusted as prior labels, `absent_means_tp` rows treated as unlabeled; (iv) after labels land: join-labels → closure edges → candidate-vs-closure split + McNemar/Holm, same methodology as §6.1. Report the prior-shift rate used for the transfer run alongside results.

**Funnel-coverage datapoint (feeds §6.6)**: 7 previously-known matches.csv pairs were never retrieved by top-10 at all — the pre-model zero-shot audit of R on an external corpus; report the certified-tier retrieval-recovery ratio as the first measured funnel-coverage estimate.

### 6.8 Label-efficiency curve ⚠TODO:Jordan
Train on 25/50/75/100% of labeled families; F1 vs labels. Supports the "few hundred adjudications" claim.

### 6.9 Conformal risk–coverage ⚠TODO:Jordan (blocked on §5.4)
Fills [α] and review-reduction [XX]% in the abstract.

## 7. Discussion and Limitations
- **What the graph buys today**: decidability of transitively-implied pairs (structural, not incidental — no pointwise rule can pose the question); Holm-confirmed gains from relational context; a single trained layer that transfers (pending §6.7) where hand constants may not.
- **What it does not buy yet**: superiority on matched evidence under Tier-0 training; the FY/nested-clustering objective is unimplemented; θ_S is weak-label-only pending the negative audit.
- **Scene distinction**: a label-taxonomy and structural prior, not an automatic classifier (per §4.3); misclassification risk between near-identical scenes is exactly why the class is expert-defined.
- Threats to validity: single corpus family for training; shard sizes; closure edges all positive by construction (their 30/30 is coverage, not discrimination — stated plainly).

## 8. Ethics, Data, and Historical Context ⚠TODO:Lin(one paragraph)
Colonial/occupation-era archive and magazine handled as historical sources; dataset release documents provenance and context neutrally; no personal data beyond historical publication; licensing per CODH terms. The method's civic relevance (decontextualized image reuse) stated without overclaiming.

---

## Open items (single list, deduplicated, with owners)

| # | Item | Owner | Blocks |
|---|---|---|---|
| 1 | Scene-class resolution: audit (a) vs re-scope (b) in §4.3 | Lin + group call | Abstract wording, §4, §7 |
| 2 | JOCCH zero-shot: NEG-Net vs frozen thresholds | Jordan | §6.7, abstract [XX] |
| 3 | Conformal layer + risk–coverage | Jordan (Deqian spec) | §5.4, §6.9, abstract [α] |
| 4 | Discovery-bias audit (stratified rejects + pre-model pairs) | Jordan run, Lin adjudicate | §6.6 |
| 5 | Label-efficiency curve | Jordan | §6.8 |
| 6 | Related work pass | Jordan + Deqian | §2 |
| 7 | Formal ¶ for §3 | Deqian | §3 |
| 8 | [XX]% agreement number choice (F1 vs accuracy) + final abstract numbers | group | Abstract |
| 9 | HF release mechanics + licensing | Lin | §4.5 |
| 10 | Ethics paragraph | Lin | §8 |
