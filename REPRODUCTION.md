# Reproducing NCR-Match Pipeline Results

This document gives the exact command sequence to reproduce the P/R/F1 numbers reported
in the paper from a fresh clone of this repository.

---

## Environment

| Dependency | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0.0+cu118 |
| torchvision | 0.15.1+cu118 |
| opencv-python | ≥ 4.8 |
| ASpanFormer | see `ml-aspanformer/requirements.txt` |
| vggt-omega | pinned commit SHA — see install note below |

Install core dependencies:

```bash
pip install -r requirements.txt
pip install opencv-python
pip install git+https://github.com/facebookresearch/vggt-omega.git
```

ASpanFormer (Step 2 only):

```bash
pip install -r ml-aspanformer/requirements.txt
```

> **Note:** vggt-omega is a gated/research model. A commit SHA will be pinned here once
> the model is publicly available. Until then, install from HEAD and record the SHA of your
> install with `pip show vggt-omega`.

---

## Checkpoints

| File | Used by |
|---|---|
| `best.pt` | Step 1 — DINO retrieval backbone |
| ASpanFormer weights | Step 2 — geometric verification |
| `vggt_omega_1b_512.pt` | Step 3 — VGGT-Omega signal generation |

The notebook (`main.ipynb`) stages all checkpoints from Google Drive to the local runtime.
For CLI reproduction, supply paths via the `--weights` / `--checkpoint` arguments below.

---

## Pipeline

The NCR-Match pipeline is four sequential steps. Each step reads the previous step's
manifest and writes its own. All steps are CPU/GPU scripts with no Colab dependency.

### Step 1 — Retrieval (`retrieve.py`)

DINO-based image retrieval: scores each source against all targets, emits the top-K
candidates per source.

```bash
python retrieve.py \
  --weights path/to/best.pt \
  --model-definition ModelComboDINO.py \
  --source path/to/source_images/ \
  --target path/to/target_images/ \
  --output-dir output/retrieval/
```

Output: `output/retrieval/retrieval_manifest.jsonl`

### Step 2 — Geometric verification (`geometry_filter.py`)

ASpanFormer keypoint matching and homography estimation. Passes candidates with
≥ 50 filtered keypoints to the VGGT stage.

```bash
python geometry_filter.py \
  --input-manifest output/retrieval/retrieval_manifest.jsonl \
  --output-dir output/aspan/ \
  --aspanpath ml-aspanformer/ \
  --weights_path path/to/aspan_weights.ckpt \
  --config_path ml-aspanformer/configs/aspan/outdoor/aspan_test.py \
  --breakpoint-value 50
```

Output: `output/aspan/vggt_candidates_manifest.jsonl`

### Step 3 — VGGT-Omega signal generation (`vggt_signals.py`)

Runs VGGT-Omega on ASpan-passed pairs to record global similarity, pose encodings,
and pose component scores. This step is GPU-bound (~2 s per pair on A100).

```bash
python vggt_signals.py \
  --input-manifest output/aspan/vggt_candidates_manifest.jsonl \
  --output-dir output/vggt/ \
  --checkpoint path/to/vggt_omega_1b_512.pt
```

Output: `output/vggt/vggt_judged_manifest.jsonl`

### Step 4 — Decision layer (`pose_scoring.py`)

Applies the paper's published decision thresholds to the stored VGGT signals.
Running with defaults exactly reproduces the paper's P/R/F1.

```bash
python pose_scoring.py \
  --input-manifest output/vggt/vggt_judged_manifest.jsonl \
  --output-dir output/vggt/
```

Defaults: `--inlier-ratio-threshold 0.65  --pose-component-threshold 2.13`

Output: `output/vggt/pose_scored_manifest.jsonl`

---

## Shard roles

| Shard | Role | Notes |
|---|---|---|
| Shard 1 | Development / threshold derivation | Thresholds 0.65 / 2.13 were selected here |
| Shard 2 | Held-out validation | Thresholds applied frozen from Shard 1 |
| Shard 3 | Test | Evaluated once with frozen thresholds |

---

## Expected results

| Shard | P | R | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|
| Shard 1 (dev) | 0.867 | 0.963 | 0.913 | 313 | 48 | 268 | 12 |
| Shard 2 (val) | 0.902 | 0.984 | 0.941 | 248 | 27 | 359 | 4 |

Numbers are computed over pairs that passed Steps 1–3. Pairs labeled "Unsure" or
"Unknown" in the ground-truth CSV are excluded from all metrics.

---

## Verification

Run the acceptance test (requires ground-truth CSVs and stored vggt manifests in `_local/`):

```bash
python _local/acceptance_test.py
```

Expected output: `ACCEPTANCE TEST PASSED — Fatal F2 is closed.`

---

## Ablation studies

`pose_scoring.py` supports all ablations from the paper via CLI flags without any
code changes — see `python pose_scoring.py --help` and the ablation notes in
`pose_scoring.py`'s module docstring.
