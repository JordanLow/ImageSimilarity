# NCR-Match

A four-stage pipeline for detecting near-copy reproductions of historical press photographs
across large archival image collections. The pipeline combines DINO-based retrieval with
ASpanFormer geometric verification and VGGT-Omega pose estimation to produce high-precision
match decisions.

---

## Pipeline overview

| Step | Script | Description |
|---|---|---|
| 1 | `retrieve.py` | DINO embedding retrieval — top-K candidates per source |
| 2 | `geometry_filter.py` | ASpanFormer keypoint matching and homography verification |
| 3 | `vggt_signals.py` | VGGT-Omega pose signal generation |
| 4 | `pose_scoring.py` | Decision layer — applies paper's published thresholds |

See [REPRODUCTION.md](REPRODUCTION.md) for the full command sequence and expected results.

---

## Installation

```bash
pip install -r requirements.txt
pip install opencv-python
pip install git+https://github.com/facebookresearch/vggt-omega.git
```

ASpanFormer (Step 2):

```bash
pip install -r ml-aspanformer/requirements.txt
```

---

## Quick start (Colab / Google Drive)

Open `main.ipynb` in Google Colab. Edit Section B paths to point to your Drive root,
then run all sections top to bottom.

---

## Training

To fine-tune the DINO retrieval backbone on a new image collection:

```bash
python train.py \
  --weights path/to/initial_weights.pt \
  --model-definition ModelComboDINO.py \
  --source path/to/source_images/ \
  --target path/to/target_images/ \
  --save-dir path/to/weights/ \
  --learning-rate 1e-5 \
  --batch-size 256 \
  --epochs 50 \
  --save-best-weights
```

---

## Reproducing paper results

See [REPRODUCTION.md](REPRODUCTION.md).
