"""One-pass dump of pre-head 3024-d concat features for NEG-Net node inputs.

ModelComboDINO.forward() returns the post-fusion-head 1000-d embedding (fc2
output). NEG-Net uses the PRE-head 3024-d concatenation instead — 1024
(DINOv3-L) + 1000 (EfficientNetV2) + 1000 (Swin) — because it depends only on
the three frozen backbones: whichever way the fusion-head ablation lands, the
graph's node features do not change.

Features are stored RAW (no per-branch or global L2 normalization); NEG-Net
applies LayerNorm on the node input, so scaling is handled downstream.

Writes an .npz cache with the same layout as retrieve.py's feature cache
(source_paths / target_paths / source_features / target_features / metadata),
so downstream code can reference either cache interchangeably.

Usage:
    python dump_prehead_features.py \
        --model-definition ModelComboDINO.py \
        --source path/to/source_images/ \
        --target path/to/target_images/ \
        --output output/prehead_cache.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from retrieve import collect_images, load_model_from_file, load_weights, read_img


class PreHeadCombo(nn.Module):
    """Wraps ModelComboDINO to emit the pre-head 3024-d backbone concat."""

    def __init__(self, combo: nn.Module):
        super().__init__()
        self.combo = combo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.combo.model1(x.clone())
        x2 = self.combo.model2(x.clone())
        x3 = self.combo.model3(x.clone())
        return torch.cat((x1, x2, x3), dim=1)


def featurize_raw(paths: Sequence[str], model: nn.Module, device: torch.device) -> np.ndarray:
    features: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i, path in enumerate(paths, start=1):
            tensor = read_img(path).to(device)
            vec = model(tensor)
            features.append(vec.detach().cpu().numpy().astype(np.float32, copy=False))
            if i % 100 == 0 or i == len(paths):
                print(f"  featurized {i}/{len(paths)}")
    if not features:
        raise ValueError("No image features were produced")
    return np.concatenate(features, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-definition", default="ModelComboDINO.py")
    parser.add_argument("--weights", default=None,
                        help="Optional fine-tuned checkpoint; backbones are frozen, "
                             "so this only matters if backbone weights were ever touched.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combo = load_model_from_file(args.model_definition)
    if args.weights:
        load_weights(combo, args.weights, device)
    model = PreHeadCombo(combo).to(device)

    source_paths = collect_images(args.source)
    target_paths = collect_images(args.target)
    print(f"Found {len(source_paths)} source images and {len(target_paths)} target images")

    source_features = featurize_raw(source_paths, model, device)
    target_features = featurize_raw(target_paths, model, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_kind": "prehead_concat_3024",
        "components": ["dinov3_vitl16:1024", "efficientnet_v2_m:1000", "swin_t:1000"],
        "normalized": False,
        "model_definition": args.model_definition,
        "weights": args.weights,
    }
    np.savez_compressed(
        out,
        source_paths=np.asarray(source_paths, dtype=str),
        target_paths=np.asarray(target_paths, dtype=str),
        source_features=source_features,
        target_features=target_features,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=str),
    )
    print(f"Wrote {source_features.shape} + {target_features.shape} features to {out}")


if __name__ == "__main__":
    main()
