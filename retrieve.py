"""Step 1 of 4 — DINO retrieval for the NCR-Match pipeline.

Featurizes source and target image collections with the DINO model and writes:
- retrieval_manifest.jsonl: one row per top-X source/target candidate pair.
- feature_cache.npz: source/target feature matrices plus same-order paths.

The feature cache stores embeddings, not the full cosine matrix. Rankings can be
regenerated from cached features without rerunning the representation model.

Use --topx / --topk to control how many candidates are retained per source image
(default 15; sweep 5→30 for retrieval Recall@K ablations).

Next step: geometry_filter.py (ASpanFormer geometric verification, Step 2).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from PIL import Image

IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}


def load_model_from_file(model_path: str, class_name: str | None = None):
    if class_name is None:
        class_name = Path(model_path).stem
    spec = importlib.util.spec_from_file_location(class_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model definition: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, class_name)
    return model_class()


def read_img(path: str) -> torch.Tensor:
    """Match match.py preprocessing: RGB -> 224x224 -> 3-channel grayscale."""
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = tf.to_tensor(img)
    img = tf.rgb_to_grayscale(img, num_output_channels=3)
    img = tf.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return img.unsqueeze(0)


def norm_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def collect_images(root: str | Path) -> list[str]:
    paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(norm_path(Path(dirpath) / filename))
    return sorted(paths)


def l2_normalize_np(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(denom, eps)


def featurize(paths: Sequence[str], model: torch.nn.Module, device: torch.device) -> np.ndarray:
    features: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i, path in enumerate(paths, start=1):
            tensor = read_img(path).to(device)
            vec = nn.functional.normalize(model(tensor), dim=1)
            features.append(vec.detach().cpu().numpy().astype(np.float32, copy=False))
            if i % 100 == 0 or i == len(paths):
                print(f"  featurized {i}/{len(paths)}")
    if not features:
        raise ValueError("No image features were produced")
    # Normalize once more after concatenation so cached float matrices stay unit-length
    # even if model outputs or future dtype conversions introduce small drift.
    return l2_normalize_np(np.concatenate(features, axis=0)).astype(np.float32, copy=False)


def torch_load_weights(path: str, map_location):
    """Load trusted model weights with PyTorch-version compatibility."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location=map_location)


def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    payload = torch_load_weights(weights_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    model.load_state_dict(payload, strict=False)


def save_cache(
    path: Path,
    source_paths: Sequence[str],
    target_paths: Sequence[str],
    source_features: np.ndarray,
    target_features: np.ndarray,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_paths=np.asarray(source_paths, dtype=str),
        target_paths=np.asarray(target_paths, dtype=str),
        source_features=source_features,
        target_features=target_features,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=str),
    )


def load_cache(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        source_paths = [str(p) for p in data["source_paths"].tolist()]
        target_paths = [str(p) for p in data["target_paths"].tolist()]
        source_features = data["source_features"]
        target_features = data["target_features"]
        metadata = json.loads(str(data["metadata"].tolist())) if "metadata" in data else {}
    return source_paths, target_paths, source_features, target_features, metadata


def iter_topx_rows(
    source_paths: Sequence[str],
    target_paths: Sequence[str],
    source_features: np.ndarray,
    target_features: np.ndarray,
    topx: int,
    chunk_size: int,
    device: torch.device,
    metadata: dict,
) -> Iterable[dict]:
    if topx < 1:
        raise ValueError("topx/topk must be >= 1")
    if len(target_paths) == 0:
        raise ValueError("No target paths available")

    k = min(topx, len(target_paths))
    source_stems = [Path(p).stem for p in source_paths]
    target_stems = [Path(p).stem for p in target_paths]
    target_tensor = torch.as_tensor(target_features.astype(np.float32), device=device)

    for start in range(0, len(source_paths), chunk_size):
        end = min(start + chunk_size, len(source_paths))
        source_tensor = torch.as_tensor(source_features[start:end].astype(np.float32), device=device)
        scores = torch.matmul(source_tensor, target_tensor.T)

        # Preserve match.py behavior: suppress same-basename self matches.
        for local_i, source_i in enumerate(range(start, end)):
            for target_i, target_stem in enumerate(target_stems):
                if source_stems[source_i] == target_stem:
                    scores[local_i, target_i] = float("-inf")

        values, indices = torch.topk(scores, k=k, dim=1)
        values_np = values.detach().cpu().numpy()
        indices_np = indices.detach().cpu().numpy()

        for local_i, source_i in enumerate(range(start, end)):
            for rank0 in range(k):
                score = float(values_np[local_i, rank0])
                if not np.isfinite(score):
                    continue
                target_i = int(indices_np[local_i, rank0])
                source_path = source_paths[source_i]
                target_path = target_paths[target_i]
                yield {
                    "candidate_id": f"s{source_i:08d}_r{rank0 + 1:03d}_t{target_i:08d}",
                    "source_index": source_i,
                    "target_index": target_i,
                    "source_id": Path(source_path).stem,
                    "target_id": Path(target_path).stem,
                    "source_path": source_path,
                    "target_path": target_path,
                    "rank": rank0 + 1,
                    "similarity_score": score,
                    "retrieval_metadata": metadata,
                }
        print(f"  ranked sources {start + 1}-{end}/{len(source_paths)}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    return device


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate top-X manifest and feature cache.")
    parser.add_argument("--weights", type=str, default="weights_2-11_199.pt")
    parser.add_argument("--model-definition", type=str, default="ModelCombo.py")
    parser.add_argument("--model-class", type=str, default=None)
    parser.add_argument("--source", type=str, default="./eval")
    parser.add_argument("--target", type=str, default="./target")
    parser.add_argument("--topx", "--topk", dest="topx", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default="match_new_output")
    parser.add_argument("--manifest-name", type=str, default="retrieval_manifest.jsonl")
    parser.add_argument("--feature-cache-name", type=str, default="feature_cache.npz")
    parser.add_argument("--features-cache", type=str, default=None)
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--feature-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / args.manifest_name
    cache_path = Path(args.features_cache) if args.features_cache else output_dir / args.feature_cache_name
    device = resolve_device(args.device)
    feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32

    metadata = {
        "model_definition": args.model_definition,
        "model_class": args.model_class or Path(args.model_definition).stem,
        "weights": args.weights,
        "topx": args.topx,
        "source_root": args.source,
        "target_root": args.target,
        "features_normalized": True,
        "feature_cache": norm_path(cache_path),
    }

    if args.skip_feature_extraction:
        print(f"Loading feature cache: {cache_path}")
        source_paths, target_paths, source_features, target_features, cache_metadata = load_cache(cache_path)
        metadata.update({"regenerated_from_cache": True, "cache_metadata": cache_metadata})
    else:
        source_paths = collect_images(args.source)
        target_paths = collect_images(args.target)
        print(f"Found {len(source_paths)} source images and {len(target_paths)} target images")
        if not source_paths or not target_paths:
            raise ValueError("Both source and target folders must contain supported image files")

        print(f"Loading model on {device}: {args.model_definition}")
        model = load_model_from_file(args.model_definition, args.model_class).to(device)
        load_weights(model, args.weights, device)

        print("Featurizing source images")
        source_features = featurize(source_paths, model, device).astype(feature_dtype, copy=False)
        print("Featurizing target images")
        target_features = featurize(target_paths, model, device).astype(feature_dtype, copy=False)

        cache_metadata = dict(metadata)
        cache_metadata.update(
            {
                "source_count": len(source_paths),
                "target_count": len(target_paths),
                "feature_dim": int(source_features.shape[1]),
                "feature_dtype": str(source_features.dtype),
            }
        )
        print(f"Writing feature cache: {cache_path}")
        save_cache(cache_path, source_paths, target_paths, source_features, target_features, cache_metadata)

    print(f"Writing retrieval manifest: {manifest_path}")
    row_count = write_jsonl(
        manifest_path,
        iter_topx_rows(
            source_paths,
            target_paths,
            source_features,
            target_features,
            args.topx,
            args.chunk_size,
            device,
            metadata,
        ),
    )
    print(f"Done: wrote {row_count} candidate rows")
    print(f"Feature cache: {cache_path}")


if __name__ == "__main__":
    main()
