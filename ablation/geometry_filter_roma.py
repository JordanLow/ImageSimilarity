"""Step 2 of 4 (RoMa variant) — C2 ablation for NCR-Match Stage 2.

Drop-in replacement for geometry_filter.py that uses RoMa (Robust Dense Feature
Matching, Edstedt et al., CVPR 2024) instead of ASpanFormer. The output manifest
schema and NPZ sidecar format are identical to geometry_filter.py, so vggt_signals.py
(Step 3) and pose_scoring.py (Step 4) run completely unchanged.

RoMa produces a dense correspondence warp field rather than sparse keypoints. This
script samples `--n-matches` highest-certainty point pairs from the warp, then applies
the same RANSAC step as geometry_filter.py and geometry_filter_lightglue.py.

As a result:
  raw_keypoint_count      = n_matches (samples taken before RANSAC)
  filtered_keypoint_count = RANSAC inlier count

Requirements:
    pip install git+https://github.com/Parskatt/RoMA.git einops timm

Usage (Colab):
    python geometry_filter_roma.py \\
        --input-manifest /path/to/retrieval_manifest.jsonl \\
        --output-dir    /path/to/roma_output/ \\
        --breakpoint-value 50 \\
        [--long-dim 1024] \\
        [--n-matches 5000] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

RANSAC_SEED = 0  # same value as geometry_filter.py for reproducibility


# ── Shared I/O helpers (identical to geometry_filter_lightglue.py) ─────────────

def norm_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "candidate_id" not in row:
                row["candidate_id"] = f"line_{line_no:08d}"
            yield line_no, row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_processed_ids(path: Path, skip_errors: bool = False) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate_id = row.get("candidate_id")
            if not candidate_id:
                continue
            if skip_errors and row.get("exception"):
                continue
            processed.add(str(candidate_id))
    return processed


def resize_with_scale(img: np.ndarray, long_dim: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image dimensions")
    scale = float(long_dim) / float(max(h, w))
    h_new = max(1, int(round(h * scale)))
    w_new = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    return resized, scale


def keypoints_to_original(coords: np.ndarray, scale: float) -> np.ndarray:
    if coords.size == 0:
        return coords.astype(np.float32).reshape(0, 2)
    return (coords.astype(np.float32) / float(scale)).astype(np.float32)


def sidecar_name(candidate_id: str, source_path: str, target_path: str) -> str:
    digest = hashlib.sha1(
        f"{candidate_id}|{source_path}|{target_path}".encode("utf-8")
    ).hexdigest()[:12]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in candidate_id)
    return f"{safe}_{digest}.npz"


def save_sidecar(path: Path, match: dict[str, Any]) -> None:
    """Write NPZ sidecar — identical schema to geometry_filter.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "raw_mkpts0_resized":       match["raw_mkpts0_resized"],
        "raw_mkpts1_resized":       match["raw_mkpts1_resized"],
        "raw_mkpts0_original":      match["raw_mkpts0_original"],
        "raw_mkpts1_original":      match["raw_mkpts1_original"],
        "filtered_mkpts0_resized":  match["filtered_mkpts0_resized"],
        "filtered_mkpts1_resized":  match["filtered_mkpts1_resized"],
        "filtered_mkpts0_original": match["filtered_mkpts0_original"],
        "filtered_mkpts1_original": match["filtered_mkpts1_original"],
        "ransac_mask":              match["ransac_mask"],
        "source_original_size": np.asarray(match["source_original_size"], dtype=np.int32),
        "target_original_size": np.asarray(match["target_original_size"], dtype=np.int32),
        "source_resized_size":  np.asarray(match["source_resized_size"],  dtype=np.int32),
        "target_resized_size":  np.asarray(match["target_resized_size"],  dtype=np.int32),
        "scales": np.asarray([match["source_scale"], match["target_scale"]], dtype=np.float32),
    }
    if match["fundamental_matrix"] is not None:
        arrays["fundamental_matrix"] = match["fundamental_matrix"]
    np.savez_compressed(path, **arrays)


def row_base(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id":       candidate.get("candidate_id"),
        "source_index":       candidate.get("source_index"),
        "target_index":       candidate.get("target_index"),
        "source_id":          candidate.get("source_id"),
        "target_id":          candidate.get("target_id"),
        "source_path":        candidate.get("source_path"),
        "target_path":        candidate.get("target_path"),
        "rank":               candidate.get("rank"),
        "similarity_score":   candidate.get("similarity_score"),
        "retrieval_metadata": candidate.get("retrieval_metadata"),
    }


# ── RoMa model loading ────────────────────────────────────────────────────────

def load_roma(device: torch.device):
    """Load RoMa outdoor dense matcher.

    Requires: pip install git+https://github.com/Parskatt/RoMA.git einops timm
    Weights are downloaded via torch.hub on first call (RoMa checkpoint from GitHub
    releases, DINOv2 backbone from fbaipublicfiles.com — no auth needed, ~300MB+).
    """
    try:
        from romatch import roma_outdoor
    except ImportError:
        raise ImportError(
            "romatch is required: "
            "pip install git+https://github.com/Parskatt/RoMA.git einops timm"
        )
    # roma_outdoor() raises RuntimeError unless this is set to "highest" -- set it
    # explicitly rather than relying on nothing upstream having changed it.
    torch.set_float32_matmul_precision("highest")
    # use_custom_corr=False avoids the fused-local-corr CUDA extension, which the
    # library defaults to using but Colab often can't build cleanly (mismatched
    # nvcc/CUDA_HOME) -- pure-PyTorch fallback trades some speed for reliability here.
    return roma_outdoor(
        device=device, coarse_res=560, upsample_res=864, use_custom_corr=False,
    ).eval()


# ── RoMa pair matching ────────────────────────────────────────────────────────

def run_roma_pair(
    model,
    source_path: str,
    target_path: str,
    long_dim: int,
    device: torch.device,
    n_matches: int = 5000,
) -> dict[str, Any]:
    """Run RoMa on one image pair.

    Returns a dict with the same keys as geometry_filter.py's run_aspan_pair(),
    so save_sidecar() and the manifest-writing code are fully reusable.

    Strategy:
      1. Resize both images to long_dim (same as production pipeline).
      2. Feed resized PIL images to RoMa → dense warp + certainty map.
      3. Sample n_matches highest-certainty correspondences from the warp.
      4. Convert from RoMa's normalised [-1,1] coords to pixel coords in resized space.
      5. Run fundamental-matrix RANSAC (same parameters as geometry_filter.py).
    """
    from PIL import Image as PILImage

    img0_color = cv2.imread(source_path)
    img1_color = cv2.imread(target_path)
    if img0_color is None or img1_color is None:
        raise FileNotFoundError(
            f"Could not read source or target image: {source_path}, {target_path}"
        )

    orig_h0, orig_w0 = img0_color.shape[:2]
    orig_h1, orig_w1 = img1_color.shape[:2]

    img0_resized, scale0 = resize_with_scale(img0_color, long_dim)
    img1_resized, scale1 = resize_with_scale(img1_color, long_dim)

    rh0, rw0 = img0_resized.shape[:2]
    rh1, rw1 = img1_resized.shape[:2]

    # Convert BGR→RGB and wrap as PIL images for RoMa
    src_pil = PILImage.fromarray(cv2.cvtColor(img0_resized, cv2.COLOR_BGR2RGB))
    tgt_pil = PILImage.fromarray(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB))

    with torch.inference_mode():
        # Dense warp: coordinates are relative to the resized images we passed.
        warp, certainty = model.match(src_pil, tgt_pil, device=device)
        # matches: (n_matches, 4) normalised [-1,1] — (xA, yA, xB, yB)
        matches, _ = model.sample(warp, certainty, num=n_matches)
        # to_pixel_coordinates (NOT to_pixel_coords -- that method doesn't exist on
        # the matcher) returns a tuple of two (N, 2) tensors, not one (N, 4) tensor.
        kpts0_t, kpts1_t = model.to_pixel_coordinates(matches, rh0, rw0, rh1, rw1)

    raw0 = kpts0_t.cpu().numpy().astype(np.float32)  # (n_matches, 2) [x, y]
    raw1 = kpts1_t.cpu().numpy().astype(np.float32)

    raw_count = int(len(raw0))

    # ── Fundamental-matrix RANSAC (same parameters as geometry_filter.py) ────
    fundamental  = None
    mask         = np.zeros(raw_count, dtype=bool)
    filtered0    = np.empty((0, 2), dtype=np.float32)
    filtered1    = np.empty((0, 2), dtype=np.float32)
    ransac_error = None

    if raw_count >= 8:
        try:
            cv2.setRNGSeed(RANSAC_SEED)
            fundamental, mask_raw = cv2.findFundamentalMat(
                raw0, raw1,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=1,
            )
            if mask_raw is not None:
                mask = mask_raw[:, 0].astype(bool)
            filtered0 = raw0[mask].astype(np.float32)
            filtered1 = raw1[mask].astype(np.float32)
        except cv2.error as exc:
            ransac_error = str(exc)
    else:
        ransac_error = f"not enough RoMa samples for RANSAC: {raw_count}"

    return {
        "source_original_size": [int(orig_w0), int(orig_h0)],
        "target_original_size": [int(orig_w1), int(orig_h1)],
        "source_resized_size":  [int(rw0), int(rh0)],
        "target_resized_size":  [int(rw1), int(rh1)],
        "source_scale":  float(scale0),
        "target_scale":  float(scale1),
        "raw_mkpts0_resized":       raw0,
        "raw_mkpts1_resized":       raw1,
        "raw_mkpts0_original":      keypoints_to_original(raw0, scale0),
        "raw_mkpts1_original":      keypoints_to_original(raw1, scale1),
        "filtered_mkpts0_resized":  filtered0,
        "filtered_mkpts1_resized":  filtered1,
        "filtered_mkpts0_original": keypoints_to_original(filtered0, scale0),
        "filtered_mkpts1_original": keypoints_to_original(filtered1, scale1),
        "ransac_mask":        mask.astype(np.uint8),
        "fundamental_matrix": None if fundamental is None else fundamental.astype(np.float32),
        "raw_keypoint_count":      raw_count,
        "filtered_keypoint_count": int(len(filtered0)),
        "ransac_error":            ransac_error,
    }


# ── Main processing loop ──────────────────────────────────────────────────────

def process(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    output_dir      = Path(args.output_dir)
    sidecar_dir     = output_dir / args.sidecar_dir
    all_manifest    = output_dir / args.all_manifest_name
    passed_manifest = output_dir / args.passed_manifest_name
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_ids = (
        load_processed_ids(all_manifest, skip_errors=args.retry_errors)
        if args.resume else set()
    )
    passed_ids = load_processed_ids(passed_manifest) if args.resume else set()

    if not args.resume:
        all_manifest.write_text("", encoding="utf-8")
        passed_manifest.write_text("", encoding="utf-8")

    print(f"Device: {device}")
    print(f"Loading RoMa outdoor (n_matches={args.n_matches}) ...")
    model = load_roma(device)
    print("RoMa loaded.")

    checked = passed = skipped = failed = 0

    for _line_no, candidate in read_jsonl(args.input_manifest):
        candidate_id = str(candidate.get("candidate_id"))
        if args.resume and candidate_id in processed_ids:
            skipped += 1
            continue
        if args.max_pairs is not None and checked >= args.max_pairs:
            break

        source_path = candidate.get("source_path")
        target_path = candidate.get("target_path")
        checked += 1
        start = time.perf_counter()
        base  = row_base(candidate)

        try:
            if not source_path or not target_path:
                raise ValueError("candidate row missing source_path or target_path")

            match = run_roma_pair(
                model,
                source_path, target_path,
                args.long_dim, device, args.n_matches,
            )
            runtime = time.perf_counter() - start
            is_pass = match["filtered_keypoint_count"] >= args.breakpoint_value

            audit_row = dict(base)
            audit_row.update({
                "matcher":          "roma_outdoor",
                "n_matches_cfg":    args.n_matches,
                "aspan_pass":       bool(is_pass),        # field name kept for vggt_signals.py compat
                "breakpoint_value": int(args.breakpoint_value),
                "raw_keypoint_count":      int(match["raw_keypoint_count"]),
                "filtered_keypoint_count": int(match["filtered_keypoint_count"]),
                "source_original_size": match["source_original_size"],
                "target_original_size": match["target_original_size"],
                "source_resized_size":  match["source_resized_size"],
                "target_resized_size":  match["target_resized_size"],
                "source_scale": match["source_scale"],
                "target_scale": match["target_scale"],
                "runtime_seconds": runtime,
                "error": match["ransac_error"],
            })

            if is_pass:
                passed += 1
                sc_path = sidecar_dir / sidecar_name(candidate_id, source_path, target_path)
                save_sidecar(sc_path, match)
                audit_row["sidecar_path"] = norm_path(sc_path)
                if candidate_id not in passed_ids:
                    append_jsonl(passed_manifest, audit_row)
                    passed_ids.add(candidate_id)

            append_jsonl(all_manifest, audit_row)

        except Exception as exc:
            failed += 1
            runtime = time.perf_counter() - start
            print(f"  [FAIL] {candidate_id}: {type(exc).__name__}: {exc}")
            error_row = dict(base)
            error_row.update({
                "matcher":          "roma_outdoor",
                "aspan_pass":       False,
                "exception":        True,
                "breakpoint_value": int(args.breakpoint_value),
                "raw_keypoint_count":      0,
                "filtered_keypoint_count": 0,
                "runtime_seconds": runtime,
                "error": f"{type(exc).__name__}: {exc}",
            })
            append_jsonl(all_manifest, error_row)

        if checked % args.progress_every == 0:
            print(f"checked={checked} passed={passed} failed={failed} skipped={skipped}")

    print(f"\nDone. checked={checked} passed={passed} failed={failed} skipped={skipped}")
    print(f"All-pairs manifest:  {all_manifest}")
    print(f"VGGT candidates:     {passed_manifest}")
    print(f"Sidecars:            {sidecar_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoMa geometry filter — C2 ablation replacement for geometry_filter.py"
    )
    p.add_argument("--input-manifest",  required=True, help="retrieval_manifest.jsonl from retrieve.py")
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--breakpoint-value", type=int, default=50,
                   help="Min RANSAC-filtered keypoint count to pass Filter 1 (default: 50)")
    p.add_argument("--long-dim",  type=int, default=1024, help="Resize long edge to this many pixels")
    p.add_argument("--n-matches", type=int, default=5000,
                   help="Correspondences to sample from RoMa dense warp before RANSAC (default: 5000)")
    p.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    p.add_argument("--all-manifest-name",    default="roma_all_manifest.jsonl")
    p.add_argument("--passed-manifest-name", default="vggt_candidates_manifest.jsonl",
                   help="Keep default so vggt_signals.py can be pointed at this directly")
    p.add_argument("--sidecar-dir", default="roma_sidecars")
    p.add_argument("--resume",       action="store_true", help="Skip already-processed candidate_ids")
    p.add_argument("--retry-errors", action="store_true", help="With --resume, retry exception rows")
    p.add_argument("--max-pairs",    type=int, default=None, help="Debug cap on number of pairs")
    p.add_argument("--progress-every", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None) -> None:
    process(parse_args(argv))


if __name__ == "__main__":
    main()
