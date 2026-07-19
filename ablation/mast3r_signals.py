#!/usr/bin/env python3
"""Step 3 (MASt3R variant) — B10 ablation for NCR-Match Stage 3.

Drop-in alternative to vggt_signals.py that substitutes MASt3R for VGGT-Omega as the pose
model, while reusing vggt_signals.py's own alignment code so MASt3R sees the exact same
homography-aligned pair VGGT would. See ablation/MAST3R_ABLATION_METHODOLOGY.md for the
full experimental design and rationale.

The pipeline's forensic signal is not literal pose accuracy: after Stage 2's homography
aligns a candidate pair, a true match should reduce to ~zero apparent camera motion, while
a same-scene-different-shot pair has genuine 3D structure the homography cannot fully
explain away, so a 3D-aware model given the aligned pair reports ("hallucinates") residual
motion proportional to that leftover inconsistency. This script reproduces that setup with
MASt3R: it is fed the identical aligned pair vggt_signals.py would compute (imported
directly, not reimplemented, to avoid any drift), then reports the pose MASt3R infers
between the two.

Two pose-derivation routes are computed per candidate:
  - Primary: dense matches (fast_reciprocal_NNs) -> essential-matrix RANSAC -> recoverPose
    -> rotation angle. Matches the protocol MASt3R's own paper uses for its published
    two-view relative-pose benchmark (CO3Dv2/RealEstate10K, Table 3).
  - Completeness: PnP using MASt3R's own predicted 3D points (pred2['pts3d_in_other_view'],
    already expressed in the aligned source's frame) against their matched 2D pixels in the
    target image -> properly-scaled translation magnitude, since essential-matrix
    decomposition only gives translation direction.

Camera intrinsics (needed for the essential matrix) are not available for these
uncalibrated scans, so a focal length is estimated directly from MASt3R's own predicted
point map via dust3r.post_process.estimate_focal_knowing_depth -- self-consistent with
MASt3R's own geometry rather than an external assumption. Principal point is assumed at
the image center and a single shared K is used for both views (both images are resized to
the same convention) -- both stated simplifications, not hidden ones.

This script deliberately does NOT hardcode a composite pose-score threshold or formula --
see the notebook's threshold-derivation cell, which empirically compares candidate scoring
formulas on Shard 1 only, mirroring how the existing pose-component weights were
themselves derived (not assumed).

Requirements:
    git clone --recursive https://github.com/naver/mast3r
    pip install -r mast3r/requirements.txt -r mast3r/dust3r/requirements.txt

Usage (Colab):
    python mast3r_signals.py \\
        --input-manifest /path/to/vggt_candidates_manifest.jsonl \\
        --output-dir     /path/to/mast3r_output/ \\
        --checkpoint     naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \\
        [--resume]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# vggt_signals.py's alignment code is reused directly (not reimplemented) so MASt3R sees
# a byte-identical aligned pair to what VGGT would compute -- see module docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vggt_signals import prepare_vggt_inputs, resolve_path  # noqa: E402

RANSAC_SEED = 0  # same convention as geometry_filter.py / vggt_signals.py


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_processed_ids(path: Path) -> set[str]:
    """IDs to skip on --resume. Only successful rows count as done -- a previously
    failed candidate (out_row["error"] is not None) is retried on the next --resume
    run rather than being silently skipped forever."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(path):
        candidate_id = row.get("candidate_id")
        if candidate_id and row.get("error") is None:
            ids.add(str(candidate_id))
    return ids


def cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def load_mast3r(checkpoint: str, device: str):
    from mast3r.model import AsymmetricMASt3R

    return AsymmetricMASt3R.from_pretrained(checkpoint).to(device).eval()


def run_mast3r_pair(
    model,
    source_png: io.BytesIO,
    target_png: io.BytesIO,
    device: str,
    n_matches: int,
) -> dict[str, Any]:
    """Run MASt3R on an already-aligned pair (PNG buffers from prepare_vggt_inputs)."""
    import mast3r.utils.path_to_dust3r  # noqa: F401  (registers the bundled dust3r on sys.path)
    from dust3r.inference import inference
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.image import load_images
    from mast3r.fast_nn import fast_reciprocal_NNs
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src_path = tmp_dir / "source_aligned.png"
        tgt_path = tmp_dir / "target.png"
        Image.open(source_png).convert("RGB").save(src_path)
        Image.open(target_png).convert("RGB").save(tgt_path)

        images = load_images([str(src_path), str(tgt_path)], size=512, verbose=False)
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    pred1, pred2 = output["pred1"], output["pred2"]
    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()

    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2, subsample_or_initxy1=8, device=device, dist="dot", block_size=2**13,
    )
    raw_count = int(len(matches_im0))
    if raw_count > n_matches:
        # Uniform random cap for RANSAC/runtime sanity -- fast_reciprocal_NNs does not
        # expose a per-match confidence here the way RoMa's certainty map does, so unlike
        # the RoMa ablation this is a plain cap, not a quality-informed selection. RANSAC
        # handles outlier rejection regardless of which subset survives the cap.
        keep = np.random.RandomState(RANSAC_SEED).choice(raw_count, size=n_matches, replace=False)
        matches_im0 = matches_im0[keep]
        matches_im1 = matches_im1[keep]

    pts0 = matches_im0.astype(np.float64)  # 2D pixels in image 0 (aligned source)
    pts1 = matches_im1.astype(np.float64)  # 2D pixels in image 1 (target)

    # Self-estimated intrinsics -- see module docstring for why this route was chosen
    # over an assumed/guessed focal length.
    import torch

    pts3d_for_focal = pred1["pts3d"].detach()
    H0, W0 = pts3d_for_focal.shape[1:3]
    # estimate_focal_knowing_depth requires pp as a (B, 2) tensor matching pts3d's batch
    # dim, NOT a plain (x, y) tuple -- it calls pp.view(-1, 1, 2) internally.
    pp = torch.tensor(
        [[W0 / 2, H0 / 2]], device=pts3d_for_focal.device, dtype=pts3d_for_focal.dtype,
    )
    focal = float(
        estimate_focal_knowing_depth(
            pts3d_for_focal, pp=pp, focal_mode="weiszfeld",
        ).item()
    )
    K = np.array([[focal, 0, W0 / 2], [0, focal, H0 / 2], [0, 0, 1]], dtype=np.float64)

    rotation_angle_deg = None
    translation_direction = None
    essential_error = None
    if len(pts0) >= 8:
        try:
            cv2.setRNGSeed(RANSAC_SEED)
            E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask)
                rvec, _ = cv2.Rodrigues(R)
                rotation_angle_deg = float(np.degrees(np.linalg.norm(rvec)))
                translation_direction = t.reshape(-1).tolist()
            else:
                essential_error = "findEssentialMat returned None"
        except cv2.error as exc:
            essential_error = str(exc)
    else:
        essential_error = f"not enough MASt3R matches for essential matrix: {len(pts0)}"

    # Completeness route: PnP using MASt3R's own regressed 3D points (already expressed
    # in image 0's / the aligned source's frame), against their matched 2D pixels in
    # image 1. This solves for image 1's (target's) camera pose relative to that frame --
    # object points and image points must both come from image 1's matches (matches_im1),
    # not a mix of image 0 and image 1 indices.
    pnp_translation_scaled = None
    pnp_error = None
    pts3d_im1 = pred2["pts3d_in_other_view"].squeeze(0).detach().cpu().numpy()
    if len(pts1) >= 4:
        try:
            rows_idx = matches_im1[:, 1].clip(0, pts3d_im1.shape[0] - 1)
            cols_idx = matches_im1[:, 0].clip(0, pts3d_im1.shape[1] - 1)
            object_points = pts3d_im1[rows_idx, cols_idx].astype(np.float64)
            image_points = pts1
            cv2.setRNGSeed(RANSAC_SEED)
            ok, _rvec_pnp, tvec_pnp, _inliers = cv2.solvePnPRansac(
                object_points, image_points, K, None, reprojectionError=8.0,
            )
            if ok:
                pnp_translation_scaled = float(np.linalg.norm(tvec_pnp))
            else:
                pnp_error = "solvePnPRansac failed"
        except cv2.error as exc:
            pnp_error = str(exc)
    else:
        pnp_error = f"not enough matches for PnP: {len(pts1)}"

    return {
        "raw_match_count": raw_count,
        "used_match_count": int(len(pts0)),
        "mast3r_focal_estimate": focal,
        "mast3r_rotation_angle_deg": rotation_angle_deg,
        "mast3r_translation_direction": translation_direction,
        "mast3r_pnp_translation_scaled": pnp_translation_scaled,
        "essential_error": essential_error,
        "pnp_error": pnp_error,
    }


def process(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judged_path = output_dir / args.judged_manifest_name

    processed_ids = load_processed_ids(judged_path) if args.resume else set()
    if not args.resume:
        judged_path.write_text("", encoding="utf-8")

    device = args.device if args.device != "auto" else ("cuda" if cuda_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading MASt3R ({args.checkpoint}) ...")
    model = load_mast3r(args.checkpoint, device)
    print("MASt3R loaded.")

    input_manifest = Path(args.input_manifest)
    rows = read_jsonl(input_manifest)
    print(f"Input manifest: {len(rows)} candidates")

    checked = failed = skipped = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id"))
        if args.resume and candidate_id in processed_ids:
            skipped += 1
            continue
        if args.max_pairs is not None and checked >= args.max_pairs:
            break

        source_path = resolve_path(row.get("source_path"), input_manifest.parent)
        target_path = resolve_path(row.get("target_path"), input_manifest.parent)
        checked += 1
        start = time.perf_counter()

        out_row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "source_id": row.get("source_id"),
            "target_id": row.get("target_id"),
        }
        try:
            image_inputs, input_meta = prepare_vggt_inputs(
                row, source_path, target_path,
                input_base_dir=input_manifest.parent,
                candidate_id=candidate_id,
                input_mode="aspan-aligned",
                alignment_keypoints="raw",  # matches vggt_signals.py's own default
                debug_aligned_dir=None,
            )
            out_row["aspan_2d_inlier_ratio"] = input_meta.get("aspan_2d_inlier_ratio")

            mast3r_result = run_mast3r_pair(
                model, image_inputs[0], image_inputs[1], device, args.n_matches,
            )
            out_row.update(mast3r_result)
            out_row["runtime_seconds"] = time.perf_counter() - start
            out_row["error"] = None
        except Exception as exc:
            failed += 1
            out_row["error"] = f"{type(exc).__name__}: {exc}"
            out_row["error_traceback"] = traceback.format_exc()
            out_row["runtime_seconds"] = time.perf_counter() - start
            print(f"  [FAIL] {candidate_id}: {out_row['error']}")
            print(out_row["error_traceback"])

        append_jsonl(judged_path, out_row)

        if args.progress_every and checked % args.progress_every == 0:
            print(f"checked={checked} failed={failed} skipped={skipped}")

    print(f"\nDone. checked={checked} failed={failed} skipped={skipped}")
    print(f"Judged manifest: {judged_path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MASt3R Stage-3 pose signals -- B10 ablation replacement for vggt_signals.py"
    )
    p.add_argument("--input-manifest", required=True,
                    help="vggt_candidates_manifest.jsonl with real (regenerated) ASpanFormer sidecars")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    p.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    p.add_argument("--n-matches", type=int, default=2000,
                    help="Cap on fast_reciprocal_NNs matches used for pose recovery (default: 2000)")
    p.add_argument("--judged-manifest-name", default="mast3r_judged_manifest.jsonl")
    p.add_argument("--resume", action="store_true", help="Skip already-processed candidate_ids")
    p.add_argument("--max-pairs", type=int, default=None, help="Debug cap on number of pairs")
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args(argv)


def main(argv=None) -> None:
    process(parse_args(argv))


if __name__ == "__main__":
    main()
