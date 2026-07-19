"""Step 2 of 4 — ASpanFormer geometric verification for the NCR-Match pipeline.

Consumes retrieval_manifest.jsonl from retrieve.py (Step 1) and writes:
- aspan_all_manifest.jsonl: audit row for every attempted candidate pair,
  including pairs that fail the keypoint-count gate (Filter 1). This file
  supports post-hoc keypoint-threshold sweeps (Table B, row B1) without
  re-running ASpanFormer.
- vggt_candidates_manifest.jsonl: pairs passing Filter 1 only (raw keypoint
  count >= --breakpoint-value). Consumed by vggt_signals.py (Step 3).
- NPZ sidecars for passed pairs containing keypoint/alignment data for VGGT.

Filter 1 (keypoint count gate) is a COMPUTE gate, not a match decision. It
prevents running the expensive VGGT inference on pairs with very few keypoints.
The final match decision is made by pose_scoring.py (Step 4) using the signals
recorded here (aspan_2d_inlier_ratio) and in vggt_signals.py.

RANSAC seeding: cv2.setRNGSeed(RANSAC_SEED) is called before each
findFundamentalMat call to ensure bit-for-bit reproducible results across runs.

Previous step: retrieve.py (Step 1).
Next step:     vggt_signals.py (Step 3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

import aspan_batching

RANSAC_SEED = 0  # cv2.setRNGSeed is called before each RANSAC call for determinism


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
                continue  # excluded so the pair is retried
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


def torch_load_weights(path: str, map_location):
    """Load trusted model weights with PyTorch-version compatibility."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location=map_location)


def load_aspanformer(aspanpath: str, config_path: str, weights_path: str, device: torch.device):
    sys.path.insert(0, os.path.abspath(aspanpath))
    from src.ASpanFormer.aspanformer import ASpanFormer
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config

    config = get_cfg_defaults()
    config.merge_from_file(config_path)
    lowered = lower_config(config)
    matcher = ASpanFormer(config=lowered["aspan"])
    payload = torch_load_weights(weights_path, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    matcher.load_state_dict(state_dict, strict=False)
    matcher = matcher.to(device)
    matcher.eval()
    return matcher


def keypoints_to_original(coords: np.ndarray, scale: float) -> np.ndarray:
    if coords.size == 0:
        return coords.astype(np.float32).reshape(0, 2)
    return (coords.astype(np.float32) / float(scale)).astype(np.float32)


def run_ransac(raw0: np.ndarray, raw1: np.ndarray) -> dict[str, Any]:
    """RANSAC filtering step, shared by the per-pair (run_aspan_pair) and
    batched (aspan_batching.run_aspan_batch) matcher paths -- CPU-only, fast,
    pair-independent, unchanged by batching."""
    raw_count = int(len(raw0))
    fundamental = None
    mask = np.zeros(raw_count, dtype=bool)
    filtered0 = np.empty((0, 2), dtype=np.float32)
    filtered1 = np.empty((0, 2), dtype=np.float32)
    ransac_error = None

    if raw_count >= 8:
        try:
            cv2.setRNGSeed(RANSAC_SEED)
            fundamental, mask_raw = cv2.findFundamentalMat(
                raw0, raw1, method=cv2.FM_RANSAC, ransacReprojThreshold=1
            )
            if mask_raw is not None:
                mask = mask_raw[:, 0].astype(bool)
            filtered0 = raw0[mask].astype(np.float32)
            filtered1 = raw1[mask].astype(np.float32)
        except cv2.error as exc:
            ransac_error = str(exc)
    else:
        ransac_error = f"not enough raw keypoints for RANSAC: {raw_count}"

    return {
        "filtered_mkpts0_resized": filtered0,
        "filtered_mkpts1_resized": filtered1,
        "ransac_mask": mask.astype(np.uint8),
        "fundamental_matrix": None if fundamental is None else fundamental.astype(np.float32),
        "raw_keypoint_count": raw_count,
        "filtered_keypoint_count": int(len(filtered0)),
        "ransac_error": ransac_error,
    }


def run_aspan_pair(
    matcher,
    source_path: str,
    target_path: str,
    long_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    img0_color = cv2.imread(source_path)
    img1_color = cv2.imread(target_path)
    img0_gray = cv2.imread(source_path, 0)
    img1_gray = cv2.imread(target_path, 0)
    if img0_color is None or img1_color is None or img0_gray is None or img1_gray is None:
        raise FileNotFoundError(f"Could not read source or target image: {source_path}, {target_path}")

    h0, w0 = img0_gray.shape[:2]
    h1, w1 = img1_gray.shape[:2]
    img0_resized, scale0 = resize_with_scale(img0_gray, long_dim)
    img1_resized, scale1 = resize_with_scale(img1_gray, long_dim)
    rh0, rw0 = img0_resized.shape[:2]
    rh1, rw1 = img1_resized.shape[:2]

    data = {
        "image0": torch.from_numpy(img0_resized / 255.0)[None, None].to(device).float(),
        "image1": torch.from_numpy(img1_resized / 255.0)[None, None].to(device).float(),
    }

    with torch.no_grad():
        matcher(data, online_resize=True)
        raw0 = data["mkpts0_f"].detach().cpu().numpy().astype(np.float32)
        raw1 = data["mkpts1_f"].detach().cpu().numpy().astype(np.float32)

    ransac = run_ransac(raw0, raw1)

    return {
        "source_original_size": [int(w0), int(h0)],
        "target_original_size": [int(w1), int(h1)],
        "source_resized_size": [int(rw0), int(rh0)],
        "target_resized_size": [int(rw1), int(rh1)],
        "source_scale": float(scale0),
        "target_scale": float(scale1),
        "raw_mkpts0_resized": raw0,
        "raw_mkpts1_resized": raw1,
        "raw_mkpts0_original": keypoints_to_original(raw0, scale0),
        "raw_mkpts1_original": keypoints_to_original(raw1, scale1),
        "filtered_mkpts0_original": keypoints_to_original(ransac["filtered_mkpts0_resized"], scale0),
        "filtered_mkpts1_original": keypoints_to_original(ransac["filtered_mkpts1_resized"], scale1),
        **ransac,
    }


def sidecar_name(candidate_id: str, source_path: str, target_path: str) -> str:
    digest = hashlib.sha1(f"{candidate_id}|{source_path}|{target_path}".encode("utf-8")).hexdigest()[:12]
    safe_candidate = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in candidate_id)
    return f"{safe_candidate}_{digest}.npz"


def save_sidecar(path: Path, match: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "raw_mkpts0_resized": match["raw_mkpts0_resized"],
        "raw_mkpts1_resized": match["raw_mkpts1_resized"],
        "raw_mkpts0_original": match["raw_mkpts0_original"],
        "raw_mkpts1_original": match["raw_mkpts1_original"],
        "filtered_mkpts0_resized": match["filtered_mkpts0_resized"],
        "filtered_mkpts1_resized": match["filtered_mkpts1_resized"],
        "filtered_mkpts0_original": match["filtered_mkpts0_original"],
        "filtered_mkpts1_original": match["filtered_mkpts1_original"],
        "ransac_mask": match["ransac_mask"],
        "source_original_size": np.asarray(match["source_original_size"], dtype=np.int32),
        "target_original_size": np.asarray(match["target_original_size"], dtype=np.int32),
        "source_resized_size": np.asarray(match["source_resized_size"], dtype=np.int32),
        "target_resized_size": np.asarray(match["target_resized_size"], dtype=np.int32),
        "scales": np.asarray([match["source_scale"], match["target_scale"]], dtype=np.float32),
    }
    if match["fundamental_matrix"] is not None:
        arrays["fundamental_matrix"] = match["fundamental_matrix"]
    np.savez_compressed(path, **arrays)


def row_base(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source_index": candidate.get("source_index"),
        "target_index": candidate.get("target_index"),
        "source_id": candidate.get("source_id"),
        "target_id": candidate.get("target_id"),
        "source_path": candidate.get("source_path"),
        "target_path": candidate.get("target_path"),
        "rank": candidate.get("rank"),
        "similarity_score": candidate.get("similarity_score"),
        "retrieval_metadata": candidate.get("retrieval_metadata"),
    }


def process(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")

    output_dir = Path(args.output_dir)
    sidecar_dir = output_dir / args.sidecar_dir
    all_manifest = output_dir / args.all_manifest_name
    passed_manifest = output_dir / args.passed_manifest_name
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(all_manifest, skip_errors=args.retry_errors) if args.resume else set()
    passed_ids = load_processed_ids(passed_manifest) if args.resume else set()
    if not args.resume:
        all_manifest.write_text("", encoding="utf-8")
        passed_manifest.write_text("", encoding="utf-8")

    matcher = load_aspanformer(args.aspanpath, args.config_path, args.weights_path, device)
    counters = {"checked": 0, "passed": 0, "skipped": 0, "failed": 0}

    def pending_candidates() -> Iterable[dict[str, Any]]:
        for _line_no, candidate in read_jsonl(args.input_manifest):
            candidate_id = str(candidate.get("candidate_id"))
            if args.resume and candidate_id in processed_ids:
                counters["skipped"] += 1
                continue
            if args.max_pairs is not None and counters["checked"] >= args.max_pairs:
                return
            counters["checked"] += 1
            yield candidate

    def handle_result(
        candidate: dict[str, Any], match: dict[str, Any] | None, error: str | None, runtime: float
    ) -> None:
        candidate_id = str(candidate.get("candidate_id"))
        base = row_base(candidate)

        if error is not None:
            counters["failed"] += 1
            print(f"  [FAIL] {candidate_id}: {error}")
            error_row = dict(base)
            error_row.update(
                {
                    "aspan_pass": False,
                    "exception": True,
                    "breakpoint_value": int(args.breakpoint_value),
                    "raw_keypoint_count": 0,
                    "filtered_keypoint_count": 0,
                    "runtime_seconds": runtime,
                    "error": error,
                }
            )
            append_jsonl(all_manifest, error_row)
            if counters["checked"] % args.progress_every == 0:
                print(f"checked={counters['checked']} passed={counters['passed']} failed={counters['failed']} skipped={counters['skipped']}")
            return

        source_path = candidate.get("source_path")
        target_path = candidate.get("target_path")
        is_pass = int(match["filtered_keypoint_count"]) >= int(args.breakpoint_value)

        audit_row = dict(base)
        audit_row.update(
            {
                "aspan_pass": bool(is_pass),
                "breakpoint_value": int(args.breakpoint_value),
                "raw_keypoint_count": int(match["raw_keypoint_count"]),
                "filtered_keypoint_count": int(match["filtered_keypoint_count"]),
                "source_original_size": match["source_original_size"],
                "target_original_size": match["target_original_size"],
                "source_resized_size": match["source_resized_size"],
                "target_resized_size": match["target_resized_size"],
                "source_scale": match["source_scale"],
                "target_scale": match["target_scale"],
                "runtime_seconds": runtime,
                "error": match["ransac_error"],
            }
        )

        if is_pass:
            counters["passed"] += 1
            sidecar_path = sidecar_dir / sidecar_name(candidate_id, source_path, target_path)
            save_sidecar(sidecar_path, match)
            audit_row["sidecar_path"] = norm_path(sidecar_path)
            if candidate_id not in passed_ids:
                append_jsonl(passed_manifest, audit_row)
                passed_ids.add(candidate_id)

        append_jsonl(all_manifest, audit_row)
        if counters["checked"] % args.progress_every == 0:
            print(f"checked={counters['checked']} passed={counters['passed']} failed={counters['failed']} skipped={counters['skipped']}")

    if args.batch_size <= 1:
        # Default path, byte-for-byte identical to the pre-batching code:
        # one ASpanFormer forward pass per pair, in order.
        for candidate in pending_candidates():
            source_path = candidate.get("source_path")
            target_path = candidate.get("target_path")
            start = time.perf_counter()
            try:
                if not source_path or not target_path:
                    raise ValueError("candidate row missing source_path or target_path")
                match = run_aspan_pair(matcher, source_path, target_path, args.long_dim, device)
                handle_result(candidate, match, None, time.perf_counter() - start)
            except Exception as exc:  # keep runs alive; record pair-level failures
                handle_result(candidate, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    else:
        # Batched path: candidates are first grouped by their exact post-resize
        # (source_shape, target_shape) -- see aspan_batching's module docstring
        # for why. Every resulting group can be chunked into batches of up to
        # --batch-size with zero padding needed, which is what keeps this safe
        # (padding a batch of genuinely different shapes together was found to
        # corrupt keypoints near the padded boundary -- not exercised here by
        # construction). Per-pair runtime_seconds is an average over the chunk
        # (real GPU work is fused, not isolable per pair) -- everything else
        # (audit rows, sidecars, manifests, RANSAC) is identical to the
        # batch_size<=1 path.
        def flush(chunk_rows: list[dict[str, Any]]) -> None:
            if not chunk_rows:
                return
            start = time.perf_counter()
            results = aspan_batching.run_aspan_batch(
                matcher, chunk_rows, args.long_dim, device, resize_with_scale, keypoints_to_original, run_ransac
            )
            per_row_runtime = (time.perf_counter() - start) / len(chunk_rows)
            for candidate, result in zip(chunk_rows, results):
                if "error" in result:
                    handle_result(candidate, None, result["error"], per_row_runtime)
                else:
                    handle_result(candidate, result, None, per_row_runtime)

        buckets, unresolved = aspan_batching.bucket_candidates(pending_candidates(), args.long_dim, resize_with_scale)
        print(f"Grouped {sum(len(v) for v in buckets.values())} candidates into {len(buckets)} shape buckets "
              f"({len(unresolved)} unresolved -- unreadable source/target image).")

        for candidate, error in unresolved:
            handle_result(candidate, None, error, 0.0)

        for bucket_rows in buckets.values():
            for i in range(0, len(bucket_rows), args.batch_size):
                flush(bucket_rows[i : i + args.batch_size])

    print(f"Done. checked={counters['checked']} passed={counters['passed']} failed={counters['failed']} skipped={counters['skipped']}")
    print(f"All-pairs manifest: {all_manifest}")
    print(f"VGGT candidates: {passed_manifest}")
    print(f"Sidecars: {sidecar_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter retrieval manifest pairs with ASpanFormer.")
    parser.add_argument("--input-manifest", required=True, help="retrieval_manifest.jsonl from match_new.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--breakpoint-value", type=int, required=True)
    parser.add_argument("--aspanpath", required=True, help="Path containing ml-aspanformer/src")
    parser.add_argument("--weights_path", type=str, default="./weights/outdoor.ckpt")
    parser.add_argument("--config_path", type=str, default="./ml-aspanformer/configs/aspan/outdoor/aspan_test.py")
    parser.add_argument("--long_dim", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of candidate pairs to run through one ASpanFormer forward pass. "
        "Default 1 preserves the exact pre-batching per-pair code path unchanged.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--all-manifest-name", default="aspan_all_manifest.jsonl")
    parser.add_argument("--passed-manifest-name", default="vggt_candidates_manifest.jsonl")
    parser.add_argument("--sidecar-dir", default="aspan_sidecars")
    parser.add_argument("--resume", action="store_true", help="Skip candidate_ids already present in all manifest")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, re-process pairs that previously raised an exception (exception=True rows)")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional debug cap")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    process(parse_args(argv))


if __name__ == "__main__":
    main()
