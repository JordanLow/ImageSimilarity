#!/usr/bin/env python3
"""Run VGGT-Omega on ASpan-filtered candidate pairs and write match-judgement manifests.

This script intentionally starts *after* `aspanfilter.py`.
It consumes `vggt_candidates_manifest.jsonl` (or another aspanfilter JSONL
manifest), reads the source/target image paths from each row, runs VGGT-Omega on
that two-image sequence, and judges true matches using only the current VGGT
signals Jordan kept:

- global register-token cosine similarity
- raw `pose_enc` L1 camera-parameter shift heuristic

It does NOT rerun ASpanFormer and does NOT use the old Tier-2 structural/depth
anomaly mask.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable


TRUE_MATCH_DEFAULT = "true_matches_manifest.jsonl"
JUDGED_DEFAULT = "vggt_judged_manifest.jsonl"
SUMMARY_DEFAULT = "vggt_judge_summary.json"


def json_default(value: Any) -> Any:
    """JSON serializer for numpy/torch scalars and Path objects."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row on {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    handle.flush()


def aspan_passed(row: dict[str, Any]) -> bool:
    """Interpret aspanfilter's `passed` flag defensively.

    `vggt_candidates_manifest.jsonl` should contain only passed rows, but this
    lets the same script safely consume `aspan_all_manifest.jsonl` too.
    """
    value = row.get("passed", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}
    return bool(value)


def existing_candidate_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(path):
        candidate_id = row.get("candidate_id")
        if candidate_id is not None:
            ids.add(str(candidate_id))
    return ids


def resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def pick_device(requested: str):
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def safe_torch_load(path: Path, *, map_location: str = "cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # Some older checkpoints may include objects that are not accepted by the
        # safer weights_only loader. Fall back to the legacy mode, but only for the
        # user-provided local checkpoint path.
        return torch.load(path, map_location=map_location)


def load_vggt_omega_model(checkpoint_path: Path, device: str):
    from vggt_omega.models import VGGTOmega

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = VGGTOmega().to(device=device).eval()
    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded VGGT-Omega checkpoint {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return model


def run_vggt_pair(model, image_paths: list[Path], *, device: str, max_res: int) -> dict[str, Any]:
    import torch
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    loaded = load_and_preprocess_images([str(p) for p in image_paths], image_resolution=max_res)
    images = loaded.to(device=device, dtype=torch.float32)
    if images.ndim == 4:
        images = images.unsqueeze(0)  # [frames, C, H, W] -> [1, frames, C, H, W]
    elif images.ndim != 5:
        raise ValueError(f"Unexpected VGGT input tensor shape: {tuple(images.shape)}")

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=autocast_device,
            dtype=torch.float16,
            enabled=(autocast_device == "cuda"),
        ):
            preds = model(images)
    if not isinstance(preds, dict):
        raise TypeError(f"Expected VGGT-Omega output dict, got {type(preds)!r}")
    return preds


def tensor_frame_pair(preds: dict[str, Any], keys: Iterable[str], *, name: str):
    """Return source/target frame tensors from a VGGT prediction tensor.

    Expected common shape: [1, 2, ...]. The helper is deliberately tolerant of
    an already-squeezed [2, ...] tensor.
    """
    tensor = None
    used_key = None
    for key in keys:
        tensor = preds.get(key)
        if tensor is not None:
            used_key = key
            break
    if tensor is None:
        raise KeyError(f"Missing VGGT prediction tensor for {name}; tried {list(keys)}")

    tensor = tensor.detach().float().cpu()
    if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] >= 2:
        tensor = tensor[0]
    elif tensor.ndim >= 1 and tensor.shape[0] >= 2:
        pass
    else:
        squeezed = tensor.squeeze()
        if squeezed.ndim >= 1 and squeezed.shape[0] >= 2:
            tensor = squeezed
        else:
            raise ValueError(f"Cannot split {used_key} tensor into frame pair; shape={tuple(tensor.shape)}")
    return used_key, tensor[0], tensor[1]


def pose_values(preds: dict[str, Any]) -> tuple[list[float] | None, list[float] | None, float | None]:
    try:
        _, pose_src, pose_tgt = tensor_frame_pair(preds, ["pose_enc"], name="pose_enc")
    except KeyError:
        return None, None, None

    pose_src_np = pose_src.reshape(-1).numpy()
    pose_tgt_np = pose_tgt.reshape(-1).numpy()
    delta = abs(pose_src_np - pose_tgt_np)
    return pose_src_np.tolist(), pose_tgt_np.tolist(), float(delta.sum())


def judge_pair(
    model,
    source_path: Path,
    target_path: Path,
    *,
    device: str,
    max_res: int,
    global_sim_threshold: float,
    pose_shift_threshold: float,
    allow_missing_pose: bool,
) -> dict[str, Any]:
    import torch.nn.functional as F

    start = time.time()
    preds = run_vggt_pair(model, [source_path, target_path], device=device, max_res=max_res)

    token_key, reg_src, reg_tgt = tensor_frame_pair(
        preds,
        ["camera_and_register_tokens"],
        name="camera/register tokens",
    )
    global_similarity = float(
        F.cosine_similarity(
            reg_src.reshape(1, -1),
            reg_tgt.reshape(1, -1),
        ).item()
    )

    pose_src, pose_tgt, pose_shift_total = pose_values(preds)
    global_pass = global_similarity >= global_sim_threshold
    if pose_shift_total is None:
        pose_pass = bool(allow_missing_pose)
    else:
        pose_pass = pose_shift_total <= pose_shift_threshold

    true_match = bool(global_pass and pose_pass)
    if true_match:
        reason = "global_similarity_and_pose_shift_pass"
    elif not global_pass:
        reason = "global_similarity_below_threshold"
    elif pose_shift_total is None:
        reason = "pose_enc_missing"
    else:
        reason = "pose_shift_above_threshold"

    result = {
        "judgement": "true_match" if true_match else "rejected",
        "true_match": true_match,
        "reason": reason,
        "global_similarity": global_similarity,
        "global_sim_threshold": global_sim_threshold,
        "global_pass": global_pass,
        "pose_shift_total": pose_shift_total,
        "pose_shift_threshold": pose_shift_threshold,
        "pose_pass": pose_pass,
        "pose_src": pose_src,
        "pose_tgt": pose_tgt,
        "vggt_token_key": token_key,
        "max_res": max_res,
        "device": device,
        "runtime_seconds": round(time.time() - start, 4),
        "judge_version": "vggt_global_pose_v1_no_tier2_structural_anomaly",
    }
    cleanup_cuda()
    return result


def build_output_row(aspan_row: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    selected_aspan_keys = [
        "candidate_id",
        "source_id",
        "target_id",
        "source_path",
        "target_path",
        "rank",
        "similarity_score",
        "passed",
        "raw_match_count",
        "filtered_match_count",
        "sidecar_path",
        "error",
    ]
    row: dict[str, Any] = {k: aspan_row.get(k) for k in selected_aspan_keys if k in aspan_row}
    row.update(judge)
    row["aspanfilter"] = aspan_row
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VGGT-Omega on aspanfilter.py candidate rows and write a judged "
            "manifest plus a true-match-only manifest."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True, help="aspanfilter JSONL, usually vggt_candidates_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for VGGT judgement outputs")
    parser.add_argument("--checkpoint", type=Path, required=True, help="VGGT-Omega checkpoint, e.g. vggt_omega_1b_512.pt")
    parser.add_argument("--judged-manifest-name", default=JUDGED_DEFAULT)
    parser.add_argument("--true-match-manifest-name", default=TRUE_MATCH_DEFAULT)
    parser.add_argument("--summary-name", default=SUMMARY_DEFAULT)
    parser.add_argument("--global-sim-threshold", type=float, default=0.90)
    parser.add_argument("--pose-shift-threshold", type=float, default=0.10)
    parser.add_argument("--max-res", type=int, default=384)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Append only candidate_ids not already present in judged manifest")
    parser.add_argument("--allow-missing-pose", action="store_true", help="If pose_enc is absent, allow global similarity alone to pass")
    parser.add_argument(
        "--include-aspan-failed",
        action="store_true",
        help="Process rows even when row.passed is false. Default skips failed aspanfilter rows.",
    )
    parser.add_argument("--fail-on-pair-error", action="store_true", help="Stop on first per-pair VGGT error instead of writing rejected error rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_manifest = args.input_manifest
    if not input_manifest.exists():
        raise FileNotFoundError(input_manifest)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    judged_path = output_dir / args.judged_manifest_name
    true_path = output_dir / args.true_match_manifest_name
    summary_path = output_dir / args.summary_name

    rows = read_jsonl(input_manifest)
    if not args.include_aspan_failed:
        rows = [row for row in rows if aspan_passed(row)]
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]

    already_done = existing_candidate_ids(judged_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    device = pick_device(args.device)

    print(f"Input manifest: {input_manifest}")
    print(f"Rows to consider: {len(rows)}")
    print(f"Already judged: {len(already_done)}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print("Judge: global register-token similarity + pose_enc L1 shift; no Tier-2 structural anomaly mask")

    model = load_vggt_omega_model(args.checkpoint, device)

    counts = {
        "input_rows_considered": len(rows),
        "skipped_resume": 0,
        "judged": 0,
        "true_matches": 0,
        "rejected": 0,
        "errors": 0,
    }

    with judged_path.open(mode, encoding="utf-8") as judged_f, true_path.open(mode, encoding="utf-8") as true_f:
        for idx, aspan_row in enumerate(rows, start=1):
            candidate_id = str(aspan_row.get("candidate_id") or f"candidate_{idx:08d}")
            if candidate_id in already_done:
                counts["skipped_resume"] += 1
                continue

            source_path = resolve_path(aspan_row.get("source_path"), input_manifest.parent)
            target_path = resolve_path(aspan_row.get("target_path"), input_manifest.parent)
            if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0):
                print(f"[{idx}/{len(rows)}] {candidate_id}: {source_path.name} -> {target_path.name}")

            try:
                if not source_path.exists():
                    raise FileNotFoundError(f"source_path does not exist: {source_path}")
                if not target_path.exists():
                    raise FileNotFoundError(f"target_path does not exist: {target_path}")
                judge = judge_pair(
                    model,
                    source_path,
                    target_path,
                    device=device,
                    max_res=args.max_res,
                    global_sim_threshold=args.global_sim_threshold,
                    pose_shift_threshold=args.pose_shift_threshold,
                    allow_missing_pose=args.allow_missing_pose,
                )
            except Exception as exc:
                cleanup_cuda()
                if args.fail_on_pair_error:
                    raise
                counts["errors"] += 1
                judge = {
                    "judgement": "error",
                    "true_match": False,
                    "reason": "vggt_pair_error",
                    "error": repr(exc),
                    "global_similarity": None,
                    "global_pass": False,
                    "pose_shift_total": None,
                    "pose_pass": False,
                    "global_sim_threshold": args.global_sim_threshold,
                    "pose_shift_threshold": args.pose_shift_threshold,
                    "max_res": args.max_res,
                    "device": device,
                    "judge_version": "vggt_global_pose_v1_no_tier2_structural_anomaly",
                }

            out_row = build_output_row(aspan_row, judge)
            out_row["candidate_id"] = candidate_id
            out_row["source_path"] = str(source_path)
            out_row["target_path"] = str(target_path)
            write_jsonl_row(judged_f, out_row)
            counts["judged"] += 1
            if out_row.get("true_match") is True:
                write_jsonl_row(true_f, out_row)
                counts["true_matches"] += 1
            else:
                counts["rejected"] += 1

    summary = {
        **counts,
        "input_manifest": str(input_manifest),
        "judged_manifest": str(judged_path),
        "true_match_manifest": str(true_path),
        "checkpoint": str(args.checkpoint),
        "global_sim_threshold": args.global_sim_threshold,
        "pose_shift_threshold": args.pose_shift_threshold,
        "max_res": args.max_res,
        "device": device,
        "allow_missing_pose": args.allow_missing_pose,
        "judge_version": "vggt_global_pose_v1_no_tier2_structural_anomaly",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
