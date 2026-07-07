#!/usr/bin/env python3
"""Step 3 of 4 — VGGT-Omega signal generation for the NCR-Match pipeline.

Consumes vggt_candidates_manifest.jsonl from geometry_filter.py (Step 2),
reconstructs the ASpan-aligned source → target view from the NPZ sidecar,
feeds the aligned image pair to VGGT-Omega, and records all diagnostic signals:

- global register-token cosine similarity (global_similarity)
- component-weighted pose_enc heuristic (pose_component_score, pose_component_terms)
- raw 9-D VGGT camera pose vectors (pose_src, pose_tgt)
- extremely lenient ASpanFormer 2D pre-pass for egregious non-alignments

NOTE: The true_match and judgement fields in the output manifest are PROVISIONAL,
based on this script's internal default thresholds. The paper's actual decision
rule is applied by pose_scoring.py (Step 4): inlier_ratio >= 0.65 AND
pose_component_score <= 2.13. Do not use this script's true_match field for
precision/recall numbers — use pose_scoring.py instead.

This script is a pure signal recorder. Running it with different --pose-component-
threshold or --global-sim-threshold values does NOT change the paper's results;
those thresholds in pose_scoring.py do.

It does NOT rerun ASpanFormer and does NOT use the old Tier-2 structural/depth
anomaly mask. By default it uses the official VGGT-Omega loader with in-memory
BytesIO image buffers, so no permanent aligned image files are required.

Previous step: geometry_filter.py (Step 2).
Next step:     pose_scoring.py (Step 4, the actual decision layer).
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable


TRUE_MATCH_DEFAULT = "true_matches_manifest.jsonl"
JUDGED_DEFAULT = "vggt_judged_manifest.jsonl"
SUMMARY_DEFAULT = "vggt_judge_summary.json"
JUDGE_VERSION = "vggt_global_pose_v4_component_pose_aspan_prepass_aligned_bytesio_no_tier2"
RAW_JUDGE_VERSION = "vggt_global_pose_v4_component_pose_raw_paths_no_tier2"
RANSAC_SEED = 0  # cv2.setRNGSeed is called before each findHomography call for determinism


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


def _truthy(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "passed", "pass"}:
            return True
        if text in {"0", "false", "no", "n", "failed", "fail", "rejected", "reject"}:
            return False
        return default
    return bool(value)


def aspan_passed(row: dict[str, Any]) -> bool:
    """Interpret aspanfilter pass flags defensively.

    aspanfilter.py writes `aspan_pass`; older/different manifests may use
    `passed`. `vggt_candidates_manifest.jsonl` contains only passed rows, so a
    missing flag defaults to True for backwards compatibility.
    """
    if "aspan_pass" in row:
        return _truthy(row.get("aspan_pass"), default=False)
    if "passed" in row:
        return _truthy(row.get("passed"), default=False)
    return True


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


def _load_npz_dict(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _size_tuple(value: Any, *, name: str) -> tuple[int, int]:
    import numpy as np

    # aspanfilter.py stores sizes as [width, height] because cv2.resize uses
    # (width, height). Keep that convention explicit here.
    arr = np.asarray(value).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must contain width,height; got shape {arr.shape}")
    width, height = int(arr[0]), int(arr[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive; got {(width, height)}")
    return width, height


def _resize_bgr_to_size(image, size: tuple[int, int]):
    import cv2

    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image.copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _bgr_to_png_buffer(image_bgr) -> io.BytesIO:
    import cv2
    from PIL import Image

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _write_debug_aligned_images(debug_dir: Path, candidate_id: str, source_aligned_bgr, target_bgr) -> dict[str, str]:
    import cv2

    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in candidate_id)
    pair_dir = debug_dir / safe_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    source_path = pair_dir / "source_aligned.jpg"
    target_path = pair_dir / "target.jpg"
    cv2.imwrite(str(source_path), source_aligned_bgr)
    cv2.imwrite(str(target_path), target_bgr)
    return {
        "debug_aligned_source_path": str(source_path),
        "debug_aligned_target_path": str(target_path),
    }


def _write_inspect_images(
    inspect_dir: Path,
    source_path: Path,
    target_path: Path,
    rank: int,
    true_match: bool,
) -> None:
    import shutil
    source_dir = inspect_dir / source_path.stem
    source_dir.mkdir(parents=True, exist_ok=True)

    dest_source = source_dir / ("source" + source_path.suffix)
    if not dest_source.exists():
        shutil.copy2(source_path, dest_source)

    subfolder = source_dir / ("true_matches" if true_match else "rejected")
    subfolder.mkdir(exist_ok=True)
    shutil.copy2(target_path, subfolder / f"r{rank:03d}_{target_path.name}")


def _keypoint_variants(sidecar: dict[str, Any], requested: str) -> list[tuple[str, Any, Any]]:
    variants = {
        "raw": ("raw_mkpts0_resized", "raw_mkpts1_resized"),
        "filtered": ("filtered_mkpts0_resized", "filtered_mkpts1_resized"),
    }
    if requested == "auto":
        order = ["raw", "filtered"]
    else:
        order = [requested]
    out = []
    for name in order:
        key0, key1 = variants[name]
        if key0 in sidecar and key1 in sidecar:
            out.append((name, sidecar[key0], sidecar[key1]))
    return out


def _aspan_reprojection_metrics(pts0, pts1, homography, mask) -> dict[str, Any]:
    import cv2
    import numpy as np

    raw_count = int(len(pts0))
    if raw_count == 0:
        return {
            "aspan_2d_raw_match_count": 0,
            "aspan_2d_homography_inliers": 0,
            "aspan_2d_inlier_ratio": None,
            "aspan_2d_mean_reprojection_error": None,
            "aspan_2d_median_reprojection_error": None,
        }

    if mask is not None:
        inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
    else:
        inlier_mask = np.ones(raw_count, dtype=bool)
    inlier_count = int(inlier_mask.sum())
    projected = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2).astype(np.float32), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - pts1.astype(np.float32), axis=1)
    inlier_errors = errors[inlier_mask] if inlier_count > 0 else errors
    mean_error = float(np.mean(inlier_errors)) if len(inlier_errors) else None
    median_error = float(np.median(inlier_errors)) if len(inlier_errors) else None
    return {
        "aspan_2d_raw_match_count": raw_count,
        "aspan_2d_homography_inliers": inlier_count,
        "aspan_2d_inlier_ratio": float(inlier_count / raw_count) if raw_count else None,
        "aspan_2d_mean_reprojection_error": mean_error,
        "aspan_2d_median_reprojection_error": median_error,
    }


def _aspan_overlap_metrics(homography, source_size: tuple[int, int], target_size: tuple[int, int]) -> dict[str, Any]:
    import cv2
    import numpy as np

    source_width, source_height = source_size
    target_width, target_height = target_size
    out: dict[str, Any] = {
        "aspan_2d_aligned_overlap_fraction": None,
        "aspan_2d_homography_sanity_pass": False,
        "aspan_2d_homography_sanity_reason": None,
    }
    try:
        H = np.asarray(homography, dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            out["aspan_2d_homography_sanity_reason"] = "homography_not_finite_3x3"
            return out
        det = float(np.linalg.det(H))
        if not np.isfinite(det) or abs(det) < 1e-8:
            out["aspan_2d_homography_sanity_reason"] = "homography_near_singular"
            return out
        corners = np.array(
            [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if not np.isfinite(warped).all():
            out["aspan_2d_homography_sanity_reason"] = "warped_corners_not_finite"
            return out
        # Rasterized clipped overlap of the warped source quadrilateral inside the target canvas.
        mask = np.zeros((max(1, target_height), max(1, target_width)), dtype=np.uint8)
        poly = np.round(warped).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 1)
        overlap_pixels = int(mask.sum())
        target_pixels = max(1, target_width * target_height)
        out.update(
            {
                "aspan_2d_aligned_overlap_fraction": float(overlap_pixels / target_pixels),
                "aspan_2d_homography_sanity_pass": True,
                "aspan_2d_homography_sanity_reason": "ok",
                "aspan_2d_warped_source_corners": warped.tolist(),
            }
        )
        return out
    except Exception as exc:
        out["aspan_2d_homography_sanity_reason"] = f"overlap_metric_error: {exc!r}"
        return out


def _estimate_homography_from_sidecar(sidecar: dict[str, Any], *, keypoints: str) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np

    errors: list[str] = []
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    for variant_name, mkpts0, mkpts1 in _keypoint_variants(sidecar, keypoints):
        pts0 = np.asarray(mkpts0, dtype=np.float32).reshape(-1, 2)
        pts1 = np.asarray(mkpts1, dtype=np.float32).reshape(-1, 2)
        if len(pts0) < 4 or len(pts1) < 4:
            errors.append(f"{variant_name}: need >=4 keypoints, got {len(pts0)}/{len(pts1)}")
            continue
        cv2.setRNGSeed(RANSAC_SEED)
        homography, mask = cv2.findHomography(pts0, pts1, method, 5.0)
        if homography is None:
            errors.append(f"{variant_name}: cv2.findHomography returned None")
            continue
        inliers = int(mask.sum()) if mask is not None else None
        metrics = _aspan_reprojection_metrics(pts0, pts1, homography, mask)
        return homography, {
            "alignment_keypoints": variant_name,
            "alignment_keypoint_count": int(len(pts0)),
            "alignment_homography_inliers": inliers,
            "alignment_homography_method": "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") else "RANSAC",
            **metrics,
        }
    raise ValueError("Could not estimate homography from sidecar: " + "; ".join(errors))


def prepare_vggt_inputs(
    aspan_row: dict[str, Any],
    source_path: Path,
    target_path: Path,
    *,
    input_base_dir: Path,
    candidate_id: str,
    input_mode: str,
    alignment_keypoints: str,
    debug_aligned_dir: Path | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Prepare the exact two images fed to VGGT-Omega.

    Default mode reconstructs the Experiment-notebook-style ASpan alignment from
    the aspanfilter sidecar and passes in-memory PNG buffers into the official
    VGGT-Omega loader. Raw path mode is retained only as an explicit escape hatch.
    """
    if input_mode == "raw-paths":
        return [str(source_path), str(target_path)], {
            "vggt_input_mode": "raw_paths",
            "alignment_applied": False,
            "source_vggt_input": str(source_path),
            "target_vggt_input": str(target_path),
        }

    import cv2

    sidecar_value = aspan_row.get("sidecar_path")
    if not sidecar_value:
        raise ValueError("aspan-aligned VGGT input requires aspanfilter sidecar_path")
    sidecar_path = resolve_path(sidecar_value, input_base_dir)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"sidecar_path does not exist: {sidecar_path}")

    sidecar = _load_npz_dict(sidecar_path)
    source_size = _size_tuple(sidecar["source_resized_size"], name="source_resized_size")
    target_size = _size_tuple(sidecar["target_resized_size"], name="target_resized_size")
    homography, homography_meta = _estimate_homography_from_sidecar(sidecar, keypoints=alignment_keypoints)
    overlap_meta = _aspan_overlap_metrics(homography, source_size, target_size)

    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise FileNotFoundError(f"Could not read source image with cv2: {source_path}")
    target_bgr = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if target_bgr is None:
        raise FileNotFoundError(f"Could not read target image with cv2: {target_path}")

    source_resized = _resize_bgr_to_size(source_bgr, source_size)
    target_resized = _resize_bgr_to_size(target_bgr, target_size)
    target_width, target_height = target_size
    source_aligned = cv2.warpPerspective(source_resized, homography, (target_width, target_height))

    meta: dict[str, Any] = {
        "vggt_input_mode": "aspan_aligned_bytesio",
        "alignment_applied": True,
        "alignment_sidecar_path": str(sidecar_path),
        "alignment_source_resized_size": list(source_size),
        "alignment_target_resized_size": list(target_size),
        "alignment_canvas_size": [target_width, target_height],
        "alignment_homography": homography.tolist(),
        "source_vggt_input": "aspan_aligned_source_bytesio_png",
        "target_vggt_input": "target_resized_bytesio_png",
        **homography_meta,
        **overlap_meta,
    }
    if "scales" in sidecar:
        try:
            meta["alignment_scales"] = sidecar["scales"].tolist()
        except Exception:
            pass
    if debug_aligned_dir is not None:
        meta.update(_write_debug_aligned_images(debug_aligned_dir, candidate_id, source_aligned, target_resized))

    return [_bgr_to_png_buffer(source_aligned), _bgr_to_png_buffer(target_resized)], meta


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


def run_vggt_pair(model, image_inputs: list[Any], *, device: str, max_res: int) -> dict[str, Any]:
    import torch
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    # Official VGGT-Omega utility is path-list shaped but uses PIL.Image.open
    # internally. Verified against facebookresearch/vggt-omega HEAD 39a0cb8:
    # io.BytesIO image buffers are accepted, while direct PIL.Image objects are not.
    try:
        loaded = load_and_preprocess_images(image_inputs, image_resolution=max_res)
    finally:
        for item in image_inputs:
            if isinstance(item, io.BytesIO):
                item.close()
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


def _l2(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def quaternion_rotation_angle_degrees(pose_src: list[float], pose_tgt: list[float]) -> float | None:
    """Return quaternion angular difference in degrees for VGGT pose_enc dims 3:7.

    VGGT-Omega's released checkpoints encode pose as:
      [tx, ty, tz, qx, qy, qz, qw, vertical_fov, horizontal_fov]

    Quaternions q and -q represent the same rotation, so use abs(dot).
    """
    if len(pose_src) < 7 or len(pose_tgt) < 7:
        return None
    q_src = [float(v) for v in pose_src[3:7]]
    q_tgt = [float(v) for v in pose_tgt[3:7]]
    src_norm = _l2(q_src)
    tgt_norm = _l2(q_tgt)
    if src_norm <= 0 or tgt_norm <= 0:
        return None
    dot = sum((a / src_norm) * (b / tgt_norm) for a, b in zip(q_src, q_tgt))
    dot = max(-1.0, min(1.0, abs(dot)))
    return float(math.degrees(2.0 * math.acos(dot)))


def pose_component_metrics(
    pose_src: list[float],
    pose_tgt: list[float],
    *,
    rotation_scale_deg: float,
    xy_scale: float,
    z_scale: float,
    fov_scale: float,
    rotation_weight: float,
    xy_weight: float,
    z_weight: float,
    fov_weight: float,
    term_cap: float,
) -> dict[str, Any]:
    """Compute Jordan's component-weighted pose heuristic.

    Raw pose_enc L1 mixes translation, quaternion, and FOV in incompatible units.
    This heuristic treats rotation and lateral x/y translation as stronger evidence
    of a different photo, while downweighting/capping z-depth and FOV changes that
    often arise from crop/scan/zoom artifacts in flat archival photos.
    """
    if len(pose_src) < 9 or len(pose_tgt) < 9:
        return {"pose_component_error": f"expected pose_enc length >= 9, got {len(pose_src)}/{len(pose_tgt)}"}

    deltas = [abs(float(a) - float(b)) for a, b in zip(pose_src, pose_tgt)]
    raw_total = float(sum(deltas))
    rotation_deg = quaternion_rotation_angle_degrees(pose_src, pose_tgt)
    xy_shift = float(math.hypot(float(pose_src[0]) - float(pose_tgt[0]), float(pose_src[1]) - float(pose_tgt[1])))
    z_shift = float(abs(float(pose_src[2]) - float(pose_tgt[2])))
    fov_shift = float(math.hypot(float(pose_src[7]) - float(pose_tgt[7]), float(pose_src[8]) - float(pose_tgt[8])))
    zoom_depth_fraction = float((z_shift + deltas[7] + deltas[8]) / raw_total) if raw_total > 0 else None

    def capped(value: float | None, scale: float, weight: float) -> float | None:
        if value is None or scale <= 0:
            return None
        return float(weight * min(value / scale, term_cap))

    rotation_term = capped(rotation_deg, rotation_scale_deg, rotation_weight)
    xy_term = capped(xy_shift, xy_scale, xy_weight)
    z_term = capped(z_shift, z_scale, z_weight)
    fov_term = capped(fov_shift, fov_scale, fov_weight)
    terms = [rotation_term, xy_term, z_term, fov_term]
    component_score = None if any(term is None for term in terms) else float(sum(term for term in terms if term is not None))

    return {
        "pose_rotation_deg": rotation_deg,
        "pose_translation_xy_l2": xy_shift,
        "pose_translation_z_abs": z_shift,
        "pose_fov_l2": fov_shift,
        "pose_zoom_depth_fraction": zoom_depth_fraction,
        "pose_component_score": component_score,
        "pose_component_terms": {
            "rotation": rotation_term,
            "translation_xy": xy_term,
            "translation_z": z_term,
            "fov": fov_term,
        },
        "pose_component_config": {
            "rotation_scale_deg": rotation_scale_deg,
            "xy_scale": xy_scale,
            "z_scale": z_scale,
            "fov_scale": fov_scale,
            "rotation_weight": rotation_weight,
            "xy_weight": xy_weight,
            "z_weight": z_weight,
            "fov_weight": fov_weight,
            "term_cap": term_cap,
        },
    }


def judge_pair(
    model,
    image_inputs: list[Any],
    input_meta: dict[str, Any],
    *,
    device: str,
    max_res: int,
    global_sim_threshold: float,
    pose_shift_threshold: float,
    pose_score_mode: str,
    pose_component_threshold: float,
    pose_component_config: dict[str, float],
    allow_missing_pose: bool,
) -> dict[str, Any]:
    import torch.nn.functional as F

    start = time.time()
    preds = run_vggt_pair(model, image_inputs, device=device, max_res=max_res)

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
    pose_component: dict[str, Any] = {}
    if pose_src is not None and pose_tgt is not None:
        pose_component = pose_component_metrics(pose_src, pose_tgt, **pose_component_config)

    global_pass = global_similarity >= global_sim_threshold
    if pose_shift_total is None:
        pose_pass = bool(allow_missing_pose)
    elif pose_score_mode == "raw":
        pose_pass = pose_shift_total <= pose_shift_threshold
    else:
        component_score = pose_component.get("pose_component_score")
        pose_pass = component_score is not None and component_score <= pose_component_threshold

    true_match = bool(global_pass and pose_pass)
    if true_match:
        reason = "global_similarity_and_pose_component_pass" if pose_score_mode == "component" else "global_similarity_and_pose_shift_pass"
    elif not global_pass:
        reason = "global_similarity_below_threshold"
    elif pose_shift_total is None:
        reason = "pose_enc_missing"
    elif pose_score_mode == "component" and pose_component.get("pose_component_score") is None:
        reason = "pose_component_score_missing"
    else:
        reason = "pose_component_score_above_threshold" if pose_score_mode == "component" else "pose_shift_above_threshold"

    result = {
        "judgement": "true_match" if true_match else "rejected",
        "true_match": true_match,
        "reason": reason,
        "global_similarity": global_similarity,
        "global_sim_threshold": global_sim_threshold,
        "global_pass": global_pass,
        "pose_shift_total": pose_shift_total,
        "pose_shift_threshold": pose_shift_threshold,
        "pose_score_mode": pose_score_mode,
        "pose_component_threshold": pose_component_threshold,
        "pose_pass": pose_pass,
        "pose_src": pose_src,
        "pose_tgt": pose_tgt,
        "vggt_token_key": token_key,
        "max_res": max_res,
        "device": device,
        "runtime_seconds": round(time.time() - start, 4),
        "judge_version": JUDGE_VERSION if input_meta.get("alignment_applied") else RAW_JUDGE_VERSION,
    }
    result.update(pose_component)
    result.update(input_meta)
    cleanup_cuda()
    return result



def aspan_prepass_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_keypoints": args.aspan_prepass_min_keypoints,
        "min_inliers": args.aspan_prepass_min_inliers,
        "min_inlier_ratio": args.aspan_prepass_min_inlier_ratio,
        "max_median_reprojection_error": args.aspan_prepass_max_median_reprojection_error,
        "min_overlap_fraction": args.aspan_prepass_min_overlap_fraction,
        "require_sanity": args.aspan_prepass_require_sanity,
        "require_metrics": args.aspan_prepass_require_metrics,
    }


def evaluate_aspan_prepass(input_meta: dict[str, Any], *, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    """Extremely lenient 2D alignment gate over ASpan homography metrics.

    The default is designed to let through almost everything except obvious bad
    2D geometry: too few matches/inliers, near-zero inlier ratio, wild reprojection
    error, nonsensical homography, or almost no warped-source overlap with target.
    Missing metrics pass by default because this is a lenient pre-pass, not a hard
    dependency, unless --aspan-prepass-require-metrics is set.
    """
    if mode == "off" or not input_meta.get("alignment_applied"):
        return {
            "aspan_2d_prepass_mode": mode,
            "aspan_2d_prepass_pass": True,
            "aspan_2d_prepass_fail_reasons": [],
            "aspan_2d_prepass_config": config,
        }

    fail_reasons: list[str] = []
    missing: list[str] = []

    def require_number(key: str):
        value = input_meta.get(key)
        if value is None:
            missing.append(key)
        return value

    raw_count = require_number("aspan_2d_raw_match_count")
    inliers = require_number("aspan_2d_homography_inliers")
    ratio = require_number("aspan_2d_inlier_ratio")
    median_error = require_number("aspan_2d_median_reprojection_error")
    overlap = require_number("aspan_2d_aligned_overlap_fraction")
    sanity = input_meta.get("aspan_2d_homography_sanity_pass")

    if config.get("require_metrics") and missing:
        fail_reasons.append("missing_metrics:" + ",".join(sorted(missing)))
    if raw_count is not None and raw_count < config["min_keypoints"]:
        fail_reasons.append(f"raw_match_count<{config['min_keypoints']} ({raw_count})")
    if inliers is not None and inliers < config["min_inliers"]:
        fail_reasons.append(f"homography_inliers<{config['min_inliers']} ({inliers})")
    if ratio is not None and ratio < config["min_inlier_ratio"]:
        fail_reasons.append(f"inlier_ratio<{config['min_inlier_ratio']} ({ratio:.4f})")
    if median_error is not None and median_error > config["max_median_reprojection_error"]:
        fail_reasons.append(f"median_reprojection_error>{config['max_median_reprojection_error']} ({median_error:.4f})")
    if overlap is not None and overlap < config["min_overlap_fraction"]:
        fail_reasons.append(f"aligned_overlap_fraction<{config['min_overlap_fraction']} ({overlap:.4f})")
    if config.get("require_sanity") and sanity is False:
        fail_reasons.append("homography_sanity_failed:" + str(input_meta.get("aspan_2d_homography_sanity_reason")))

    return {
        "aspan_2d_prepass_mode": mode,
        "aspan_2d_prepass_pass": len(fail_reasons) == 0,
        "aspan_2d_prepass_fail_reasons": fail_reasons,
        "aspan_2d_prepass_config": config,
    }


def aspan_prepass_rejection(input_meta: dict[str, Any], args: argparse.Namespace, device: str, pose_component_config: dict[str, float]) -> dict[str, Any]:
    judge = {
        "judgement": "rejected",
        "true_match": False,
        "reason": "aspan_2d_prepass_failed",
        "global_similarity": None,
        "global_pass": False,
        "pose_shift_total": None,
        "pose_shift_threshold": args.pose_shift_threshold,
        "pose_score_mode": args.pose_score_mode,
        "pose_component_threshold": args.pose_component_threshold,
        "pose_pass": False,
        "pose_src": None,
        "pose_tgt": None,
        "pose_component_config": pose_component_config,
        "pose_component_score": None,
        "pose_rotation_deg": None,
        "pose_translation_xy_l2": None,
        "pose_translation_z_abs": None,
        "pose_fov_l2": None,
        "pose_zoom_depth_fraction": None,
        "vggt_token_key": None,
        "max_res": args.max_res,
        "device": device,
        "runtime_seconds": 0.0,
        "vggt_skipped": True,
        "skip_reason": "aspan_2d_prepass_failed",
        "judge_version": JUDGE_VERSION if input_meta.get("alignment_applied") else RAW_JUDGE_VERSION,
    }
    judge.update(input_meta)
    return judge


def build_output_row(aspan_row: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    selected_aspan_keys = [
        "candidate_id",
        "source_id",
        "target_id",
        "source_path",
        "target_path",
        "rank",
        "similarity_score",
        "aspan_pass",
        "passed",
        "raw_keypoint_count",
        "filtered_keypoint_count",
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
    parser.add_argument("--pose-shift-threshold", type=float, default=0.10, help="Legacy raw pose_enc L1 threshold; still logged and used only with --pose-score-mode raw")
    parser.add_argument(
        "--pose-score-mode",
        choices=("component", "raw"),
        default="component",
        help="Pose decision mode. component uses the new weighted rotation/x-y/z/FOV heuristic; raw uses legacy pose_enc L1.",
    )
    parser.add_argument("--pose-component-threshold", type=float, default=3.0, help="Pass threshold for the new weighted component pose score")
    parser.add_argument("--pose-rotation-scale-deg", type=float, default=2.0, help="Rotation degrees that contribute 1.0 before weight/cap")
    parser.add_argument("--pose-xy-scale", type=float, default=0.020, help="x/y translation L2 that contributes 1.0 before weight/cap")
    parser.add_argument("--pose-z-scale", type=float, default=0.100, help="z translation magnitude that contributes 1.0 before weight/cap")
    parser.add_argument("--pose-fov-scale", type=float, default=0.100, help="FOV L2 delta that contributes 1.0 before weight/cap")
    parser.add_argument("--pose-rotation-weight", type=float, default=1.0)
    parser.add_argument("--pose-xy-weight", type=float, default=1.0)
    parser.add_argument("--pose-z-weight", type=float, default=0.25)
    parser.add_argument("--pose-fov-weight", type=float, default=0.25)
    parser.add_argument("--pose-term-cap", type=float, default=3.0, help="Per-component cap before weight is applied")
    parser.add_argument("--max-res", type=int, default=384)
    parser.add_argument(
        "--vggt-input-mode",
        choices=("aspan-aligned", "raw-paths"),
        default="aspan-aligned",
        help=(
            "Images fed to VGGT. Default reconstructs the ASpan homography alignment from "
            "sidecar NPZ data and passes in-memory PNG buffers to the official VGGT loader. "
            "Use raw-paths only as a debugging escape hatch."
        ),
    )
    parser.add_argument(
        "--alignment-keypoints",
        choices=("raw", "filtered", "auto"),
        default="raw",
        help="Sidecar keypoints used for homography. raw matches the Experiment notebook; auto tries raw then filtered.",
    )
    parser.add_argument(
        "--debug-aligned-dir",
        type=Path,
        default=None,
        help="Optional directory where per-candidate source_aligned.jpg/target.jpg debug renders are written.",
    )
    parser.add_argument(
        "--debug-inspect-dir",
        type=Path,
        default=None,
        help="If set, copies source and target images into per-source subfolders split "
             "by true_matches/ and rejected/ for human inspection.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Append only candidate_ids not already present in judged manifest")
    parser.add_argument("--allow-missing-pose", action="store_true", help="If pose_enc is absent, allow global similarity alone to pass")
    parser.add_argument(
        "--aspan-prepass-mode",
        choices=("lenient", "off"),
        default="lenient",
        help="Extremely lenient ASpan 2D geometry gate before VGGT. Use off to log metrics but never skip VGGT.",
    )
    parser.add_argument("--aspan-prepass-min-keypoints", type=int, default=8)
    parser.add_argument("--aspan-prepass-min-inliers", type=int, default=4)
    parser.add_argument("--aspan-prepass-min-inlier-ratio", type=float, default=0.05)
    parser.add_argument("--aspan-prepass-max-median-reprojection-error", type=float, default=30.0)
    parser.add_argument("--aspan-prepass-min-overlap-fraction", type=float, default=0.02)
    parser.add_argument("--aspan-prepass-require-sanity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aspan-prepass-require-metrics", action="store_true", help="By default missing ASpan metrics pass; set this to reject missing metrics")
    parser.add_argument(
        "--include-aspan-failed",
        action="store_true",
        help="Process rows even when aspan_pass/passed is false. Default skips failed aspanfilter rows.",
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
    print(f"VGGT input mode: {args.vggt_input_mode}")
    print(f"Alignment keypoints: {args.alignment_keypoints}")
    if args.debug_aligned_dir is not None:
        print(f"Debug aligned dir: {args.debug_aligned_dir}")
    if args.debug_inspect_dir is not None:
        print(f"Debug inspect dir: {args.debug_inspect_dir}")
    print("Judge: global register-token similarity + component-weighted pose_enc heuristic; no Tier-2 structural anomaly mask")
    print(f"Pose score mode: {args.pose_score_mode}")
    if args.pose_score_mode == "component":
        print(f"Pose component threshold: {args.pose_component_threshold}")
    else:
        print(f"Legacy raw pose shift threshold: {args.pose_shift_threshold}")

    pose_component_config = {
        "rotation_scale_deg": args.pose_rotation_scale_deg,
        "xy_scale": args.pose_xy_scale,
        "z_scale": args.pose_z_scale,
        "fov_scale": args.pose_fov_scale,
        "rotation_weight": args.pose_rotation_weight,
        "xy_weight": args.pose_xy_weight,
        "z_weight": args.pose_z_weight,
        "fov_weight": args.pose_fov_weight,
        "term_cap": args.pose_term_cap,
    }
    aspan_prepass_config = aspan_prepass_config_from_args(args)
    print(f"ASpan 2D pre-pass mode: {args.aspan_prepass_mode}")
    print(f"ASpan 2D pre-pass config: {aspan_prepass_config}")

    model = load_vggt_omega_model(args.checkpoint, device)

    counts = {
        "input_rows_considered": len(rows),
        "skipped_resume": 0,
        "judged": 0,
        "true_matches": 0,
        "rejected": 0,
        "errors": 0,
        "aspan_prepass_rejected": 0,
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
                image_inputs, input_meta = prepare_vggt_inputs(
                    aspan_row,
                    source_path,
                    target_path,
                    input_base_dir=input_manifest.parent,
                    candidate_id=candidate_id,
                    input_mode=args.vggt_input_mode,
                    alignment_keypoints=args.alignment_keypoints,
                    debug_aligned_dir=args.debug_aligned_dir,
                )
                prepass = evaluate_aspan_prepass(
                    input_meta,
                    mode=args.aspan_prepass_mode,
                    config=aspan_prepass_config,
                )
                input_meta.update(prepass)
                if not prepass["aspan_2d_prepass_pass"]:
                    counts["aspan_prepass_rejected"] += 1
                    judge = aspan_prepass_rejection(input_meta, args, device, pose_component_config)
                else:
                    judge = judge_pair(
                        model,
                        image_inputs,
                        input_meta,
                        device=device,
                        max_res=args.max_res,
                        global_sim_threshold=args.global_sim_threshold,
                        pose_shift_threshold=args.pose_shift_threshold,
                        pose_score_mode=args.pose_score_mode,
                        pose_component_threshold=args.pose_component_threshold,
                        pose_component_config=pose_component_config,
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
                    "pose_score_mode": args.pose_score_mode,
                    "pose_component_threshold": args.pose_component_threshold,
                    "pose_component_config": pose_component_config,
                    "pose_component_score": None,
                    "pose_rotation_deg": None,
                    "pose_translation_xy_l2": None,
                    "pose_translation_z_abs": None,
                    "pose_fov_l2": None,
                    "pose_zoom_depth_fraction": None,
                    "max_res": args.max_res,
                    "device": device,
                    "judge_version": JUDGE_VERSION if args.vggt_input_mode == "aspan-aligned" else RAW_JUDGE_VERSION,
                    "vggt_input_mode": "aspan_aligned_bytesio" if args.vggt_input_mode == "aspan-aligned" else "raw_paths",
                    "alignment_applied": False,
                    "aspan_2d_prepass_mode": args.aspan_prepass_mode,
                    "aspan_2d_prepass_pass": None,
                    "aspan_2d_prepass_fail_reasons": [],
                    "aspan_2d_prepass_config": aspan_prepass_config,
                    "aspan_2d_raw_match_count": None,
                    "aspan_2d_homography_inliers": None,
                    "aspan_2d_inlier_ratio": None,
                    "aspan_2d_mean_reprojection_error": None,
                    "aspan_2d_median_reprojection_error": None,
                    "aspan_2d_aligned_overlap_fraction": None,
                    "aspan_2d_homography_sanity_pass": None,
                    "aspan_2d_homography_sanity_reason": None,
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

            if args.debug_inspect_dir is not None:
                try:
                    _write_inspect_images(
                        args.debug_inspect_dir,
                        source_path,
                        target_path,
                        rank=int(aspan_row.get("rank") or 0),
                        true_match=bool(out_row.get("true_match")),
                    )
                except Exception as exc:
                    print(f"  [inspect] Warning: could not copy images for {candidate_id}: {exc}")

    summary = {
        **counts,
        "input_manifest": str(input_manifest),
        "judged_manifest": str(judged_path),
        "true_match_manifest": str(true_path),
        "checkpoint": str(args.checkpoint),
        "global_sim_threshold": args.global_sim_threshold,
        "pose_shift_threshold": args.pose_shift_threshold,
        "pose_score_mode": args.pose_score_mode,
        "pose_component_threshold": args.pose_component_threshold,
        "pose_component_config": pose_component_config,
        "aspan_prepass_mode": args.aspan_prepass_mode,
        "aspan_prepass_config": aspan_prepass_config,
        "max_res": args.max_res,
        "device": device,
        "allow_missing_pose": args.allow_missing_pose,
        "vggt_input_mode": args.vggt_input_mode,
        "alignment_keypoints": args.alignment_keypoints,
        "debug_aligned_dir": str(args.debug_aligned_dir) if args.debug_aligned_dir else None,
        "judge_version": JUDGE_VERSION if args.vggt_input_mode == "aspan-aligned" else RAW_JUDGE_VERSION,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
