"""Batched ASpanFormer inference for geometry_filter.py.

geometry_filter.py's default path (--batch-size 1, unchanged) calls the
matcher with online_resize=True, one image pair per GPU forward pass. That
mode is asserted to batch=1 upstream because it computes a per-image-adaptive
positional-encoding calibration (see ml-aspanformer/src/ASpanFormer/
aspanformer.py:40-42,113-124) -- a convenience wrapper never written to
handle more than one image at a time.

This module reimplements the *orchestration* of ASpanFormer.forward() (same
file, lines 30-102) to run B independent pairs through one forward pass,
while preserving that exact per-image calibration semantics rather than
switching to the model's alternative fixed-encoding online_resize=False path:

- Each image's own resize_df crop and positional-encoding pos_scale are
  computed exactly as resize_input() does (same formulas, same bound
  `matcher.resize_df` method reused directly -- not reimplemented), just
  evaluated once per image before padding/stacking instead of once per
  single-image call.
- The positional encoding itself is produced by calling the model's own
  PositionEncodingSine.forward() once per image in a cheap Python loop (a
  tensor fill + one add -- not the backbone or attention transformer, which
  ARE batched below) and concatenated, so the sinusoidal-encoding formula is
  never reimplemented, only looped.
- Padding to a common canvas is masked out via the same mask0/mask1 ->
  mask_c0/mask_c1 mechanism CoarseMatching already supports and the upstream
  MegaDepth training/eval pipeline already exercises (coarse_matching.py's
  mask_border_with_padding); geometry_filter.py's single-pair path just never
  populated it before, since a lone image needs no padding against anything.
  IMPORTANT (found via local smoke-testing, not just static reading): passing
  a mask at all -- even a trivially all-True one -- routes through a
  genuinely different code path inside the attention transformer
  (aspan_module/transformer.py:166-168 downsamples the mask via max_pool2d
  before it reaches attention), which is NOT numerically equivalent to
  mask=None and measurably changed keypoint counts in local testing. So masks
  are only ever passed when a chunk actually contains real padding (i.e. not
  every image on a given side already shares the same H,W) -- at batch size 1
  this is never true, so batch=1 takes the exact same mask=None path today's
  code does, byte-for-byte.
- The final keypoint rescale (undoing resize_df's floor-to-32 crop) is
  applied per-pair via a b_ids/m_bids gather, mirroring the same
  data['scale0'][b_ids] indexing pattern coarse_matching.py already uses
  elsewhere in this same file, rather than aspanformer.py's blanket
  single-image multiply.
- ds0/ds1 (aspanformer.py:75-76) are a hardcoded [4,4] constant under
  online_resize=True, independent of image content or batch size -- carried
  over unchanged, no patch needed.

geometry_filter.py remains the single source of truth for image I/O, the
long_dim resize, and the RANSAC step -- this module only replaces the single
GPU forward-pass call, and returns the same per-pair dict shape
run_aspan_pair() does, so downstream handling doesn't need to know which path
produced it.

Shape bucketing (compute_resized_shape / bucket_candidates below): local CPU
testing found that padding two *different*-shaped images into one batch
produces real, non-trivial keypoint corruption near the padded boundary --
not a shape bug, but a mask-unaware bilinear upsample inside the vendored
flow_initializer (ml-aspanformer/.../transformer.py:177-180) blending
padded/zero content into nearby valid cells. Rather than patch that (deep,
hard to fully verify without GPU access), batching here only ever groups
candidates whose source/target images already resolve to the exact same
(h,w) after resize_with_scale + resize_df -- so _pad_and_stack's mask=None
path (the one Correctness A already validates) is the ONLY path ever
exercised, for any batch size, not just B=1. Because resize_with_scale always
pins the long edge to exactly `long_dim` and resize_df floors the short edge
to a multiple of 32, the space of possible shapes is small and bounded
(at most long_dim/32 short-edge values per orientation) -- real candidates
cluster into a manageable number of buckets rather than each needing its own
singleton batch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ResizeWithScale = Callable[[np.ndarray, int], tuple[np.ndarray, float]]
KeypointsToOriginal = Callable[[np.ndarray, float], np.ndarray]
RunRansac = Callable[[np.ndarray, np.ndarray], dict[str, Any]]

ShapeKey = tuple[int, int]
BucketKey = tuple[ShapeKey, ShapeKey]


def compute_resized_shape(
    path: str, long_dim: int, resize_with_scale: ResizeWithScale
) -> ShapeKey | None:
    """The (h, w) a source/target image will end up as after resize_with_scale +
    resize_df -- i.e. its actual shape inside the batch tensor. Decodes the full
    image (not just a header peek) via the same cv2.imread(path, 0) grayscale
    read run_aspan_pair/run_aspan_batch use, so this can never silently disagree
    with what real processing later sees (e.g. from an EXIF-orientation
    discrepancy a header-only read could introduce). Returns None if the image
    can't be read -- callers should route that candidate to the same per-pair
    error handling as a missing/corrupt image anywhere else in this pipeline,
    not drop it silently.
    """
    gray = cv2.imread(path, 0)
    if gray is None:
        return None
    resized, _ = resize_with_scale(gray, long_dim)
    h, w = resized.shape[:2]
    df = 32  # mirrors resize_df's floor arithmetic exactly (aspanformer.py:126-133)
    return (h // df * df, w // df * df)


def bucket_candidates(
    rows: Iterable[dict[str, Any]],
    long_dim: int,
    resize_with_scale: ResizeWithScale,
) -> tuple[dict[BucketKey, list[dict[str, Any]]], list[tuple[dict[str, Any], str]]]:
    """Group candidate rows by (source_shape, target_shape) so every resulting
    group can be batched through _pad_and_stack's mask=None path -- the one
    Correctness A already validates -- regardless of batch size. Returns
    (buckets, unresolved) where unresolved is a list of (row, error_message)
    for candidates whose source/target image couldn't be read, isolated up
    front rather than silently dropped or left to fail mid-batch.

    Per-image shapes are cached by path, since the same source/target image
    typically recurs across many candidate rows (top-K retrieval per source).
    """
    shape_cache: dict[str, ShapeKey | None] = {}

    def shape_of(path: str) -> ShapeKey | None:
        if path not in shape_cache:
            shape_cache[path] = compute_resized_shape(path, long_dim, resize_with_scale)
        return shape_cache[path]

    buckets: dict[BucketKey, list[dict[str, Any]]] = defaultdict(list)
    unresolved: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        source_path = row.get("source_path")
        target_path = row.get("target_path")
        if not source_path or not target_path:
            unresolved.append((row, "candidate row missing source_path or target_path"))
            continue
        source_shape = shape_of(source_path)
        target_shape = shape_of(target_path)
        if source_shape is None or target_shape is None:
            bad_path = source_path if source_shape is None else target_path
            unresolved.append((row, f"FileNotFoundError: Could not read image: {bad_path}"))
            continue
        buckets[(source_shape, target_shape)].append(row)

    return dict(buckets), unresolved


def _prepare_single_image(
    matcher,
    gray_uint8: np.ndarray,
    long_dim: int,
    device: torch.device,
    resize_with_scale: ResizeWithScale,
) -> dict[str, Any]:
    """Per-image prep mirroring run_aspan_pair's resize_with_scale call plus
    aspanformer.py's resize_input()/resize_df() -- same formulas and the same
    bound resize_df method, evaluated per image so results can be padded and
    stacked into a batch afterward."""
    resized, scale = resize_with_scale(gray_uint8, long_dim)
    h0, w0 = resized.shape[:2]

    tensor = torch.from_numpy(resized / 255.0)[None, None].to(device).float()  # [1,1,h0,w0]
    tensor = matcher.resize_df(tensor)[0]  # -> [1,h_new,w_new]; reuses the exact bound method
    h_new, w_new = tensor.shape[-2], tensor.shape[-1]

    train_res = matcher.config["coarse"]["train_res"]
    if isinstance(train_res, (list, tuple)):
        train_res_h, train_res_w = train_res[0], train_res[1]
    else:
        train_res_h = train_res_w = train_res
    pos_scale = [train_res_h / h_new, train_res_w / w_new]  # [y_scale, x_scale], matches aspanformer.py:121-122
    online_resize_scale = [w0 / w_new, h0 / h_new]  # [x_scale, y_scale], matches aspanformer.py:123-124

    return {
        "tensor": tensor,
        "resize_scale": scale,
        "pos_scale": pos_scale,
        "online_resize_scale": online_resize_scale,
        "resized_size": (w0, h0),  # pre-resize_df size == run_aspan_pair's *_resized_size
    }


def _pad_and_stack(tensors: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Zero-pad a list of [1,h,w] tensors to a common canvas and stack to
    [B,1,H,W]. Returns a matching [B,H,W] boolean valid-region mask (True
    where real pixels are, False where padded) -- or None if every tensor
    already shared the same H,W, meaning no padding actually happened (this
    is always the case at B=1). Passing a mask at all changes behavior (see
    module docstring), so callers must only pass on a real mask, never a
    trivially-all-True one."""
    max_h = max(t.shape[-2] for t in tensors)
    max_w = max(t.shape[-1] for t in tensors)
    if all(t.shape[-2] == max_h and t.shape[-1] == max_w for t in tensors):
        return torch.cat([t.unsqueeze(0) for t in tensors], dim=0), None

    batch = torch.zeros((len(tensors), 1, max_h, max_w), dtype=torch.float32, device=device)
    mask = torch.zeros((len(tensors), max_h, max_w), dtype=torch.bool, device=device)
    for i, t in enumerate(tensors):
        h, w = t.shape[-2], t.shape[-1]
        batch[i, :, :h, :w] = t
        mask[i, :h, :w] = True
    return batch, mask


def _forward_batch(
    matcher,
    image0: torch.Tensor,
    image1: torch.Tensor,
    mask0: torch.Tensor | None,
    mask1: torch.Tensor | None,
    pos_scale0: list[list[float]],
    pos_scale1: list[list[float]],
) -> dict[str, Any]:
    """Batched equivalent of ASpanFormer.forward(data, online_resize=True)
    minus the batch=1 assert (aspanformer.py:30-102). image0/image1:
    [B,1,H,W], already padded to a common canvas. mask0/mask1: [B,H,W]
    boolean pixel-level padding masks, or None if this chunk has no real
    padding to mask (always true at B=1) -- passing a mask at all measurably
    changes behavior (see module docstring), so it's only ever supplied when
    genuinely needed. Returns the data dict with mkpts0_f/mkpts1_f already
    rescaled per-pair by online_resize_scale, plus m_bids for splitting
    results back out per pair."""
    assert (mask0 is None) == (mask1 is None), (
        "mask0/mask1 must both be None (no padding on either side) or both real -- "
        "callers rely on shape-bucketing to guarantee this; an asymmetric case here "
        "would mean bucketing only matched one side's shape, which should be impossible."
    )
    data: dict[str, Any] = {"image0": image0, "image1": image1}
    bs = image0.shape[0]
    data["bs"] = bs
    data["hw0_i"], data["hw1_i"] = image0.shape[2:], image1.shape[2:]

    # 1. Backbone (aspanformer.py:52-59). Every image is padded to one shared
    # canvas, so hw0_i == hw1_i always holds and this takes the concat
    # branch -- a compute-efficiency choice only: the backbone runs in
    # eval() mode (geometry_filter.py:118), so BatchNorm uses fixed running
    # stats and each sample's output is independent of what shares its batch
    # or whether calls are fused.
    if data["hw0_i"] == data["hw1_i"]:
        feats_c, feats_f = matcher.backbone(torch.cat([image0, image1], dim=0))
        (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(bs), feats_f.split(bs)
    else:
        (feat_c0, feat_f0), (feat_c1, feat_f1) = matcher.backbone(image0), matcher.backbone(image1)

    data["hw0_c"], data["hw1_c"] = feat_c0.shape[2:], feat_c1.shape[2:]
    data["hw0_f"], data["hw1_f"] = feat_f0.shape[2:], feat_f1.shape[2:]

    # 2. Per-image positional encoding. Calls the model's own
    # PositionEncodingSine.forward() once per image -- a cheap tensor fill,
    # not the backbone or attention transformer above/below, which stay
    # fully batched -- so the sinusoidal formula itself is never
    # reimplemented, only looped and concatenated.
    feat_c0_list, pe0_list, feat_c1_list, pe1_list = [], [], [], []
    for b in range(bs):
        fc0, pe0 = matcher.pos_encoding(feat_c0[b : b + 1], pos_scale0[b])
        fc1, pe1 = matcher.pos_encoding(feat_c1[b : b + 1], pos_scale1[b])
        feat_c0_list.append(fc0)
        pe0_list.append(pe0)
        feat_c1_list.append(fc1)
        pe1_list.append(pe1)
    feat_c0 = torch.cat(feat_c0_list, dim=0)
    feat_c1 = torch.cat(feat_c1_list, dim=0)
    pos_encoding0 = torch.cat(pe0_list, dim=0)
    pos_encoding1 = torch.cat(pe1_list, dim=0)

    # 3. ds0/ds1: hardcoded constant under online_resize=True
    # (aspanformer.py:75-76), independent of image content or batch size.
    ds0 = ds1 = [4, 4]

    if mask0 is None:
        # No real padding in this chunk (always true at B=1) -- take the
        # exact same mask=None path today's single-pair code does.
        mask_c0_flat = mask_c1_flat = None
    else:
        # Coarse-resolution padding mask -- same interpolate-to-coarse-shape
        # pattern the upstream MegaDepth batched pipeline uses
        # (ml-aspanformer/src/datasets/megadepth.py:119-125), sized exactly to
        # this batch's actual feature-map shape (data['hw0_c']) rather than a
        # hardcoded downsample ratio.
        mask_c0 = F.interpolate(mask0.float().unsqueeze(1), size=data["hw0_c"], mode="nearest").squeeze(1).bool()
        mask_c1 = F.interpolate(mask1.float().unsqueeze(1), size=data["hw1_c"], mode="nearest").squeeze(1).bool()
        mask_c0_flat, mask_c1_flat = mask_c0.flatten(-2), mask_c1.flatten(-2)
        # get_coarse_match's mask_border_with_padding reads data['mask0']/
        # data['mask1'] directly (unflattened) -- coarse_matching.py:185-189.
        data["mask0"], data["mask1"] = mask_c0, mask_c1

    feat_c0, feat_c1, flow_list = matcher.loftr_coarse(
        feat_c0, feat_c1, pos_encoding0, pos_encoding1, mask_c0_flat, mask_c1_flat, ds0, ds1
    )
    matcher.coarse_matching(feat_c0, feat_c1, flow_list, data, mask_c0=mask_c0_flat, mask_c1=mask_c1_flat)

    feat_f0_unfold, feat_f1_unfold = matcher.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
    if feat_f0_unfold.size(0) != 0:
        feat_f0_unfold, feat_f1_unfold = matcher.loftr_fine(feat_f0_unfold, feat_f1_unfold)
    matcher.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    return data


def run_aspan_batch(
    matcher,
    rows: list[dict[str, Any]],
    long_dim: int,
    device: torch.device,
    resize_with_scale: ResizeWithScale,
    keypoints_to_original: KeypointsToOriginal,
    run_ransac: RunRansac,
) -> list[dict[str, Any]]:
    """Batched equivalent of calling run_aspan_pair() once per row in `rows`.

    Returns a list the same length and order as `rows`. Each element is
    either a match dict with exactly the keys run_aspan_pair() returns, or
    {"error": "ExceptionType: message"} if that row's images failed to load
    -- isolated before the GPU batch is built, same per-pair failure
    semantics as today's try/except in process().
    """
    results: list[dict[str, Any] | None] = [None] * len(rows)
    valid_indices: list[int] = []
    p0_list: list[dict[str, Any]] = []
    p1_list: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            source_path = row.get("source_path")
            target_path = row.get("target_path")
            if not source_path or not target_path:
                raise ValueError("candidate row missing source_path or target_path")
            img0_color = cv2.imread(source_path)
            img1_color = cv2.imread(target_path)
            img0_gray = cv2.imread(source_path, 0)
            img1_gray = cv2.imread(target_path, 0)
            if img0_color is None or img1_color is None or img0_gray is None or img1_gray is None:
                raise FileNotFoundError(f"Could not read source or target image: {source_path}, {target_path}")

            h0, w0 = img0_gray.shape[:2]
            h1, w1 = img1_gray.shape[:2]
            p0 = _prepare_single_image(matcher, img0_gray, long_dim, device, resize_with_scale)
            p1 = _prepare_single_image(matcher, img1_gray, long_dim, device, resize_with_scale)

            valid_indices.append(idx)
            p0_list.append(p0)
            p1_list.append(p1)
            results[idx] = {
                "source_original_size": [int(w0), int(h0)],
                "target_original_size": [int(w1), int(h1)],
            }
        except Exception as exc:
            results[idx] = {"error": f"{type(exc).__name__}: {exc}"}

    if valid_indices:
        image0, mask0 = _pad_and_stack([p["tensor"] for p in p0_list], device)
        image1, mask1 = _pad_and_stack([p["tensor"] for p in p1_list], device)
        pos_scale0 = [p["pos_scale"] for p in p0_list]
        pos_scale1 = [p["pos_scale"] for p in p1_list]
        online_resize_scale0 = torch.tensor(
            [p["online_resize_scale"] for p in p0_list], device=device, dtype=torch.float32
        )
        online_resize_scale1 = torch.tensor(
            [p["online_resize_scale"] for p in p1_list], device=device, dtype=torch.float32
        )

        with torch.no_grad():
            data = _forward_batch(matcher, image0, image1, mask0, mask1, pos_scale0, pos_scale1)
            m_bids = data["m_bids"]
            # Per-pair rescale via a b_ids-style gather, mirroring
            # coarse_matching.py:250-251's data['scale0'][b_ids] pattern,
            # replacing aspanformer.py:101-102's blanket single-image multiply.
            mkpts0_f = data["mkpts0_f"] * online_resize_scale0[m_bids]
            mkpts1_f = data["mkpts1_f"] * online_resize_scale1[m_bids]
            raw0_all = mkpts0_f.detach().cpu().numpy().astype(np.float32)
            raw1_all = mkpts1_f.detach().cpu().numpy().astype(np.float32)
            m_bids_all = m_bids.detach().cpu().numpy()

        for local_i, orig_idx in enumerate(valid_indices):
            sel = m_bids_all == local_i
            raw0 = raw0_all[sel]
            raw1 = raw1_all[sel]
            p0, p1 = p0_list[local_i], p1_list[local_i]
            ransac = run_ransac(raw0, raw1)

            row_result = results[orig_idx]
            row_result.update(
                {
                    "source_resized_size": [int(p0["resized_size"][0]), int(p0["resized_size"][1])],
                    "target_resized_size": [int(p1["resized_size"][0]), int(p1["resized_size"][1])],
                    "source_scale": float(p0["resize_scale"]),
                    "target_scale": float(p1["resize_scale"]),
                    "raw_mkpts0_resized": raw0,
                    "raw_mkpts1_resized": raw1,
                    "raw_mkpts0_original": keypoints_to_original(raw0, p0["resize_scale"]),
                    "raw_mkpts1_original": keypoints_to_original(raw1, p1["resize_scale"]),
                    "filtered_mkpts0_original": keypoints_to_original(
                        ransac["filtered_mkpts0_resized"], p0["resize_scale"]
                    ),
                    "filtered_mkpts1_original": keypoints_to_original(
                        ransac["filtered_mkpts1_resized"], p1["resize_scale"]
                    ),
                }
            )
            row_result.update(ransac)

    return results
