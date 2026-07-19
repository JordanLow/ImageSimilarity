"""NEG-Net — the learned decision model for the NCR-Match graph (note §5.2).

Architecture (defaults per the 2026-07-18 review):
- Edge features: fixed-length summary statistics extracted from the frozen
  evidence vector g_ij (see EDGE_FEATURES), standardized per-dim with
  Shard-1 training-fold statistics, then frozen. Missing values are imputed
  to the training mean (0 after standardization) with per-feature missing
  indicators plus one edge-level missing_evidence flag.
- Node features: pre-head 3024-d backbone concat (dump_prehead_features.py),
  LayerNorm + linear projection to the hidden width.
- Edge encoder: 2-layer MLP at width 256 — with --mp-rounds 0 this IS the
  pair-MLP baseline (main-table attribution row: hand rule → pair-MLP →
  +message passing → +consistency loss).
- Message passing: L rounds (default 3) of edge/node updates with residuals.
- Heads: two linear potential heads — theta_S (same scene), theta_N (same
  exposure). Nesting p_N <= p_S is enforced by the loss, not the architecture.

Total parameters at defaults: ~2.0M (print with `python negnet.py`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ── Edge feature extraction ────────────────────────────────────────────────────
# (key, transform) — dotted keys reach into nested dicts. Counts are log1p'd.

_LOG = "log1p"
_ID = "identity"

EDGE_FEATURES: list[tuple[str, str]] = [
    ("aspan_2d_raw_match_count", _LOG),
    ("aspan_2d_homography_inliers", _LOG),
    ("aspan_2d_inlier_ratio", _ID),
    ("aspan_2d_mean_reprojection_error", _LOG),
    ("aspan_2d_median_reprojection_error", _LOG),
    ("aspan_2d_aligned_overlap_fraction", _ID),
    ("alignment_keypoint_count", _LOG),
    ("global_similarity", _ID),
    ("pose_rotation_deg", _ID),
    ("pose_translation_xy_l2", _ID),
    ("pose_translation_z_abs", _ID),
    ("pose_fov_l2", _ID),
    ("pose_zoom_depth_fraction", _ID),
    ("pose_component_score", _ID),
    ("pose_component_terms.rotation", _ID),
    ("pose_component_terms.translation_xy", _ID),
    ("pose_component_terms.translation_z", _ID),
    ("pose_component_terms.fov", _ID),
]
N_HOMOGRAPHY_DIMS = 8  # 3x3 homography / H[2,2], last entry dropped
EDGE_DIM = len(EDGE_FEATURES) + N_HOMOGRAPHY_DIMS          # raw values
EDGE_INPUT_DIM = 2 * EDGE_DIM + 1                          # + indicators + edge flag


def _get_dotted(d: dict[str, Any] | None, key: str) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def extract_edge_features(evidence: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (values[EDGE_DIM], missing[EDGE_DIM]) from a g_ij evidence dict.
    An entirely missing evidence dict (closure edges) yields all-missing."""
    values = np.zeros(EDGE_DIM, dtype=np.float32)
    missing = np.ones(EDGE_DIM, dtype=np.float32)
    if evidence is None:
        return values, missing
    for i, (key, transform) in enumerate(EDGE_FEATURES):
        raw = _get_dotted(evidence, key)
        if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
            continue
        val = float(raw)
        if transform == _LOG:
            val = math.log1p(max(val, 0.0))
        values[i], missing[i] = val, 0.0
    h = evidence.get("alignment_homography")
    if h is not None:
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        if h.size == 9 and np.isfinite(h).all() and abs(h[8]) > 1e-12:
            h = h / h[8]
            base = len(EDGE_FEATURES)
            values[base:base + N_HOMOGRAPHY_DIMS] = h[:8].astype(np.float32)
            missing[base:base + N_HOMOGRAPHY_DIMS] = 0.0
    return values, missing


class Standardizer:
    """Per-dim mean/std fitted on Shard-1 TRAINING folds only, then frozen.
    Missing entries are excluded from the fit and imputed to the mean (0 after
    standardization); their indicators carry the missingness signal."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean, self.std = mean, std

    def fit(self, values: np.ndarray, missing: np.ndarray) -> "Standardizer":
        present = missing < 0.5
        mean = np.zeros(values.shape[1], dtype=np.float64)
        std = np.ones(values.shape[1], dtype=np.float64)
        for j in range(values.shape[1]):
            col = values[present[:, j], j]
            if col.size:
                mean[j] = col.mean()
                std[j] = max(col.std(), 1e-6)
        self.mean, self.std = mean, std
        return self

    def transform(self, values: np.ndarray, missing: np.ndarray,
                  edge_missing: np.ndarray) -> np.ndarray:
        z = (values - self.mean) / self.std
        z = np.where(missing < 0.5, z, 0.0)
        return np.concatenate(
            [z, missing, edge_missing[:, None]], axis=1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"mean": self.mean.tolist(), "std": self.std.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(blob["mean"]), np.asarray(blob["std"]))


# ── Model ──────────────────────────────────────────────────────────────────────

def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])  # no trailing ReLU


class NEGNet(nn.Module):
    def __init__(self, node_dim: int = 3024, hidden: int = 256,
                 edge_input_dim: int = EDGE_INPUT_DIM, mp_rounds: int = 3):
        super().__init__()
        self.mp_rounds = mp_rounds
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_proj = nn.Linear(node_dim, hidden)
        # Stage A edge encoder — 2-layer MLP at width `hidden`.
        self.edge_encoder = _mlp([edge_input_dim, hidden, hidden])
        self.edge_updates = nn.ModuleList(
            _mlp([3 * hidden, hidden]) for _ in range(mp_rounds))
        self.node_updates = nn.ModuleList(
            _mlp([2 * hidden, hidden]) for _ in range(mp_rounds))
        self.readout = _mlp([3 * hidden, hidden])
        self.head_scene = nn.Linear(hidden, 1)      # theta_S
        self.head_exposure = nn.Linear(hidden, 1)   # theta_N

    def forward(self, node_x: torch.Tensor, edge_x: torch.Tensor,
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """node_x: (N, node_dim); edge_x: (E, edge_input_dim);
        edge_index: (2, E) long — returns (logit_S, logit_N), each (E,)."""
        h = self.node_proj(self.node_norm(node_x))
        e = self.edge_encoder(edge_x)
        src, dst = edge_index[0], edge_index[1]
        for edge_up, node_up in zip(self.edge_updates, self.node_updates):
            e = e + edge_up(torch.cat([e, h[src], h[dst]], dim=1))
            # Mean-aggregate incident edge messages (edges are undirected:
            # scatter to both endpoints).
            agg = torch.zeros_like(h)
            cnt = torch.zeros(h.shape[0], 1, device=h.device)
            for idx in (src, dst):
                agg = agg.index_add(0, idx, e)
                cnt = cnt.index_add(0, idx, torch.ones(len(idx), 1, device=h.device))
            agg = agg / cnt.clamp(min=1.0)
            h = h + node_up(torch.cat([h, agg], dim=1))
        r = self.readout(torch.cat([e, h[src], h[dst]], dim=1))
        return self.head_scene(r).squeeze(-1), self.head_exposure(r).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for rounds in (0, 3):
        model = NEGNet(mp_rounds=rounds)
        print(f"mp_rounds={rounds}: edge_input_dim={EDGE_INPUT_DIM}, "
              f"params={count_parameters(model):,}")
