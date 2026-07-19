"""Losses and calibration utilities for NEG-Net (note §3 / §5.2).

Tier 0 (default here, per the 2026-07-18 review): BCE anchors on labeled edges
plus consistency penalties with the two details that matter —

- Triangle penalty with DETACHED premises: hinge(sg[p_ik * p_kj] - p_ij)^2.
  The gradient only lifts the low edge p_ij; the premise edges are stop-grads,
  so the model cannot satisfy the constraint by dragging good edges down.
- Normalized over ACTIVE constraints (those with positive violation), not over
  all enumerated paths, so the penalty scale does not vanish as the number of
  satisfied triangles grows.

Also included:
- Nesting penalty hinge(p_N - p_S)^2 (N <= S facet), same normalization.
- prior_shift_logits: standard class-prior-shift logit correction, applied
  BEFORE any thresholding, calibration, or conformal step. Training-graph
  positive rate (Shard 1: 0.507) is far above validation (Shard 2: 0.395) and
  deployment rates; without the shift, downstream calibration is
  systematically off and the accept/review/reject layer inherits the bias.

The perturbed Fenchel–Young loss over Y(V) (note §3, default instantiation)
plugs in behind --loss fy once nested_cc.py lands; Tier 0 is the flag-guarded
fallback with the same architecture.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bce_anchor_loss(logits: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    """BCE over labeled edges only. labels in {0,1}; mask selects labeled edges."""
    if mask.sum() == 0:
        return logits.new_zeros(())
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask].float(), pos_weight=pw)


def triangle_penalty(probs: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """triangles: (T, 3) long — columns are edge indices (ij, ik, kj).

    penalty = mean over ACTIVE constraints of relu(sg[p_ik * p_kj] - p_ij)^2.
    """
    if triangles.numel() == 0:
        return probs.new_zeros(())
    p_ij = probs[triangles[:, 0]]
    premise = (probs[triangles[:, 1]] * probs[triangles[:, 2]]).detach()
    violation = F.relu(premise - p_ij)
    active = violation > 0
    if active.sum() == 0:
        return probs.new_zeros(())
    return (violation[active] ** 2).mean()


def nesting_penalty(p_exposure: torch.Tensor, p_scene: torch.Tensor) -> torch.Tensor:
    """Same-exposure implies same-scene: penalize relu(p_N - sg[p_S])^2 over
    active constraints. The scene premise is detached for symmetry with the
    triangle term — the gradient pushes the offending N edge down rather than
    inflating S."""
    violation = F.relu(p_exposure - p_scene.detach())
    active = violation > 0
    if active.sum() == 0:
        return p_exposure.new_zeros(())
    return (violation[active] ** 2).mean()


def tier0_loss(logit_s: torch.Tensor, logit_n: torch.Tensor,
               labels_n: torch.Tensor, mask_n: torch.Tensor,
               labels_s: torch.Tensor, mask_s: torch.Tensor,
               triangles_n: torch.Tensor,
               lambda_tri: float = 1.0, lambda_nest: float = 1.0,
               pos_weight: float | None = None) -> dict[str, torch.Tensor]:
    """Full Tier-0 objective. Scene labels are sparse pre-audit (¬N conflates
    S\\N with U), so mask_s is typically a subset of mask_n until the negative
    audit lands."""
    p_s, p_n = torch.sigmoid(logit_s), torch.sigmoid(logit_n)
    loss_bce_n = bce_anchor_loss(logit_n, labels_n, mask_n, pos_weight)
    loss_bce_s = bce_anchor_loss(logit_s, labels_s, mask_s, pos_weight)
    loss_tri = triangle_penalty(p_n, triangles_n)
    loss_nest = nesting_penalty(p_n, p_s)
    total = loss_bce_n + loss_bce_s + lambda_tri * loss_tri + lambda_nest * loss_nest
    return {"total": total, "bce_n": loss_bce_n, "bce_s": loss_bce_s,
            "triangle": loss_tri, "nesting": loss_nest}


def fy_loss(*args, **kwargs):
    """Perturbed Fenchel–Young loss over Y(V) — requires the nested correlation
    clustering decoder (nested_cc.py). Deliberately unimplemented in Tier 0."""
    raise NotImplementedError(
        "FY loss needs nested_cc.py (greedy CC + KL moves); run Tier 0 first.")


# ── Prior-shift correction (before thresholding / calibration / conformal) ─────

def prior_shift_logits(logits: torch.Tensor, train_pos_rate: float,
                       deploy_pos_rate: float) -> torch.Tensor:
    """Standard class-prior-shift correction (saerens-style logit adjustment):

        logit' = logit + log(deploy_odds) - log(train_odds)

    Example: Shard-1 training prior 0.507 → Shard-2 prior 0.395 subtracts
    ~0.455 from every logit before any threshold is applied."""
    def _log_odds(p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError(f"positive rate must be in (0, 1), got {p}")
        return math.log(p / (1.0 - p))

    return logits + (_log_odds(deploy_pos_rate) - _log_odds(train_pos_rate))
