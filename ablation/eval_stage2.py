"""Stage 2 CPU evaluation — geometry filter ablation (C1-LightGlue, C2-RoMa).

Loads vggt_judged_manifest.jsonl files produced by the Colab run, applies frozen
pose_scoring thresholds, and computes F1 / McNemar vs B4 for each matcher.

Run after copying the Colab outputs to D:\\DINO OUTPUTS\\.

Usage:
    python ablation/eval_stage2.py \\
        --c1-shard1 "D:\\DINO OUTPUTS\\c1_lightglue_shard1.jsonl" \\
        --c1-shard2 "D:\\DINO OUTPUTS\\c1_lightglue_shard2.jsonl" \\
        --c2-shard1 "D:\\DINO OUTPUTS\\c2_roma_shard1.jsonl" \\
        --c2-shard2 "D:\\DINO OUTPUTS\\c2_roma_shard2.jsonl" \\
        --b4-shard1 "D:\\DINO OUTPUTS\\Shard1 Judge Manifest.jsonl" \\
        --b4-shard2 "D:\\DINO OUTPUTS\\Shard2 Judge Manifest.jsonl" \\
        --gt-shard1 "D:\\DINO OUTPUTS\\match_manifest_shard1.csv" \\
        --gt-shard2 "D:\\DINO OUTPUTS\\match_manifest_shard2.csv" \\
        --output    "D:\\DINO OUTPUTS\\stage2_geometry_results.json"

Skips any variant whose manifest paths are not provided (pass --c2-shard1 etc. only
when those files are available — e.g. if only C1 is done so far).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── Repo imports ──────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))              # pose_scoring is at repo root
sys.path.insert(0, str(Path(__file__).parent))  # ablation_utils is in ablation/

from ablation_utils import (
    load_ground_truth, compute_metrics, EXCLUDE_LABELS,
)
from pose_scoring import score_row, INLIER_RATIO_THRESHOLD, POSE_COMPONENT_THRESHOLD

# ── Statistical helpers (inlined to avoid importing statistics.py's side-effects) ──

N_BOOT    = 10_000
BOOT_SEED = 42
ALPHA     = 0.05


def _chi2_sf_1dof(stat: float) -> float:
    if stat <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(stat / 2.0))


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided McNemar p-value. Uses exact binomial for b+c < 25, chi-sq otherwise."""
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        # Exact: 2 * sum_{k=0}^{min(b,c)} C(n,k) / 2^n
        from math import comb
        lo = min(b, c)
        p = sum(comb(n, k) for k in range(lo + 1)) / (2 ** n)
        return min(1.0, 2 * p)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return _chi2_sf_1dof(stat)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values (same order as input)."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (n - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def bootstrap_f1_ci(y_true: list[int], y_pred: list[int],
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                    alpha: float = ALPHA) -> tuple[float, float]:
    """Bootstrap percentile 95% CI for F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(yt[idx].tolist(), yp[idx].tolist())
        f1s.append(m["f1"] if not math.isnan(m["f1"]) else 0.0)
    lo = float(np.percentile(f1s, 100 * alpha / 2))
    hi = float(np.percentile(f1s, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Data loading ──────────────────────────────────────────────────────────────

def load_judged_manifest(path: Path) -> dict[tuple[str, str], dict]:
    """Returns {(source_id, target_id_stem): row} for every line."""
    rows: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("source_id", ""))
            tid = str(row.get("target_id", ""))
            # normalise: strip .jpg extension if present so keys match GT stems
            tid_stem = Path(tid).stem if tid.endswith(".jpg") else tid
            rows[(sid, tid_stem)] = row
    return rows


def apply_pose_scoring(rows: dict[tuple[str, str], dict]) -> set[tuple[str, str]]:
    """Returns the set of (source_id, target_id_stem) accepted by pose_scoring."""
    accepted: set[tuple[str, str]] = set()
    for key, row in rows.items():
        result = score_row(row)
        if result.get("true_match"):
            accepted.add(key)
    return accepted


def build_prediction_vector(
    gt: dict[tuple[str, str], str],
    accepted: set[tuple[str, str]],
) -> tuple[list[int], list[int]]:
    """
    For every (Positive|Negative) pair in GT, emit (y_true, y_pred).
    Pairs not in `accepted` are treated as predicted Negative.
    """
    y_true, y_pred = [], []
    for key, label in gt.items():
        if label.lower() in EXCLUDE_LABELS:
            continue
        if label not in ("Positive", "Negative"):
            continue
        y_true.append(1 if label == "Positive" else 0)
        y_pred.append(1 if key in accepted else 0)
    return y_true, y_pred


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_variant(
    name: str,
    shard1_path: Path | None,
    shard2_path: Path | None,
    gt_s1: dict,
    gt_s2: dict,
    b4_accepted_s1: set,
    b4_accepted_s2: set,
) -> dict[str, Any] | None:
    """Evaluate a single matcher variant across both shards."""
    if shard1_path is None and shard2_path is None:
        return None

    results: dict[str, Any] = {"name": name, "shards": {}}
    all_yt, all_yp, all_b4 = [], [], []

    for shard_name, path, gt, b4_acc in [
        ("Shard1", shard1_path, gt_s1, b4_accepted_s1),
        ("Shard2", shard2_path, gt_s2, b4_accepted_s2),
    ]:
        if path is None:
            print(f"  [{name}] {shard_name}: skipped (no manifest provided)")
            continue
        print(f"  [{name}] {shard_name}: loading {path.name} ...")
        rows    = load_judged_manifest(path)
        accepted = apply_pose_scoring(rows)
        yt, yp  = build_prediction_vector(gt, accepted)
        _, b4p  = build_prediction_vector(gt, b4_acc)
        m = compute_metrics(yt, yp)
        ci_lo, ci_hi = bootstrap_f1_ci(yt, yp)

        # McNemar contingency: pairs where predictions differ
        b = sum(1 for a, b_ in zip(yp, b4p) if a == 1 and b_ == 0)
        c = sum(1 for a, b_ in zip(yp, b4p) if a == 0 and b_ == 1)
        p_raw = mcnemar_p(b, c)

        shard_result = {
            **m,
            "f1_ci_lo": ci_lo, "f1_ci_hi": ci_hi,
            "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_p_raw": p_raw,
            "n_accepted": len(accepted),
            "n_gt_pairs": len(yt),
        }
        results["shards"][shard_name] = shard_result
        all_yt.extend(yt)
        all_yp.extend(yp)
        all_b4.extend(b4p)
        print(
            f"    F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}"
            f"  [{ci_lo:.3f}–{ci_hi:.3f}]  McNemar raw p={p_raw:.4f}"
        )

    if all_yt:
        m_all = compute_metrics(all_yt, all_yp)
        ci_lo, ci_hi = bootstrap_f1_ci(all_yt, all_yp)
        b = sum(1 for a, b_ in zip(all_yp, all_b4) if a == 1 and b_ == 0)
        c = sum(1 for a, b_ in zip(all_yp, all_b4) if a == 0 and b_ == 1)
        p_raw = mcnemar_p(b, c)
        results["combined"] = {
            **m_all,
            "f1_ci_lo": ci_lo, "f1_ci_hi": ci_hi,
            "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_p_raw": p_raw,
        }

    return results


# ── Markdown table ────────────────────────────────────────────────────────────

def render_table(b4_combined: dict, variants: list[dict]) -> str:
    lines = [
        "| Variant | Shard | F1 | 95% CI | McNemar p (Holm) |",
        "|---|---|---|---|---|",
    ]
    b4_f1 = b4_combined.get("f1", float("nan"))

    def row_line(label: str, shard: str, m: dict, p_holm: float | None) -> str:
        f1    = m.get("f1", float("nan"))
        ci_lo = m.get("f1_ci_lo", float("nan"))
        ci_hi = m.get("f1_ci_hi", float("nan"))
        marker = " ▼" if (not math.isnan(f1) and f1 < b4_f1) else ""
        ci_str = f"[{ci_lo:.3f}–{ci_hi:.3f}]"
        p_str  = f"{p_holm:.4f}" if p_holm is not None else "—"
        return f"| {label}{marker} | {shard} | {f1:.3f} | {ci_str} | {p_str} |"

    lines.append(row_line("B4 (ASpanFormer)", "Combined", b4_combined, None))
    for v in variants:
        for shard_name, shard_data in v.get("shards", {}).items():
            p_holm = shard_data.get("mcnemar_p_holm")
            lines.append(row_line(v["name"], shard_name, shard_data, p_holm))
        if "combined" in v:
            p_holm = v["combined"].get("mcnemar_p_holm")
            lines.append(row_line(v["name"], "Combined", v["combined"], p_holm))

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Stage 2 geometry filter ablation evaluation")
    p.add_argument("--c1-shard1", help="C1 (LightGlue) vggt_judged_manifest — Shard 1")
    p.add_argument("--c1-shard2", help="C1 (LightGlue) vggt_judged_manifest — Shard 2")
    p.add_argument("--c2-shard1", help="C2 (RoMa) vggt_judged_manifest — Shard 1")
    p.add_argument("--c2-shard2", help="C2 (RoMa) vggt_judged_manifest — Shard 2")
    p.add_argument("--b4-shard1", required=True, help="B4 vggt_judged_manifest — Shard 1")
    p.add_argument("--b4-shard2", required=True, help="B4 vggt_judged_manifest — Shard 2")
    p.add_argument("--gt-shard1", required=True, help="match_manifest_shard1.csv")
    p.add_argument("--gt-shard2", required=True, help="match_manifest_shard2.csv")
    p.add_argument("--output",    required=True, help="JSON output path")
    args = p.parse_args(argv)

    def _path(v: str | None) -> Path | None:
        return Path(v) if v else None

    print("Loading ground truth ...")
    gt_s1 = load_ground_truth(Path(args.gt_shard1))
    gt_s2 = load_ground_truth(Path(args.gt_shard2))

    print("Loading B4 manifests ...")
    b4_rows_s1 = load_judged_manifest(Path(args.b4_shard1))
    b4_rows_s2 = load_judged_manifest(Path(args.b4_shard2))
    b4_acc_s1  = apply_pose_scoring(b4_rows_s1)
    b4_acc_s2  = apply_pose_scoring(b4_rows_s2)

    yt_b4, yp_b4 = [], []
    for gt, acc in [(gt_s1, b4_acc_s1), (gt_s2, b4_acc_s2)]:
        yt, yp = build_prediction_vector(gt, acc)
        yt_b4.extend(yt)
        yp_b4.extend(yp)
    b4_combined = {**compute_metrics(yt_b4, yp_b4)}
    ci_lo, ci_hi = bootstrap_f1_ci(yt_b4, yp_b4)
    b4_combined.update({"f1_ci_lo": ci_lo, "f1_ci_hi": ci_hi})
    print(f"  [B4] Combined F1={b4_combined['f1']:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]")

    variant_defs = [
        ("C1 (LightGlue)", _path(args.c1_shard1), _path(args.c1_shard2)),
        ("C2 (RoMa)",      _path(args.c2_shard1), _path(args.c2_shard2)),
    ]

    print("\nEvaluating variants ...")
    variants: list[dict] = []
    raw_ps: list[float]  = []
    for name, s1, s2 in variant_defs:
        result = evaluate_variant(name, s1, s2, gt_s1, gt_s2, b4_acc_s1, b4_acc_s2)
        if result is not None:
            variants.append(result)
            if "combined" in result:
                raw_ps.append(result["combined"]["mcnemar_p_raw"])

    # Holm-Bonferroni correction across all tested variants (combined results)
    holm_ps = holm_bonferroni(raw_ps)
    vi = 0
    for v in variants:
        if "combined" in v:
            v["combined"]["mcnemar_p_holm"] = holm_ps[vi]
            # propagate to shards too (informational)
            for shard_data in v["shards"].values():
                shard_data["mcnemar_p_holm"] = None  # per-shard not Holm-corrected
            vi += 1

    print("\n── Table B (geometry filter ablation) ──")
    table_md = render_table(b4_combined, variants)
    print(table_md)

    output = {
        "b4": b4_combined,
        "variants": variants,
        "thresholds": {
            "inlier_ratio": INLIER_RATIO_THRESHOLD,
            "pose_component": POSE_COMPONENT_THRESHOLD,
        },
        "bootstrap_n": N_BOOT,
        "holm_bonferroni_n": len(raw_ps),
        "table_md": table_md,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
