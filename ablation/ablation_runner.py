"""Task 1 — Stage 1 ablation runner (B1, B2, B4, B5, B6).

Sweeps decision rules over the stored manifests and emits a structured
ablation_results.json covering all free ablation rows. Run this first;
statistics.py, sensitivity_curves.py, and pose_signal_analysis.py all read
its output.

Usage:
    python ablation/ablation_runner.py
    python ablation/ablation_runner.py --shards Shard1   # one shard only

Outputs:
    D:/DINO OUTPUTS/ablation_results.json
    D:/DINO OUTPUTS/ablation_table.md          (human-readable summary table)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))              # pose_scoring is at repo root
sys.path.insert(0, str(Path(__file__).parent))  # ablation_utils is in ablation/
from ablation_utils import (  # noqa: E402
    DINO_ROOT, SHARDS, load_ground_truth, load_aspan_all,
    compute_metrics, pr_auc, EXCLUDE_LABELS,
)
from pose_scoring import score_row, INLIER_RATIO_THRESHOLD, POSE_COMPONENT_THRESHOLD  # noqa: E402

OUTPUT_JSON = DINO_ROOT / "ablation_results.json"
OUTPUT_MD   = DINO_ROOT / "ablation_table.md"

# Keypoint sweep parameters for B1 — sweep dynamically to max observed kp
KP_SWEEP_START = 0
KP_SWEEP_STEP  = 25

ABLATION_ROWS = {
    "B1": "Keypoint-count filter only (sweep 0-500)",
    "B2": "Inlier-ratio only (no pose gate)",
    "B4": "Full pipeline — paper defaults (inlier>=0.65, pose<=2.13)",
    "B5": "Pose: rotation+xy translation only",
    "B6": "Pose: fov+z translation only",
}


# ── Per-shard evaluation ───────────────────────────────────────────────────────

def eval_b1(shard_name: str) -> dict:
    """B1: sweep keypoint floor. Uses aspan_all_manifest for the full range."""
    paths = SHARDS[shard_name]
    gt = load_ground_truth(paths["manifest_csv"])
    aspan_all = load_aspan_all(paths["aspan_all"])

    # Build (label, keypoint_count) for all labeled pairs
    labeled: list[tuple[str, int]] = []
    for (sid, tid), label in gt.items():
        if label.lower() in EXCLUDE_LABELS or label not in ("Positive", "Negative"):
            continue
        row = aspan_all.get((sid, tid))
        if row is None:
            continue
        kp = row.get("raw_keypoint_count") or row.get("filtered_keypoint_count", 0)
        labeled.append((label, int(kp)))

    y_true = [1 if lbl == "Positive" else 0 for lbl, _ in labeled]
    scores  = [kp for _, kp in labeled]

    # PR-AUC over the full continuous keypoint-count score
    auc = pr_auc(y_true, scores)

    # Sweep from 0 to just above the max observed keypoint count
    max_kp = max(scores) if scores else 500
    sweep_stop = int(max_kp) + KP_SWEEP_STEP
    sweep = []
    for floor in range(KP_SWEEP_START, sweep_stop, KP_SWEEP_STEP):
        y_pred = [1 if kp >= floor else 0 for _, kp in labeled]
        m = compute_metrics(y_true, y_pred)
        m["keypoint_floor"] = floor
        sweep.append(m)

    best = max(sweep, key=lambda m: m["f1"] if not np.isnan(m["f1"]) else -1)

    # Closest sweep point at kp=50 (original breakpoint)
    closest_50 = min(sweep, key=lambda m: abs(m["keypoint_floor"] - 50))
    at_50 = next((m for m in sweep if m["keypoint_floor"] == 50), closest_50)

    print(f"  [B1/{shard_name}] labeled={len(labeled)} PR-AUC={auc:.4f} "
          f"best F1={best['f1']:.3f} @ kp>={best['keypoint_floor']}")
    return {
        "pr_auc": auc,
        "n_labeled": len(labeled),
        "sweep": sweep,
        "best": best,
        "at_kp50": at_50,
    }


def eval_pose_variant(
    shard_name: str,
    row_label: str,
    *,
    inlier_ratio_threshold: float = INLIER_RATIO_THRESHOLD,
    pose_component_threshold: float = POSE_COMPONENT_THRESHOLD,
    pose_components: str = "all",
    global_sim_threshold: float | None = None,
    keypoint_floor: int = 0,
) -> dict:
    """B2, B4, B5, B6: apply a fixed scoring config to the vggt-judged rows."""
    paths = SHARDS[shard_name]
    gt = load_ground_truth(paths["manifest_csv"])
    aspan_all = load_aspan_all(paths["aspan_all"])

    y_true, y_pred, scores = [], [], []
    n_no_gt = n_no_vggt = 0

    for (sid, tid), row in aspan_all.items():
        # Only evaluate pairs that have VGGT signals (aspan_2d_inlier_ratio present)
        if "aspan_2d_inlier_ratio" not in row:
            n_no_vggt += 1
            continue
        label = gt.get((sid, tid))
        if label is None or label.lower() in EXCLUDE_LABELS:
            n_no_gt += 1
            continue
        if label not in ("Positive", "Negative"):
            continue

        predicted, _ = score_row(
            row,
            inlier_ratio_threshold=inlier_ratio_threshold,
            pose_component_threshold=pose_component_threshold,
            global_sim_threshold=global_sim_threshold,
            pose_components=pose_components,
            keypoint_floor=keypoint_floor,
        )
        y_true.append(1 if label == "Positive" else 0)
        y_pred.append(1 if predicted else 0)

        # Soft score for PR-AUC: primary continuous signal per variant
        if row_label == "B2":
            scores.append(row.get("aspan_2d_inlier_ratio") or 0.0)
        else:
            # For B4/B5/B6: combined soft score = inlier_ratio minus normalised pose score
            ir = row.get("aspan_2d_inlier_ratio") or 0.0
            ps = row.get("pose_component_score") or 9.0
            scores.append(ir - ps / 10.0)

    m = compute_metrics(y_true, y_pred)
    auc = pr_auc(y_true, scores)
    m["pr_auc"] = auc
    m["n_evaluated"] = len(y_true)
    m["n_no_gt"] = n_no_gt
    m["n_no_vggt"] = n_no_vggt

    print(f"  [{row_label}/{shard_name}] n={len(y_true)} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
          f"PR-AUC={auc:.4f}")
    return m


# ── Output formatting ──────────────────────────────────────────────────────────

def format_cell(m: dict, key: str, digits: int = 3) -> str:
    v = m.get(key, float("nan"))
    if isinstance(v, float) and np.isnan(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown(results: dict, out_path: Path) -> None:
    lines = [
        "# Ablation Results — Table B (Stage 1)",
        "",
        f"Generated: {date.today()}",
        f"Paper defaults: inlier_ratio >= {INLIER_RATIO_THRESHOLD}, "
        f"pose_component_score <= {POSE_COMPONENT_THRESHOLD}",
        "",
        "## Point-estimate metrics (no CIs — run statistics.py to add CIs and McNemar)",
        "",
    ]

    shards = list(SHARDS.keys())
    header = "| Row | Description |" + "".join(
        f" {s} P | {s} R | {s} F1 | {s} PR-AUC |" for s in shards
    )
    sep = "|---|---|" + "".join("|---|---|---|---|" for _ in shards)
    lines += [header, sep]

    for row_label, desc in ABLATION_ROWS.items():
        row_data = results.get(row_label, {})
        cells = [f"**{row_label}**", desc]
        for shard in shards:
            m = row_data.get(shard, {})
            if row_label == "B1":
                best = m.get("best", {})
                cells += [
                    format_cell(best, "precision"),
                    format_cell(best, "recall"),
                    format_cell(best, "f1"),
                    format_cell(m, "pr_auc"),
                ]
            else:
                cells += [
                    format_cell(m, "precision"),
                    format_cell(m, "recall"),
                    format_cell(m, "f1"),
                    format_cell(m, "pr_auc"),
                ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "### B1 best operating point (by F1)", ""]
    for shard in shards:
        b1 = results.get("B1", {}).get(shard, {})
        best = b1.get("best", {})
        lines.append(
            f"- **{shard}**: keypoint_floor={best.get('keypoint_floor')}  "
            f"TP={best.get('tp')} FP={best.get('fp')} "
            f"TN={best.get('tn')} FN={best.get('fn')}  "
            f"P={format_cell(best,'precision')} R={format_cell(best,'recall')} "
            f"F1={format_cell(best,'f1')}"
        )

    lines += ["", "### B4 full detail (TP/FP/TN/FN per shard)", ""]
    for shard in shards:
        b4 = results.get("B4", {}).get(shard, {})
        lines.append(
            f"- **{shard}**: TP={b4.get('tp')} FP={b4.get('fp')} "
            f"TN={b4.get('tn')} FN={b4.get('fn')}  "
            f"P={format_cell(b4,'precision')} R={format_cell(b4,'recall')} "
            f"F1={format_cell(b4,'f1')} PR-AUC={format_cell(b4,'pr_auc')}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown table written to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", default=list(SHARDS.keys()),
                        choices=list(SHARDS.keys()),
                        help="Which shards to evaluate (default: all)")
    args = parser.parse_args(argv)

    results: dict[str, dict] = {}

    for row_label, desc in ABLATION_ROWS.items():
        print(f"\n[{row_label}] {desc}")
        results[row_label] = {}
        for shard in args.shards:
            if row_label == "B1":
                results[row_label][shard] = eval_b1(shard)
            elif row_label == "B2":
                results[row_label][shard] = eval_pose_variant(
                    shard, "B2",
                    pose_component_threshold=0.0,
                )
            elif row_label == "B4":
                results[row_label][shard] = eval_pose_variant(shard, "B4")
            elif row_label == "B5":
                results[row_label][shard] = eval_pose_variant(
                    shard, "B5",
                    pose_components="rotation_xy",
                )
            elif row_label == "B6":
                results[row_label][shard] = eval_pose_variant(
                    shard, "B6",
                    pose_components="fov_z",
                )

    # Add metadata
    output = {
        "metadata": {
            "date": str(date.today()),
            "inlier_ratio_threshold": INLIER_RATIO_THRESHOLD,
            "pose_component_threshold": POSE_COMPONENT_THRESHOLD,
            "shards_evaluated": args.shards,
            "kp_sweep_range": f"{KP_SWEEP_START}-max_observed step {KP_SWEEP_STEP}",
        },
        "rows": results,
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results written to {OUTPUT_JSON}")

    write_markdown(results, OUTPUT_MD)


if __name__ == "__main__":
    main()
