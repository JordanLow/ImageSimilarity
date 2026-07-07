"""Step 4 of 4 — Decision layer for the NCR-Match pipeline.

Reads a vggt_judged_manifest.jsonl (output of vggt_signals.py) and applies
configurable threshold rules to produce final true/false match predictions:

  Filter 2 — aspan_2d_inlier_ratio >= INLIER_RATIO_THRESHOLD (0.65)
  Filter 3 — pose_component_score   <= POSE_COMPONENT_THRESHOLD (2.13)

Both thresholds were derived on Shard 1 (dev set) and applied frozen to Shard 2
(validation). The module-level constants below are the paper's reproducibility claim.

The script also accepts aspan_all_manifest.jsonl as input for CPU ablations that
do not require VGGT signals (rows without pose_component_score are handled
gracefully — the pose gate is skipped, or the pair is rejected if the gate is
enabled).

Ablation coverage (all via CLI flags, no code changes):
  B1 (keypoint sweep):    --input-manifest aspan_all_manifest.jsonl \\
                          --keypoint-floor N --pose-component-threshold 0
  B2 (inlier-ratio only): --pose-component-threshold 0
  B4 (full system/paper): python pose_scoring.py   # defaults reproduce paper
  B5 (rotation+xy only):  --pose-components rotation_xy
  B6 (fov+z only):        --pose-components fov_z
  vggt_judge baseline:    --global-sim-threshold 0.90 --pose-component-threshold 3.0

Acceptance test (definition of correctness):
  Shard 1: P=0.867, R=0.963, F1=0.913  (TP=313 FP=48 TN=268 FN=12)
  Shard 2: P=0.902, R=0.984, F1=0.941  (TP=248 FP=27 TN=359 FN= 4)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Paper's reproducibility constants ─────────────────────────────────────────
INLIER_RATIO_THRESHOLD   = 0.65   # Filter 2: aspan_2d_inlier_ratio must be ≥ this
POSE_COMPONENT_THRESHOLD = 2.13   # Filter 3: pose_component_score must be ≤ this
# global_similarity: recorded in the manifest but excluded from the decision rule
# (finding: huge distribution overlap; ~0.9 threshold barely separates anything)

SCORING_VERSION = "pose_scoring_v1"

# pose_component_terms keys in the vggt_signals.py output manifest
_COMPONENT_KEYS = {
    "rotation_xy": ("rotation", "translation_xy"),
    "fov_z":       ("translation_z", "fov"),
}


# ── Core decision logic ────────────────────────────────────────────────────────

def _compute_pose_score(row: dict[str, Any], pose_components: str) -> float | None:
    """Return the pose score to threshold against, or None if unavailable."""
    if pose_components == "all":
        return row.get("pose_component_score")
    keys = _COMPONENT_KEYS[pose_components]
    terms = row.get("pose_component_terms")
    if not terms:
        return None
    return sum(terms.get(k, 0.0) for k in keys)


def score_row(
    row: dict[str, Any],
    *,
    inlier_ratio_threshold: float,
    pose_component_threshold: float,
    global_sim_threshold: float | None,
    pose_components: str,
    keypoint_floor: int,
) -> tuple[bool, str]:
    """Apply decision rules to one manifest row.

    Returns (true_match, reason) where reason is a machine-readable slug.
    """
    # Keypoint floor (for B1 sweeps from aspan_all_manifest.jsonl)
    if keypoint_floor > 0:
        kp = row.get("raw_keypoint_count") or row.get("filtered_keypoint_count")
        if kp is None or int(kp) < keypoint_floor:
            return False, "keypoint_below_floor"

    # Filter 2: inlier ratio
    inlier_ratio = row.get("aspan_2d_inlier_ratio")
    if inlier_ratio is None:
        return False, "inlier_ratio_missing"
    if float(inlier_ratio) < inlier_ratio_threshold:
        return False, "inlier_ratio_below_threshold"

    # Optional: global similarity gate (disabled by default; for baseline comparison)
    if global_sim_threshold is not None:
        gs = row.get("global_similarity")
        if gs is None or float(gs) < global_sim_threshold:
            return False, "global_sim_below_threshold"

    # Filter 3: pose component score (pose_component_threshold=0 disables this gate)
    if pose_component_threshold > 0:
        if pose_components not in ("all", *_COMPONENT_KEYS):
            raise ValueError(f"Unknown --pose-components value: {pose_components!r}")
        if pose_components != "all" and "pose_component_terms" not in row:
            return False, "pose_component_terms_missing"
        pose_score = _compute_pose_score(row, pose_components)
        if pose_score is None:
            return False, "pose_component_score_missing"
        if float(pose_score) > pose_component_threshold:
            return False, "pose_component_score_above_threshold"

    return True, "pass"


def process_manifest(
    input_path: Path,
    output_path: Path,
    *,
    inlier_ratio_threshold: float,
    pose_component_threshold: float,
    global_sim_threshold: float | None,
    pose_components: str,
    keypoint_floor: int,
) -> dict[str, Any]:
    """Read input manifest, apply scoring rules, write output manifest.

    Returns a summary dict with counts and config.
    """
    config = {
        "scoring_version": SCORING_VERSION,
        "inlier_ratio_threshold": inlier_ratio_threshold,
        "pose_component_threshold": pose_component_threshold,
        "global_sim_threshold": global_sim_threshold,
        "pose_components": pose_components,
        "keypoint_floor": keypoint_floor,
    }

    counts: dict[str, int] = {}
    n_total = 0
    n_accepted = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_total += 1

            true_match, reason = score_row(
                row,
                inlier_ratio_threshold=inlier_ratio_threshold,
                pose_component_threshold=pose_component_threshold,
                global_sim_threshold=global_sim_threshold,
                pose_components=pose_components,
                keypoint_floor=keypoint_floor,
            )

            counts[reason] = counts.get(reason, 0) + 1
            if true_match:
                n_accepted += 1

            out_row = dict(row)
            out_row["true_match"] = true_match
            out_row["judgement"] = "ACCEPTED" if true_match else "REJECTED"
            out_row["reason"] = reason
            out_row["pose_scoring_config"] = config

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    return {
        "config": config,
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_total - n_accepted,
        "reason_counts": counts,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-manifest", required=True, metavar="JSONL",
        help="vggt_judged_manifest.jsonl from vggt_signals.py, or "
             "aspan_all_manifest.jsonl for B1/B2 ablations.",
    )
    p.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory for pose_scored_manifest.jsonl and pose_scoring_summary.json.",
    )
    p.add_argument(
        "--inlier-ratio-threshold", type=float, default=INLIER_RATIO_THRESHOLD,
        metavar="FLOAT",
        help=f"Filter 2: pairs below this aspan_2d_inlier_ratio are rejected. "
             f"Default: {INLIER_RATIO_THRESHOLD} (paper value).",
    )
    p.add_argument(
        "--pose-component-threshold", type=float, default=POSE_COMPONENT_THRESHOLD,
        metavar="FLOAT",
        help=f"Filter 3: pairs above this pose_component_score are rejected. "
             f"Set to 0 to disable the pose gate entirely (B2 ablation). "
             f"Default: {POSE_COMPONENT_THRESHOLD} (paper value).",
    )
    p.add_argument(
        "--global-sim-threshold", type=float, default=None, metavar="FLOAT",
        help="Optional additional gate: pairs below this global_similarity are rejected. "
             "Disabled by default (global_sim is not part of the paper's decision rule). "
             "Set to 0.90 with --pose-component-threshold 3.0 to reproduce the "
             "vggt_signals.py built-in baseline.",
    )
    p.add_argument(
        "--pose-components", default="all",
        choices=["all", "rotation_xy", "fov_z"],
        help="Which pose components to include in the score threshold check. "
             "all = use stored pose_component_score directly (default, paper value). "
             "rotation_xy = sum rotation+translation_xy terms only (B5 ablation). "
             "fov_z = sum translation_z+fov terms only (B6 ablation). "
             "Requires pose_component_terms in the input manifest.",
    )
    p.add_argument(
        "--keypoint-floor", type=int, default=0, metavar="INT",
        help="Reject pairs with raw_keypoint_count below this floor. "
             "Default: 0 (disabled). Use with --input-manifest aspan_all_manifest.jsonl "
             "and --pose-component-threshold 0 for B1 keypoint-sweep ablations.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path  = Path(args.input_manifest)
    output_dir  = Path(args.output_dir)
    output_path = output_dir / "pose_scored_manifest.jsonl"
    summary_path = output_dir / "pose_scoring_summary.json"

    if not input_path.exists():
        print(f"[pose_scoring] ERROR: input manifest not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[pose_scoring] Input:  {input_path}")
    print(f"[pose_scoring] Output: {output_path}")
    print(f"[pose_scoring] Config:")
    print(f"  inlier_ratio_threshold   = {args.inlier_ratio_threshold}")
    print(f"  pose_component_threshold = {args.pose_component_threshold}"
          + (" (DISABLED)" if args.pose_component_threshold == 0 else ""))
    print(f"  global_sim_threshold     = {args.global_sim_threshold!r}")
    print(f"  pose_components          = {args.pose_components}")
    print(f"  keypoint_floor           = {args.keypoint_floor}"
          + (" (DISABLED)" if args.keypoint_floor == 0 else ""))

    summary = process_manifest(
        input_path, output_path,
        inlier_ratio_threshold=args.inlier_ratio_threshold,
        pose_component_threshold=args.pose_component_threshold,
        global_sim_threshold=args.global_sim_threshold,
        pose_components=args.pose_components,
        keypoint_floor=args.keypoint_floor,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[pose_scoring] Results:")
    print(f"  Total rows : {summary['n_total']}")
    print(f"  Accepted   : {summary['n_accepted']}")
    print(f"  Rejected   : {summary['n_rejected']}")
    print(f"  By reason  :")
    for reason, count in sorted(summary["reason_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {reason}: {count}")
    print(f"\n[pose_scoring] Summary: {summary_path}")


if __name__ == "__main__":
    main()
