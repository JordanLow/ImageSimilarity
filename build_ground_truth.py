#!/usr/bin/env python3
"""
build_ground_truth.py

Consolidates two DIFFERENT provenance sources into ONE canonical ground-truth file:

1. `--shard` (human_review_full): the per-shard match_manifest_shard{N}.csv files, from
   _local/generate_match_manifest.py's systematic Positive/Negative/Unsure/Unknown human
   review. Only Positive/Negative are confirmed either way; Unsure/Unknown are excluded.
2. `--review-folder` (ablation_review_partial): the {sscd,clip,dinov3raw}_shard1_review/
   summary.csv exports -- pairs SSCD/CLIP/rawDINO surfaced that DINO's own retrieval never
   did. The Accepted/Rejected split there is pose_scoring.py's OWN automated decision, not an
   independent human judgment -- only `[Wrong]`-tagged subfolder names reflect a human actually
   looking. Confirmed decision rule (user-approved, 2026-07-19):
     - Accepted, no [Wrong] tag  -> Positive  (human reviewed the Accepted bucket, agreed)
     - Accepted, [Wrong]-tagged  -> Negative  (human caught a false positive)
     - Rejected (any)            -> Negative  (consistent with a prior "3TP/3FP/92TN"-scale
                                                human-review characterization for a related
                                                report)

Both sources get a `source` column so the two different confidence levels (systematic full
review vs. automated-decision-plus-spot-check) stay distinguishable downstream.

Adds source_id/target_id columns using the exact same join-key derivation every other
consumer in this project uses (source_folder, Path(target_image).stem -- see
graph_assembly.py's load_labels / ablation_utils.load_ground_truth), so this file is directly
joinable against shard_membership/ and ablation_results/ without re-deriving the key.

Usage:
    python build_ground_truth.py \
        --shard "Shard1=D:/DINO OUTPUTS/match_manifest_shard1.csv" \
        --shard "Shard2=D:/DINO OUTPUTS/match_manifest_shard2.csv" \
        --review-folder "SSCD=D:/DINO OUTPUTS/sscd_shard1_review" \
        --review-folder "CLIP=D:/DINO OUTPUTS/clip_shard1_review" \
        --review-folder "DINOv3Raw=D:/DINO OUTPUTS/dinov3raw_shard1_review" \
        --output NCR/ground_truth/ground_truth.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

CONFIRMED = {"Positive", "Negative"}


def load_match_manifest_rows(spec: str) -> tuple[str, list[dict]]:
    shard_name, path = spec.split("=", 1)
    rows: list[dict] = []
    excluded_by_status: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            classification = row["classification"].strip()
            if classification not in CONFIRMED:
                excluded_by_status[classification] = excluded_by_status.get(classification, 0) + 1
                continue
            rows.append({
                "shard": shard_name,
                "source_id": row["source_folder"],
                "target_id": Path(row["target_image"]).stem,
                "classification": classification,
                "source": "human_review_full",
                "source_folder": row["source_folder"],
                "source_image": row["source_image"],
                "target_image": row["target_image"],
                "reviewed_matches_dirname": row.get("reviewed_matches_dirname", ""),
                "notes": row.get("notes", ""),
            })
    return shard_name, rows, excluded_by_status


def find_wrong_tagged_pairs(review_dir: Path) -> set[tuple[str, str]]:
    """Folder names look like '002_<source_id>__<target_id> [Wrong]' under Accepted/ or
    Rejected/. Returns the set of (source_id, target_id) pairs with at least one such folder."""
    wrong: set[tuple[str, str]] = set()
    for bucket in ("Accepted", "Rejected"):
        bucket_dir = review_dir / bucket
        if not bucket_dir.is_dir():
            continue
        for name in os.listdir(bucket_dir):
            if not name.endswith("[Wrong]"):
                continue
            stem = name[: -len("[Wrong]")].strip()
            # strip the leading numeric prefix, e.g. "002_"
            if "_" in stem:
                stem = stem.split("_", 1)[1]
            if "__" not in stem:
                continue
            source_id, target_id = stem.split("__", 1)
            wrong.add((source_id, target_id))
    return wrong


def load_review_folder_rows(spec: str) -> list[dict]:
    model_tag, path = spec.split("=", 1)
    review_dir = Path(path)
    wrong_pairs = find_wrong_tagged_pairs(review_dir)
    rows: list[dict] = []
    with open(review_dir / "summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source_id, target_id = row["source_id"], row["target_id"]
            pipeline_decision = row["classification"].strip()
            is_wrong = (source_id, target_id) in wrong_pairs
            if pipeline_decision == "Accepted":
                classification = "Negative" if is_wrong else "Positive"
            else:
                classification = "Negative"
            rows.append({
                "shard": "Shard1",
                "source_id": source_id,
                "target_id": target_id,
                "classification": classification,
                "source": "ablation_review_partial",
                "source_folder": source_id,
                "source_image": f"{source_id}.jpg",
                "target_image": f"{target_id}.jpg",
                "reviewed_matches_dirname": "",
                "notes": (f"surfaced by {model_tag}, rank {row.get('rank', '?')}; "
                          f"pipeline={pipeline_decision} ({row.get('reason', '')})"
                          + (", human-flagged [Wrong]" if is_wrong else "")),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard", action="append", default=[],
                         help="ShardName=path to a match_manifest_shard{N}.csv (repeatable)")
    parser.add_argument("--review-folder", action="append", default=[],
                         help="ModelTag=path to a {model}_shard1_review directory (repeatable)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_rows: list[dict] = []
    excluded_by_status: dict[str, int] = {}

    for spec in args.shard:
        _, rows, excl = load_match_manifest_rows(spec)
        out_rows.extend(rows)
        for k, v in excl.items():
            excluded_by_status[k] = excluded_by_status.get(k, 0) + v

    for spec in args.review_folder:
        out_rows.extend(load_review_folder_rows(spec))

    out_rows.sort(key=lambda r: (r["shard"], r["source_id"], r["target_id"], r["source"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["shard", "source_id", "target_id", "classification", "source",
                  "source_folder", "source_image", "target_image",
                  "reviewed_matches_dirname", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_positive = sum(1 for r in out_rows if r["classification"] == "Positive")
    n_negative = sum(1 for r in out_rows if r["classification"] == "Negative")
    n_full = sum(1 for r in out_rows if r["source"] == "human_review_full")
    n_partial = sum(1 for r in out_rows if r["source"] == "ablation_review_partial")
    print(f"{len(out_rows)} confirmed rows -> {out_path}")
    print(f"  Positive: {n_positive}")
    print(f"  Negative: {n_negative}")
    print(f"  source=human_review_full: {n_full}")
    print(f"  source=ablation_review_partial: {n_partial}")
    if excluded_by_status:
        print(f"  Excluded from match_manifest rows (not confirmed either way): {excluded_by_status}")


if __name__ == "__main__":
    main()
