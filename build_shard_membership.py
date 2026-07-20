#!/usr/bin/env python3
"""
build_shard_membership.py

Builds the default/raw source-shard assignment for all 6 corpus shards -- pure structural
fact (which physical DINO_Output_Shard{N} folder a source belongs to), independent of any
labeling/review status and independent of any family-cohesion/leakage-prevention logic.
NEG-Net's leakage prevention is handled entirely by NCR/negnet_training_exposure/, keyed on
node_id exposure -- this file has no bearing on that and is not consumed by it.

Two source-identity extraction methods, since Shard 1/2 and Shard 3-6 have different data
available:
- Shard 1/2: distinct source_id values in the reconstructed candidate-edge manifest
  (aspan_all_manifest_shard{N}_reconstructed.jsonl).
- Shard 3-6: no reconstructed manifest exists. Source identity comes directly from
  DINO_Output_Shard{N}/visualizations/ -- one subfolder per source, named by source_id
  (each containing a match_log.json, not read here -- only the subfolder name is used).

No family-cohesion logic, no labeled-graph input, no correction/precedence layer -- this is a
deliberately simpler replacement for the deleted apply_shard_reassignment.py /
build_full_source_partition.py, which combined raw identity with a labeled-graph override.
That override existed only to keep training/eval shards node-disjoint in advance; NEG-Net's
leakage prevention no longer needs that (see NCR/negnet_training_exposure/README.md), so
there is nothing left for a precedence layer to resolve.

A source is tie-broken to the lower-numbered shard if it's found under more than one shard's
detection method (same convention used throughout this project) -- expected to be rare.

Usage:
    python build_shard_membership.py \
        --reconstructed-manifest "Shard1=D:/DINO OUTPUTS/aspan_all_manifest_shard1_reconstructed.jsonl" \
        --reconstructed-manifest "Shard2=D:/DINO OUTPUTS/aspan_all_manifest_shard2_reconstructed.jsonl" \
        --visualizations-folder "Shard3=D:/DINO OUTPUTS/DINO_Output_Shard3/visualizations" \
        --visualizations-folder "Shard4=D:/DINO OUTPUTS/DINO_Output_Shard4/visualizations" \
        --visualizations-folder "Shard5=D:/DINO OUTPUTS/DINO_Output_Shard5/visualizations" \
        --visualizations-folder "Shard6=D:/DINO OUTPUTS/DINO_Output_Shard6/visualizations" \
        --output NCR/shard_membership/source_shard_assignment.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def sources_from_reconstructed_manifest(path: str) -> set[str]:
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid = json.loads(line).get("source_id")
            if sid:
                seen.add(sid)
    return seen


def sources_from_visualizations_folder(path: str) -> set[str]:
    root = Path(path)
    return {p.name for p in root.iterdir() if p.is_dir()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reconstructed-manifest", action="append", default=[],
                         help="ShardName=path to a *_reconstructed.jsonl (Shard 1/2 style, repeatable)")
    parser.add_argument("--visualizations-folder", action="append", default=[],
                         help="ShardName=path to a DINO_Output_Shard{N}/visualizations/ folder "
                              "(Shard 3-6 style, repeatable)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_membership: dict[str, list[str]] = defaultdict(list)
    per_shard_counts: dict[str, int] = {}

    for spec in args.reconstructed_manifest:
        shard_name, path = spec.split("=", 1)
        sources = sources_from_reconstructed_manifest(path)
        per_shard_counts[shard_name] = len(sources)
        for sid in sources:
            raw_membership[sid].append(shard_name)

    for spec in args.visualizations_folder:
        shard_name, path = spec.split("=", 1)
        sources = sources_from_visualizations_folder(path)
        per_shard_counts[shard_name] = len(sources)
        for sid in sources:
            raw_membership[sid].append(shard_name)

    assignment: dict[str, dict] = {}
    n_ties = 0
    for sid, shards in raw_membership.items():
        uniq = sorted(set(shards))
        chosen = uniq[0]  # tie-break: lower-numbered shard, same convention as elsewhere
        is_tie = len(uniq) > 1
        if is_tie:
            n_ties += 1
        assignment[sid] = {"shard": chosen, "tie": is_tie}

    shard_totals: dict[str, int] = defaultdict(int)
    for v in assignment.values():
        shard_totals[v["shard"]] += 1

    output = {
        "purpose": (
            "Default/raw source-shard assignment for all 6 corpus shards -- pure structural "
            "fact (which physical DINO_Output_Shard{N} folder a source belongs to), no "
            "family-cohesion or leakage-prevention logic involved. NEG-Net's leakage "
            "prevention (NCR/negnet_training_exposure/) is independent of shard identity "
            "entirely and does not consume this file."
        ),
        "generated_from": {
            "reconstructed_manifests": args.reconstructed_manifest,
            "visualizations_folders": args.visualizations_folder,
        },
        "n_sources": len(assignment),
        "n_ties_tiebroken": n_ties,
        "shard_totals": dict(sorted(shard_totals.items())),
        "assignment": {sid: assignment[sid] for sid in sorted(assignment)},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"{len(assignment)} sources -> {out_path}")
    print("Per-shard detection counts (before tie-break):")
    for shard, count in sorted(per_shard_counts.items()):
        print(f"  {shard}: {count}")
    print("Final shard totals (after tie-break):")
    for shard, count in sorted(shard_totals.items()):
        print(f"  {shard}: {count}")
    print(f"  ties tiebroken: {n_ties}")


if __name__ == "__main__":
    main()
