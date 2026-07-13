"""Convert old per-source DINO JSON output to a single flat retrieval_manifest.jsonl.

The old pipeline saved one JSON file per source image:
    {stem}.json  →  {"source": "...", "1": {"filepath": "...", "similarity_score": 0.449}, ...}

The modern pipeline (retrieve.py) writes a single flat JSONL — one row per
(source, target) pair. This script reconstructs that format from the old JSONs.

No sharding is done here. Stage 2 (geometry filter) handles the shard split
inside stage2_ablation_colab.ipynb using the match_manifest CSVs.

Usage:
    python _local/convert_dino_output.py \\
        --old-dir "C:\\Users\\Jorda\\Downloads\\Working_DINO_5Epochs_output\\output" \\
        --output  _local/dino_retrieval_manifest.jsonl \\
        --top-k   10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert old per-source DINO JSONs to flat retrieval_manifest.jsonl"
    )
    p.add_argument("--old-dir", required=True,
                   help="Directory containing per-source .json files "
                        "(Working_DINO_5Epochs_output/output)")
    p.add_argument("--output", required=True,
                   help="Output JSONL path (e.g. _local/dino_retrieval_manifest.jsonl)")
    p.add_argument("--top-k", type=int, default=10,
                   help="Maximum candidates to keep per source (default 10)")
    return p.parse_args(argv)


def convert(args: argparse.Namespace) -> None:
    old_dir = Path(args.old_dir)
    output_path = Path(args.output)
    top_k = args.top_k

    if not old_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {old_dir}")

    json_files = sorted(old_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {old_dir}")

    print(f"Found {len(json_files):,} source JSON files in {old_dir}")
    print(f"Keeping top-{top_k} matches per source → writing {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    short_files = 0  # files with fewer than top_k entries
    error_files = 0

    metadata = {
        "converted_from": "old_per_source_json",
        "top_k": top_k,
        "old_dir": str(old_dir),
    }

    with output_path.open("w", encoding="utf-8") as out_f:
        for json_path in json_files:
            source_id = json_path.stem

            try:
                with json_path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  [warn] Could not read {json_path.name}: {exc}")
                error_files += 1
                continue

            # Collect ranked matches — keys are string integers "1", "2", ...
            ranked: list[tuple[int, str, float]] = []
            for key, val in data.items():
                if not key.isdigit():
                    continue
                rank = int(key)
                if rank > top_k:
                    continue
                filepath = val.get("filepath", "")
                score = float(val.get("similarity_score", 0.0))
                if filepath:
                    ranked.append((rank, filepath, score))

            ranked.sort(key=lambda x: x[0])

            if len(ranked) < top_k:
                short_files += 1

            for rank, filepath, score in ranked:
                target_id = Path(filepath).stem
                candidate_id = f"{source_id}__r{rank:03d}__{target_id}"
                row = {
                    "candidate_id": candidate_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_path": "",
                    "target_path": "",
                    "rank": rank,
                    "similarity_score": score,
                    "retrieval_metadata": metadata,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

    print(f"\nDone.")
    print(f"  Source files processed : {len(json_files) - error_files:,}")
    print(f"  Total rows written     : {total_rows:,}")
    print(f"  Files with < {top_k} matches : {short_files}")
    if error_files:
        print(f"  Files with errors      : {error_files}  (skipped)")
    print(f"  Output                 : {output_path}")


def main(argv=None) -> None:
    convert(parse_args(argv))


if __name__ == "__main__":
    main()
