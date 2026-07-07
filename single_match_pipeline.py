"""Run ASpanFormer + VGGT-Omega on a single DINO match JSON.

Ingests one match JSON of the format:
  {
    "source": "/path/to/source.jpg",
    "1": {"filepath": "/path/to/target1.jpg", "similarity_score": 0.54},
    "2": {"filepath": "/path/to/target2.jpg", "similarity_score": 0.53},
    ...
  }

and runs the two-stage pipeline on every source→target pair, writing
the full judged manifest under <output-dir>/vggt/vggt_judged_manifest.jsonl.

Path remapping: if the JSON was produced on a different machine / session,
pass --path-prefix-from and --path-prefix-to to rewrite paths in bulk.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def norm_path(p: str | Path) -> str:
    return str(p).replace("\\", "/")


def remap(path: str, prefix_from: str | None, prefix_to: str | None) -> str:
    if prefix_from and prefix_to and path.startswith(prefix_from):
        path = prefix_to + path[len(prefix_from):]
    return path


def parse_match_json(path: Path, prefix_from: str | None, prefix_to: str | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))

    raw_source = data.get("source") or data.get("source_image_path")
    if not raw_source:
        raise ValueError("Match JSON must contain a 'source' key with the source image path")

    source_path = remap(raw_source, prefix_from, prefix_to)
    source_id = Path(source_path).stem

    # Collect ranked entries: keys "1", "2", … with {filepath, similarity_score}
    entries: list[tuple[int, str, float | None]] = []
    for key, val in data.items():
        if not key.isdigit():
            continue
        if isinstance(val, dict):
            fp = val.get("filepath") or val.get("path") or ""
            score = val.get("similarity_score")
        else:
            continue
        if not fp:
            continue
        entries.append((int(key), remap(fp, prefix_from, prefix_to), score))

    entries.sort(key=lambda t: t[0])

    pairs: list[dict] = []
    for rank, (_, target_path, score) in enumerate(entries, start=1):
        target_id = Path(target_path).stem
        pairs.append({
            "candidate_id": f"{source_id}__r{rank:03d}__{target_id}",
            "source_index": 0,
            "target_index": rank - 1,
            "source_id": source_id,
            "target_id": target_id,
            "source_path": norm_path(source_path),
            "target_path": norm_path(target_path),
            "rank": rank,
            "similarity_score": score,
            "retrieval_metadata": {"source": "single_match_pipeline"},
        })

    return pairs


def write_manifest(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ASpan + VGGT on a single DINO match JSON."
    )
    # I/O
    parser.add_argument("--match-json", required=True, help="Path to the DINO match JSON file")
    parser.add_argument("--output-dir", required=True, help="Where all pipeline outputs are written")

    # Optional path remapping
    parser.add_argument("--path-prefix-from", default=None, help="Path prefix to replace (e.g. old Colab /content/... root)")
    parser.add_argument("--path-prefix-to", default=None, help="Replacement prefix for remapped paths")

    # ASpanFormer paths
    parser.add_argument("--aspanpath", required=True, help="Dir containing src/ASpanFormer/")
    parser.add_argument("--aspan-weights", required=True, help="ASpanFormer checkpoint path")
    parser.add_argument("--aspan-config", required=True, help="ASpanFormer config .py path")

    # VGGT paths
    parser.add_argument("--vggt-checkpoint", required=True, help="VGGT-Omega checkpoint path")
    parser.add_argument("--vggt-repo-path", default=None, help="Prepended to sys.path to make vggt_omega importable")

    # ASpan knobs
    parser.add_argument("--aspan-breakpoint", type=int, default=50)
    parser.add_argument("--long-dim", type=int, default=1024)

    # VGGT knobs
    parser.add_argument("--global-sim-threshold", type=float, default=0.90)
    parser.add_argument("--pose-score-mode", choices=("component", "raw"), default="component")
    parser.add_argument("--pose-component-threshold", type=float, default=3.0)
    parser.add_argument("--aspan-prepass-mode", choices=("lenient", "off"), default="lenient")

    # Execution
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    if args.vggt_repo_path and str(args.vggt_repo_path) not in sys.path:
        sys.path.insert(0, str(args.vggt_repo_path))

    # Stage 0: parse match JSON → retrieval_manifest.jsonl
    print("=" * 60)
    print(f"Parsing match JSON: {args.match_json}")
    pairs = parse_match_json(
        Path(args.match_json),
        args.path_prefix_from,
        args.path_prefix_to,
    )
    print(f"  Source  : {pairs[0]['source_id'] if pairs else '(none)'}")
    print(f"  Targets : {len(pairs)}")

    retrieval_manifest = output_dir / "retrieval_manifest.jsonl"
    write_manifest(pairs, retrieval_manifest)
    print(f"  Retrieval manifest: {retrieval_manifest}")

    # Stage 1: ASpanFormer
    print("=" * 60)
    print("Stage 1: ASpanFormer …")
    import geometry_filter as aspanfilter
    importlib.reload(aspanfilter)
    aspan_dir = output_dir / "aspan"
    aspan_argv = [
        "--input-manifest", str(retrieval_manifest),
        "--output-dir",     str(aspan_dir),
        "--breakpoint-value", str(args.aspan_breakpoint),
        "--aspanpath",      str(args.aspanpath),
        "--weights_path",   str(args.aspan_weights),
        "--config_path",    str(args.aspan_config),
        "--long_dim",       str(args.long_dim),
        "--device",         args.device,
    ]
    if args.resume:
        aspan_argv.append("--resume")
    if args.max_pairs is not None:
        aspan_argv += ["--max-pairs", str(args.max_pairs)]
    aspanfilter.main(aspan_argv)

    # Stage 2: VGGT-Omega
    print("=" * 60)
    print("Stage 2: VGGT-Omega …")
    import vggt_signals as vggt_judge
    importlib.reload(vggt_judge)
    vggt_dir = output_dir / "vggt"
    vggt_argv = [
        "--input-manifest",           str(aspan_dir / "vggt_candidates_manifest.jsonl"),
        "--output-dir",               str(vggt_dir),
        "--checkpoint",               str(args.vggt_checkpoint),
        "--global-sim-threshold",     str(args.global_sim_threshold),
        "--pose-score-mode",          args.pose_score_mode,
        "--pose-component-threshold", str(args.pose_component_threshold),
        "--aspan-prepass-mode",       args.aspan_prepass_mode,
        "--device",                   args.device,
    ]
    if args.resume:
        vggt_argv.append("--resume")
    if args.max_pairs is not None:
        vggt_argv += ["--max-pairs", str(args.max_pairs)]
    vggt_judge.main(vggt_argv)

    print("=" * 60)
    print("Done.")
    print(f"  Judged manifest : {vggt_dir / 'vggt_judged_manifest.jsonl'}")
    print(f"  True matches    : {vggt_dir / 'true_matches_manifest.jsonl'}")


if __name__ == "__main__":
    main()
