"""Folder-traversal ASpanFormer + VGGT-Omega pipeline.

Experimental utility (not part of the main pipeline). Traverses a pre-curated
folder structure where each folder is named after a source image and contains a
Matches/ subfolder with the source image and candidate target images, then runs
ASpanFormer + VGGT-Omega on each pair.

Expected folder layout:
  <input-dir>/
    <source_id>/
      Matches/
        <source_id>.jpg         ← source image (stem == parent folder name)
        <target1>.jpg           ← candidate target images
        ...
      match_log.json            ← optional; keys=original abs paths, values=DINO scores
    [no matches] <source_id>/  ← skipped (prefix "[no matches]", all lowercase)

Outputs under <output-dir>:
  retrieval_manifest.jsonl         ← regenerated from discovery each run
  aspan/
    aspan_all_manifest.jsonl
    vggt_candidates_manifest.jsonl
    aspan_sidecars/*.npz
  vggt/
    vggt_judged_manifest.jsonl      ← final manifest (informationally complete)
    true_matches_manifest.jsonl
    vggt_judge_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def norm_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def discover_pairs(input_dir: Path) -> list[dict]:
    """Walk input_dir and return one retrieval-manifest row per source→target pair."""
    pairs = []
    source_idx = 0
    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("[no matches]"):
            continue
        matches_dir = folder / "Matches"
        if not matches_dir.exists():
            continue

        source_name = folder.name
        source_img = None
        target_imgs: list[Path] = []
        for img in sorted(matches_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if img.stem == source_name:
                source_img = img
            else:
                target_imgs.append(img)

        if source_img is None or not target_imgs:
            print(f"  [discover] Skip {folder.name}: missing source or no targets")
            continue

        # Optional: read DINO similarity scores from match_log.json
        scores: dict[str, float] = {}
        log_path = folder / "match_log.json"
        if log_path.exists():
            try:
                log = json.loads(log_path.read_text(encoding="utf-8"))
                for k, v in log.get("matches", {}).items():
                    scores[Path(k).name] = float(v)
            except Exception as exc:
                print(f"  [discover] Warning: could not read {log_path}: {exc}")

        # Sort targets by DINO score descending so rank 1 = best retrieval match
        target_imgs.sort(key=lambda p: scores.get(p.name, 0.0), reverse=True)

        for rank, tgt in enumerate(target_imgs, start=1):
            pairs.append({
                "candidate_id": f"{source_name}__r{rank:03d}__{tgt.stem}",
                "source_index": source_idx,
                "target_index": rank - 1,
                "source_id": source_name,
                "target_id": tgt.stem,
                "source_path": norm_path(source_img),
                "target_path": norm_path(tgt),
                "rank": rank,
                "similarity_score": scores.get(tgt.name),
                "retrieval_metadata": {"source": "folder_pipeline"},
            })
        source_idx += 1
    return pairs


def write_manifest(pairs: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(pairs)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traverse pre-curated match folders and run ASpan + VGGT on each pair."
    )
    # I/O
    parser.add_argument("--input-dir", required=True, help="Root folder containing source-named subdirs")
    parser.add_argument("--output-dir", required=True, help="Where all pipeline outputs are written")

    # ASpanFormer paths
    parser.add_argument("--aspanpath", required=True, help="Dir containing src/ASpanFormer/ (same as aspanfilter.py)")
    parser.add_argument("--aspan-weights", required=True, help="ASpanFormer checkpoint path")
    parser.add_argument("--aspan-config", required=True, help="ASpanFormer config .py path")

    # VGGT paths
    parser.add_argument("--vggt-checkpoint", required=True, help="VGGT-Omega checkpoint path")
    parser.add_argument("--vggt-repo-path", default=None, help="Prepended to sys.path to make vggt_omega importable (if not installed)")

    # ASpan knobs
    parser.add_argument("--aspan-breakpoint", type=int, default=50, help="Min filtered keypoints to pass ASpan stage")
    parser.add_argument("--long-dim", type=int, default=1024, help="ASpan resize long edge")

    # VGGT knobs
    parser.add_argument("--global-sim-threshold", type=float, default=0.90)
    parser.add_argument("--pose-score-mode", choices=("component", "raw"), default="component")
    parser.add_argument("--pose-component-threshold", type=float, default=3.0)
    parser.add_argument("--aspan-prepass-mode", choices=("lenient", "off"), default="lenient")

    # Execution
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 …")
    parser.add_argument("--resume", action="store_true", help="Skip pairs already present in output manifests")
    parser.add_argument("--max-pairs", type=int, default=None, help="Debug cap: stop after N pairs per stage")

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

    # Stage 0: discover pairs → retrieval_manifest.jsonl
    print("=" * 60)
    print("Stage 0: discovering pairs from folder structure …")
    pairs = discover_pairs(Path(args.input_dir))
    source_count = len({p["source_id"] for p in pairs})
    print(f"  Found {len(pairs)} pairs across {source_count} source folders")

    retrieval_manifest = output_dir / "retrieval_manifest.jsonl"
    write_manifest(pairs, retrieval_manifest)
    print(f"  Retrieval manifest: {retrieval_manifest}")

    # Stage 1: ASpanFormer filtering
    print("=" * 60)
    print("Stage 1: ASpanFormer …")
    import geometry_filter as aspanfilter
    import importlib
    importlib.reload(aspanfilter)
    aspan_dir = output_dir / "aspan"
    aspan_argv = [
        "--input-manifest", str(retrieval_manifest),
        "--output-dir", str(aspan_dir),
        "--breakpoint-value", str(args.aspan_breakpoint),
        "--aspanpath", str(args.aspanpath),
        "--weights_path", str(args.aspan_weights),
        "--config_path", str(args.aspan_config),
        "--long_dim", str(args.long_dim),
        "--device", args.device,
    ]
    if args.resume:
        aspan_argv.append("--resume")
    if args.max_pairs is not None:
        aspan_argv += ["--max-pairs", str(args.max_pairs)]
    aspanfilter.main(aspan_argv)

    # Stage 2: VGGT-Omega judgement
    print("=" * 60)
    print("Stage 2: VGGT-Omega …")
    import vggt_signals as vggt_judge
    importlib.reload(vggt_judge)
    vggt_dir = output_dir / "vggt"
    vggt_candidates = aspan_dir / "vggt_candidates_manifest.jsonl"
    vggt_argv = [
        "--input-manifest", str(vggt_candidates),
        "--output-dir", str(vggt_dir),
        "--checkpoint", str(args.vggt_checkpoint),
        "--global-sim-threshold", str(args.global_sim_threshold),
        "--pose-score-mode", args.pose_score_mode,
        "--pose-component-threshold", str(args.pose_component_threshold),
        "--aspan-prepass-mode", args.aspan_prepass_mode,
        "--device", args.device,
    ]
    if args.resume:
        vggt_argv.append("--resume")
    if args.max_pairs is not None:
        vggt_argv += ["--max-pairs", str(args.max_pairs)]
    vggt_judge.main(vggt_argv)

    print("=" * 60)
    print(f"Done.")
    print(f"  Judged manifest : {vggt_dir / 'vggt_judged_manifest.jsonl'}")
    print(f"  True matches    : {vggt_dir / 'true_matches_manifest.jsonl'}")


if __name__ == "__main__":
    main()
