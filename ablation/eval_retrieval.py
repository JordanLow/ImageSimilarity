"""Table A retrieval evaluation: Recall@K and mAP for DINO, SSCD, CLIP.

For each model, computes:
  Recall@K  (K = 1, 5, 10, 15, 50): fraction of queries where the true copy
            appears in the model's top-K candidates.
  mAP:      mean average precision; for queries with one GT+ target, this equals
            mean reciprocal rank (MRR) truncated at max_k.

Denominator: the set of source images that have at least one labeled GT+ target.
A source with no labeled GT+ target is excluded (we cannot measure recall for it).

Two-pass design:
  Pass 1 (conservative): uses only existing GT labels from the Shard CSVs.
         Treats all unlabeled pairs as unknown → gives a LOWER BOUND on Recall@K
         (correct if SSCD/CLIP cannot retrieve copies that DINO missed; may be
         pessimistic if they can).
  Pass 2 (complete): also loads --new-labels CSVs (your annotated files from
         export_retrieval_candidates.py). Use this for the final Table A numbers.

Usage (conservative pass, preliminary numbers):
    python _local/eval_retrieval.py \\
        --dino-manifests "D:/DINO OUTPUTS/retrieval_manifest_shard1.jsonl" \\
                         "D:/DINO OUTPUTS/retrieval_manifest_shard2.jsonl" \\
        --sscd-manifests "D:/DINO OUTPUTS/sscd_shard1.jsonl" \\
                         "D:/DINO OUTPUTS/sscd_shard2.jsonl" \\
        --clip-manifests "D:/DINO OUTPUTS/clip_shard1.jsonl" \\
                         "D:/DINO OUTPUTS/clip_shard2.jsonl" \\
        --gt-csv "D:/DINO OUTPUTS/match_manifest_shard1.csv" \\
                 "D:/DINO OUTPUTS/match_manifest_shard2.csv"

Usage (complete pass, after labeling new candidates):
    python _local/eval_retrieval.py ... \\
        --new-labels "D:/DINO OUTPUTS/retrieval_review/new_candidates_sscd_labeled.csv" \\
                     "D:/DINO OUTPUTS/retrieval_review/new_candidates_clip_labeled.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


# ── Ground truth loading ───────────────────────────────────────────────────────

def load_gt(csv_paths: list[Path]) -> dict[tuple[str, str], str]:
    """Returns {(source_id, target_id): label} from Shard CSVs."""
    gt: dict[tuple[str, str], str] = {}
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                src = row["source_folder"].strip()
                tgt = Path(row["target_image"].strip()).stem
                label = row["classification"].strip()
                gt[(src, tgt)] = label
    return gt


def load_new_labels(csv_paths: list[Path]) -> dict[tuple[str, str], str]:
    """Returns {(source_id, target_id): label} from annotated new-candidates CSVs.

    Only rows where 'label' is non-empty are included.
    """
    labels: dict[tuple[str, str], str] = {}
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = row.get("label", "").strip()
                if not label:
                    continue
                src = row["source_id"].strip()
                tgt = row["target_id"].strip()
                labels[(src, tgt)] = label
    return labels


# ── Retrieval manifest loading ─────────────────────────────────────────────────

def load_manifest(jsonl_paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in jsonl_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_ranked_lists(
    rows: list[dict],
) -> dict[str, list[tuple[int, str, float]]]:
    """Returns {source_id: [(rank, target_id, score), ...]} sorted by rank."""
    ranked: dict[str, list[tuple[int, str, float]]] = {}
    for r in rows:
        src = r["source_id"]
        tgt = r["target_id"]
        rank = int(r.get("rank", 9999))
        score = float(r.get("similarity_score", 0.0))
        ranked.setdefault(src, []).append((rank, tgt, score))
    for src in ranked:
        ranked[src].sort(key=lambda x: x[0])
    return ranked


# ── Metrics ────────────────────────────────────────────────────────────────────

def recall_at_k(
    ranked: dict[str, list[tuple[int, str, float]]],
    gt_positives: dict[str, set[str]],
    k: int,
) -> Optional[float]:
    """Recall@K over queries that have at least one GT+ target.

    For each source with a known GT+: 1 if any GT+ target appears in top-K, else 0.
    """
    hits = total = 0
    for src, pos_targets in gt_positives.items():
        candidates = ranked.get(src, [])
        top_k_targets = {tgt for rank, tgt, _ in candidates if rank <= k}
        if top_k_targets & pos_targets:
            hits += 1
        total += 1
    if total == 0:
        return None
    return hits / total


def mean_ap(
    ranked: dict[str, list[tuple[int, str, float]]],
    gt_positives: dict[str, set[str]],
    max_k: int = 50,
) -> Optional[float]:
    """mAP over queries with at least one GT+ target, truncated at max_k.

    For a query with n_rel GT+ targets in the ranked list:
      AP = (1/n_rel) * sum_{r=1}^{max_k} [rel(r) * P@r]
    where P@r = (# GT+ in top-r) / r.
    """
    aps = []
    for src, pos_targets in gt_positives.items():
        candidates = [(rank, tgt) for rank, tgt, _ in ranked.get(src, []) if rank <= max_k]
        n_rel = len(pos_targets)
        if n_rel == 0:
            continue
        hits = prec_sum = 0
        for rank, tgt in sorted(candidates, key=lambda x: x[0]):
            if tgt in pos_targets:
                hits += 1
                prec_sum += hits / rank
        ap = prec_sum / n_rel
        aps.append(ap)
    if not aps:
        return None
    return sum(aps) / len(aps)


def compute_model_metrics(
    ranked: dict[str, list[tuple[int, str, float]]],
    gt_positives: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 5, 10, 15, 50),
) -> dict:
    result: dict = {}
    for k in ks:
        r = recall_at_k(ranked, gt_positives, k)
        result[f"Recall@{k}"] = r
    result["mAP"] = mean_ap(ranked, gt_positives, max_k=max(ks))
    result["n_queries"] = len(gt_positives)
    return result


def format_pct(v) -> str:
    if v is None:
        return "  n/a "
    return f"{v * 100:6.2f}%"


def print_table(results: dict[str, dict], ks: tuple[int, ...]) -> None:
    col_w = 11
    headers = ["Model"] + [f"R@{k}" for k in ks] + ["mAP", "Queries"]
    print("")
    print("  ".join(h.ljust(col_w) for h in headers))
    print("  ".join("-" * col_w for _ in headers))
    for model_name, metrics in results.items():
        row = [model_name.ljust(col_w)]
        for k in ks:
            row.append(format_pct(metrics.get(f"Recall@{k}")).ljust(col_w))
        row.append(format_pct(metrics.get("mAP")).ljust(col_w))
        row.append(str(metrics.get("n_queries", "?")).ljust(col_w))
        print("  ".join(row))
    print("")


def save_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Table A retrieval metrics.")
    p.add_argument("--dino-manifests", nargs="+", default=[],
                   help="DINO retrieval manifest JSONL files.")
    p.add_argument("--sscd-manifests", nargs="+", default=[],
                   help="SSCD retrieval manifest JSONL files.")
    p.add_argument("--clip-manifests", nargs="+", default=[],
                   help="CLIP retrieval manifest JSONL files.")
    p.add_argument("--gt-csv", nargs="+", required=True,
                   help="Ground-truth label CSVs (Shard 1 + Shard 2).")
    p.add_argument("--new-labels", nargs="+", default=[],
                   help="Annotated new-candidates CSVs from export_retrieval_candidates.py.")
    p.add_argument("--output-json", type=str, default="",
                   help="Optional path to save results as JSON.")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 15, 50],
                   help="K values for Recall@K. Default: 1 5 10 15 50.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ks = tuple(sorted(args.ks))

    # Load ground truth
    print("Loading ground truth...")
    gt = load_gt([Path(p) for p in args.gt_csv])
    if args.new_labels:
        print(f"Loading {len(args.new_labels)} new-label file(s)...")
        new = load_new_labels([Path(p) for p in args.new_labels])
        print(f"  {len(new)} annotated new pairs loaded")
        gt.update(new)
    print(f"  Total labeled pairs: {len(gt)}")

    # Build GT positives per source: {source_id: set of GT+ target_ids}
    gt_positives: dict[str, set[str]] = {}
    for (src, tgt), label in gt.items():
        if label.lower() in ("unsure", "unknown", ""):
            continue
        if label == "Positive":
            gt_positives.setdefault(src, set()).add(tgt)
    print(f"  Sources with at least one GT+ target: {len(gt_positives)}")

    # Build ranked lists for each model and evaluate
    models_to_run: list[tuple[str, list[str]]] = [
        ("DINO",  args.dino_manifests),
        ("SSCD",  args.sscd_manifests),
        ("CLIP",  args.clip_manifests),
    ]

    all_results: dict[str, dict] = {}
    for model_name, manifest_paths in models_to_run:
        if not manifest_paths:
            continue
        paths = [Path(p) for p in manifest_paths]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"\n[{model_name}] Missing files: {missing}  — skipping")
            continue
        print(f"\nEvaluating {model_name}...")
        rows = load_manifest(paths)
        print(f"  {len(rows)} candidate rows loaded")
        ranked = build_ranked_lists(rows)
        metrics = compute_model_metrics(ranked, gt_positives, ks)
        all_results[model_name] = metrics

        # Per-model coverage report
        missing_sources = set(gt_positives) - set(ranked)
        if missing_sources:
            print(f"  {len(missing_sources)} sources with GT+ have NO candidates in this manifest")
            print(f"  (these count as misses for all Recall@K)")

    if not all_results:
        print("\nNo results to display. Check that manifest paths exist and are non-empty.")
        return

    # Print table
    print("\n" + "=" * 75)
    print("TABLE A — Retrieval baseline comparison")
    print("=" * 75)
    mode = "CONSERVATIVE (unlabeled = unknown)" if not args.new_labels else "COMPLETE (with new labels)"
    print(f"Mode: {mode}")
    print_table(all_results, ks)

    # Annotations note
    if not args.new_labels:
        print("NOTE: This is a lower-bound estimate. Run export_retrieval_candidates.py,")
        print("      label new candidates, then re-run with --new-labels for final numbers.\n")

    # Save JSON
    if args.output_json:
        out_path = Path(args.output_json)
        save_json(all_results, out_path)
        print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
