"""Table A retrieval evaluation: Recall@K and mAP for DINO, SSCD, CLIP, raw DINOv3,
and JOCCH (the original ViT ensemble).

For each model, computes:
  Recall@K  (K = 1, 5, 10, 15, 50): fraction of queries where the true copy
            appears in the model's top-K candidates.
  mAP:      mean average precision; for queries with one GT+ target, this equals
            mean reciprocal rank (MRR) truncated at max_k.
  95% CI:   bootstrap confidence interval on both, resampling queries (10,000
            resamples, seed 42 — same convention as ablation/statistics.py's
            bootstrap_ci) per the strategy report's §6.8 "every table gets CIs".

Denominator: the set of source images that have at least one labeled GT+ target.
A source with no labeled GT+ target is excluded (we cannot measure recall for it).

Per-query results (one row per source: hit/miss at each K, AP) are persisted via
--per-query-dir so Table A's numbers are independently auditable and re-testable
without rerunning this script — the same "persist per-example decisions, not just
aggregates" principle ablation/STATISTICS_METHODOLOGY.md establishes for Stage 2/3.

Two-pass design:
  Pass 1 (conservative): uses only existing GT labels from the Shard CSVs.
         Treats all unlabeled pairs as unknown → gives a LOWER BOUND on Recall@K
         (correct if SSCD/CLIP cannot retrieve copies that DINO missed; may be
         pessimistic if they can).
  Pass 2 (complete): also loads --new-labels CSVs (your annotated files from
         export_retrieval_candidates.py). Use this for the final Table A numbers.

Usage (conservative pass, preliminary numbers):
    python ablation/eval_retrieval.py \\
        --dino-manifests "D:/DINO OUTPUTS/retrieval_manifest_shard1.jsonl" \\
                         "D:/DINO OUTPUTS/retrieval_manifest_shard2.jsonl" \\
        --sscd-manifests "D:/DINO OUTPUTS/sscd_shard1.jsonl" \\
                         "D:/DINO OUTPUTS/sscd_shard2.jsonl" \\
        --clip-manifests "D:/DINO OUTPUTS/clip_shard1.jsonl" \\
                         "D:/DINO OUTPUTS/clip_shard2.jsonl" \\
        --dinov3raw-manifests "D:/DINO OUTPUTS/dinov3raw_shard1.jsonl" \\
                              "D:/DINO OUTPUTS/dinov3raw_shard2.jsonl" \\
        --jocch-manifests "D:/DINO OUTPUTS/jocch_shard1.jsonl" \\
                          "D:/DINO OUTPUTS/jocch_shard2.jsonl" \\
        --gt-csv "D:/DINO OUTPUTS/match_manifest_shard1.csv" \\
                 "D:/DINO OUTPUTS/match_manifest_shard2.csv"

Usage (complete pass, after labeling new candidates):
    python ablation/eval_retrieval.py ... \\
        --new-labels "D:/DINO OUTPUTS/retrieval_review/new_candidates_sscd_labeled.csv" \\
                     "D:/DINO OUTPUTS/retrieval_review/new_candidates_clip_labeled.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ablation_utils import load_ground_truth

N_BOOT    = 10_000
BOOT_SEED = 42


# ── Ground truth loading ───────────────────────────────────────────────────────

def load_gt(csv_paths: list[Path]) -> dict[tuple[str, str], str]:
    """Returns {(source_id, target_id): label} from Shard CSVs. Thin wrapper around
    ablation_utils.load_ground_truth (the canonical GT loader) so CSV parsing isn't
    duplicated across scripts."""
    gt: dict[tuple[str, str], str] = {}
    for p in csv_paths:
        gt.update(load_ground_truth(p))
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

def compute_per_query(
    ranked: dict[str, list[tuple[int, str, float]]],
    gt_positives: dict[str, set[str]],
    ks: tuple[int, ...],
) -> list[dict]:
    """One row per query source: recall_at_{k} (0/1) for each k, plus AP.

    This is computed once and shared by the point-estimate aggregation, the
    bootstrap CI, and the persisted per-query JSONL — no metric is computed two
    different ways.

    AP for a query with n_rel GT+ targets, truncated at max(ks):
      AP = (1/n_rel) * sum_{r=1}^{max_k} [rel(r) * P@r], P@r = (# GT+ in top-r) / r
    """
    max_k = max(ks)
    rows: list[dict] = []
    for src, pos_targets in gt_positives.items():
        candidates = ranked.get(src, [])
        row: dict = {"source_id": src, "n_gt_positive": len(pos_targets)}
        for k in ks:
            top_k_targets = {tgt for rank, tgt, _ in candidates if rank <= k}
            row[f"recall_at_{k}"] = 1 if (top_k_targets & pos_targets) else 0

        truncated = [(rank, tgt) for rank, tgt, _ in candidates if rank <= max_k]
        n_rel = len(pos_targets)
        hits = prec_sum = 0
        for rank, tgt in sorted(truncated, key=lambda x: x[0]):
            if tgt in pos_targets:
                hits += 1
                prec_sum += hits / rank
        row["ap"] = (prec_sum / n_rel) if n_rel else None
        rows.append(row)
    return rows


def aggregate_metrics(per_query_rows: list[dict], ks: tuple[int, ...]) -> dict:
    """Point-estimate Recall@K / mAP from compute_per_query's output."""
    result: dict = {}
    n = len(per_query_rows)
    for k in ks:
        result[f"Recall@{k}"] = (
            sum(r[f"recall_at_{k}"] for r in per_query_rows) / n if n else None
        )
    aps = [r["ap"] for r in per_query_rows if r["ap"] is not None]
    result["mAP"] = sum(aps) / len(aps) if aps else None
    result["n_queries"] = n
    return result


def bootstrap_ci(
    per_query_rows: list[dict],
    ks: tuple[int, ...],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Bootstrap 95% CIs for Recall@K (each k) and mAP, resampling query sources
    with replacement. Same n_boot/seed convention as ablation/statistics.py's
    bootstrap_ci, for consistency across the project's tables."""
    n = len(per_query_rows)
    out: dict = {}
    if n == 0:
        for k in ks:
            out[f"Recall@{k}_ci"] = (float("nan"), float("nan"))
        out["mAP_ci"] = (float("nan"), float("nan"))
        return out

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))  # (n_boot, n) resample indices

    for k in ks:
        vec = np.array([r[f"recall_at_{k}"] for r in per_query_rows], dtype=np.float64)
        samples = vec[idx].mean(axis=1)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        out[f"Recall@{k}_ci"] = (float(lo), float(hi))

    ap_vec = np.array(
        [r["ap"] if r["ap"] is not None else np.nan for r in per_query_rows], dtype=np.float64
    )
    ap_samples = np.nanmean(ap_vec[idx], axis=1)
    ap_samples = ap_samples[~np.isnan(ap_samples)]
    if len(ap_samples):
        lo, hi = np.percentile(ap_samples, [2.5, 97.5])
        out["mAP_ci"] = (float(lo), float(hi))
    else:
        out["mAP_ci"] = (float("nan"), float("nan"))
    return out


def save_per_query_jsonl(per_query_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in per_query_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_model_metrics(
    ranked: dict[str, list[tuple[int, str, float]]],
    gt_positives: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 5, 10, 15, 50),
) -> tuple[dict, list[dict]]:
    """Returns (point_estimate + CI dict, per_query_rows) — per_query_rows is
    exposed so callers can persist it."""
    per_query_rows = compute_per_query(ranked, gt_positives, ks)
    result = aggregate_metrics(per_query_rows, ks)
    result.update(bootstrap_ci(per_query_rows, ks))
    return result, per_query_rows


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


def format_ci(ci) -> str:
    if ci is None:
        return "n/a"
    lo, hi = ci
    if lo != lo or hi != hi:  # NaN check without importing math
        return "n/a"
    return f"[{lo * 100:.2f}, {hi * 100:.2f}]"


def print_ci_table(results: dict[str, dict], ks: tuple[int, ...]) -> None:
    col_w = 20
    headers = ["Model"] + [f"R@{k} 95% CI" for k in ks] + ["mAP 95% CI"]
    print(f"95% bootstrap CIs ({N_BOOT} resamples, seed {BOOT_SEED}, resampled over query sources):")
    print("  ".join(h.ljust(col_w) for h in headers))
    print("  ".join("-" * col_w for _ in headers))
    for model_name, metrics in results.items():
        row = [model_name.ljust(col_w)]
        for k in ks:
            row.append(format_ci(metrics.get(f"Recall@{k}_ci")).ljust(col_w))
        row.append(format_ci(metrics.get("mAP_ci")).ljust(col_w))
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
    p.add_argument("--dinov3raw-manifests", nargs="+", default=[],
                   help="Raw DINOv3 (no head/ensemble) retrieval manifest JSONL files.")
    p.add_argument("--jocch-manifests", nargs="+", default=[],
                   help="JOCCH (original ViT ensemble) retrieval manifest JSONL files.")
    p.add_argument("--gt-csv", nargs="+", required=True,
                   help="Ground-truth label CSVs (Shard 1 + Shard 2).")
    p.add_argument("--new-labels", nargs="+", default=[],
                   help="Annotated new-candidates CSVs from export_retrieval_candidates.py.")
    p.add_argument("--output-json", type=str, default="",
                   help="Optional path to save results as JSON.")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 15, 50],
                   help="K values for Recall@K. Default: 1 5 10 15 50.")
    p.add_argument("--per-query-dir", type=str, default="",
                   help="Optional directory to write one {model}.jsonl per model with "
                        "per-source recall_at_{k}/ap rows -- makes Table A auditable and "
                        "re-testable without rerunning this script.")
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
        ("DINO",      args.dino_manifests),
        ("SSCD",      args.sscd_manifests),
        ("CLIP",      args.clip_manifests),
        ("DINOv3Raw", args.dinov3raw_manifests),
        ("JOCCH",     args.jocch_manifests),
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
        metrics, per_query_rows = compute_model_metrics(ranked, gt_positives, ks)
        all_results[model_name] = metrics

        if args.per_query_dir:
            out_path = Path(args.per_query_dir) / f"{model_name}.jsonl"
            save_per_query_jsonl(per_query_rows, out_path)
            print(f"  Per-query rows saved: {out_path}")

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
    print_ci_table(all_results, ks)

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
