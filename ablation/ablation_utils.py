"""Shared data-loading and metric utilities for all Stage 1 ablation scripts.

Every script in _local/ imports from here instead of duplicating load logic.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Canonical data paths ───────────────────────────────────────────────────────

DINO_ROOT = Path(r"D:\DINO OUTPUTS")

SHARDS: dict[str, dict[str, Path]] = {
    "Shard1": {
        "manifest_csv":   DINO_ROOT / "match_manifest_shard1.csv",
        "judge_jsonl":    DINO_ROOT / "Shard1 Judge Manifest.jsonl",
        "viz_root":       DINO_ROOT / "DINO_Output_Shard1" / "visualizations",
        "aspan_all":      DINO_ROOT / "aspan_all_manifest_shard1.jsonl",   # written by consolidate_aspan_jsons.py
    },
    "Shard2": {
        "manifest_csv":   DINO_ROOT / "match_manifest_shard2.csv",
        "judge_jsonl":    DINO_ROOT / "Shard2 Judge Manifest.jsonl",
        "viz_root":       DINO_ROOT / "DINO_Output_Shard2" / "visualizations",
        "aspan_all":      DINO_ROOT / "aspan_all_manifest_shard2.jsonl",
    },
}

EXCLUDE_LABELS = {"unsure", "unknown"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ground_truth(csv_path: Path) -> dict[tuple[str, str], str]:
    """Returns {(source_id, target_stem): classification}. Preserves all labels
    including Unsure/Unknown — callers decide whether to exclude."""
    gt: dict[tuple[str, str], str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["source_folder"], Path(row["target_image"]).stem)
            gt[key] = row["classification"].strip()
    return gt


def load_judge_manifest(jsonl_path: Path) -> dict[tuple[str, str], dict]:
    """Returns {(source_id, target_id): row_dict} for each line in the manifest."""
    judge: dict[tuple[str, str], dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            judge[(row["source_id"], row["target_id"])] = row
    return judge


def load_aspan_all(jsonl_path: Path) -> dict[tuple[str, str], dict]:
    """Returns {(source_id, target_id): row_dict} from the consolidated aspan_all manifest."""
    return load_judge_manifest(jsonl_path)


def load_decisive_pairs(
    csv_path: Path,
    jsonl_path: Path,
) -> list[tuple[str, dict]]:
    """Join ground truth with judge manifest. Returns list of (label, row_dict)
    for Positive and Negative pairs that appear in the manifest.
    Unsure/Unknown pairs are excluded."""
    gt = load_ground_truth(csv_path)
    judge = load_judge_manifest(jsonl_path)
    pairs = []
    for key, label in gt.items():
        if label.lower() in EXCLUDE_LABELS:
            continue
        if label not in ("Positive", "Negative"):
            continue
        row = judge.get(key)
        if row is None:
            continue
        pairs.append((label, row))
    return pairs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Compute TP/FP/TN/FN/P/R/F1 from binary lists (1=Positive)."""
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1)


def pr_auc(y_true: list[int], scores: list[float]) -> float:
    """PR-AUC (average precision) without sklearn.

    Computes the area under the precision-recall curve by sorting pairs by
    descending score and computing trapezoidal interpolation. Equivalent to
    sklearn's average_precision_score with the same sign convention."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(set(y)) < 2:
        return float("nan")
    # Sort descending by score (higher score = predicted positive)
    order = np.argsort(s)[::-1]
    y_sorted = y[order]
    n_pos = y.sum()
    precisions, recalls = [], []
    tp = fp = 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    # Prepend (recall=0, precision=1) sentinel
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions
    # Trapezoidal integration
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    auc_val = float(trapz(precisions, recalls))
    return auc_val


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    """ROC-AUC without sklearn (Mann-Whitney U statistic)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(set(y)) < 2:
        return float("nan")
    pos_scores = s[y == 1]
    neg_scores = s[y == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    # U statistic = # (pos, neg) pairs where pos_score > neg_score + 0.5 * ties
    u = sum(
        1.0 if p > n else (0.5 if p == n else 0.0)
        for p in pos_scores for n in neg_scores
    )
    return float(u / (n_pos * n_neg))
