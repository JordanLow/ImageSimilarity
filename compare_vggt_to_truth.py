#!/usr/bin/env python3
"""Compare VGGT-Omega judge outputs against a source-of-truth match manifest.

Typical use after running vggt_judge.py:

    python compare_vggt_to_truth.py \
      --vggt-manifest /content/work/vggt_judge_outputs/vggt_judged_manifest.jsonl \
      --truth-matches /content/drive/MyDrive/ImageSimilarity/SOURCE_OF_TRUTH/manifests/matches.csv \
      --truth-images /content/drive/MyDrive/ImageSimilarity/SOURCE_OF_TRUTH/manifests/images.csv \
      --output-dir /content/work/vggt_judge_outputs/truth_compare

The script intentionally does not rerun ASpanFormer or VGGT. It only compares
manifest rows that already exist on disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
}

DEFAULT_POSITIVE_LABELS = ("true_positive", "positive", "match", "exact", "true_match")
DEFAULT_NON_POSITIVE_LABELS = (
    "false_positive",
    "not_applicable",
    "unlabeled",
    "needs_review",
    "conflict",
    "negative",
    "non_match",
    "sister",
)

SOURCE_FIELDS = (
    "source_id",
    "origin_image_id",
    "origin_id",
    "query_id",
    "image_a_id",
    "source_filename",
    "origin_filename",
    "query_filename",
    "image_a_filename",
    "source_path",
    "origin_path",
    "query_path",
    "image_a_path",
    "source",
    "origin",
    "image_a",
)
TARGET_FIELDS = (
    "target_id",
    "target_image_id",
    "match_image_id",
    "candidate_id_target",
    "image_b_id",
    "target_filename",
    "match_filename",
    "candidate_filename",
    "image_b_filename",
    "target_path",
    "match_path",
    "candidate_path",
    "image_b_path",
    "target",
    "match",
    "image_b",
)

# Columns kept in the main prediction CSV when present in VGGT rows.
OPTIONAL_VGGT_FIELDS = (
    "candidate_id",
    "rank",
    "similarity_score",
    "aspan_pass",
    "passed",
    "raw_matches",
    "filtered_matches",
    "sidecar_path",
    "global_similarity",
    "global_similarity_threshold",
    "global_similarity_pass",
    "pose_shift_total",
    "pose_shift_threshold",
    "pose_shift_pass",
    "judgement",
    "reason",
    "error",
)


def normalize_token(value: Any) -> str:
    """Normalize an image id, filename, or path to a comparison key.

    Only real image suffixes are stripped, so IDs containing periods such as
    ``10.s0001f000766`` are preserved.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    name = Path(text).name
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        name = name[: -len(Path(name).suffix)]
    return unicodedata.normalize("NFC", name).casefold().strip()


def raw_token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def pair_key(a: str, b: str, *, directed: bool) -> Tuple[str, str]:
    if directed:
        return (a, b)
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def label_key(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "_").replace("-", "_")


def truthy(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value) if isinstance(value, float) else False:
            return None
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "t", "yes", "y", "1", "pass", "passed", "match", "true_match"}:
        return True
    if text in {"false", "f", "no", "n", "0", "fail", "failed", "reject", "rejected"}:
        return False
    return None


def first_present(row: Mapping[str, Any], fields: Sequence[str]) -> Tuple[str, str]:
    for field in fields:
        if field in row and raw_token(row.get(field)):
            return field, raw_token(row.get(field))
    return "", ""


def read_table(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(item).__name__}")
                rows.append(item)
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict):
            for key in ("rows", "matches", "data", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return [dict(x) for x in value]
        raise ValueError(f"{path}: JSON must be a list or an object containing rows/matches/data/items")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred_fields: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for name in preferred_fields:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_image_aliases(images_path: Optional[Path]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    aliases: Dict[str, str] = {}
    anomalies: List[Dict[str, str]] = []
    if not images_path:
        return aliases, anomalies
    for idx, row in enumerate(read_table(images_path), 1):
        image_id = raw_token(row.get("image_id") or row.get("id") or row.get("source_id") or row.get("target_id"))
        if not image_id:
            anomalies.append({"type": "image_row_missing_id", "row_number": str(idx), "row": json.dumps(row, ensure_ascii=False)})
            continue
        canonical = normalize_token(image_id)
        candidates = {
            canonical,
            normalize_token(row.get("original_filename")),
            normalize_token(row.get("filename")),
            normalize_token(row.get("relative_path")),
            normalize_token(row.get("path")),
            normalize_token(row.get("normalized_key")),
        }
        for alias in candidates:
            if not alias:
                continue
            previous = aliases.get(alias)
            if previous and previous != canonical:
                anomalies.append(
                    {
                        "type": "ambiguous_image_alias",
                        "alias": alias,
                        "image_id_a": previous,
                        "image_id_b": canonical,
                    }
                )
                continue
            aliases[alias] = canonical
    return aliases, anomalies


def resolve_id(value: Any, aliases: Mapping[str, str]) -> str:
    normalized = normalize_token(value)
    return aliases.get(normalized, normalized)


@dataclass
class TruthPair:
    key: Tuple[str, str]
    source_id: str
    target_id: str
    label: str
    row: Dict[str, Any]


def extract_pair(row: Mapping[str, Any], aliases: Mapping[str, str]) -> Tuple[str, str, str, str]:
    source_field, source_raw = first_present(row, SOURCE_FIELDS)
    target_field, target_raw = first_present(row, TARGET_FIELDS)
    return source_field, resolve_id(source_raw, aliases), target_field, resolve_id(target_raw, aliases)


def load_truth_pairs(
    truth_matches_path: Path,
    aliases: Mapping[str, str],
    *,
    directed: bool,
) -> Tuple[Dict[Tuple[str, str], TruthPair], List[Dict[str, str]]]:
    truth: Dict[Tuple[str, str], TruthPair] = {}
    anomalies: List[Dict[str, str]] = []
    for idx, row in enumerate(read_table(truth_matches_path), 1):
        source_field, source_id, target_field, target_id = extract_pair(row, aliases)
        if not source_id or not target_id:
            anomalies.append(
                {
                    "type": "truth_row_missing_pair",
                    "row_number": str(idx),
                    "source_field": source_field,
                    "target_field": target_field,
                    "row": json.dumps(row, ensure_ascii=False),
                }
            )
            continue
        label = label_key(row.get("label") or row.get("truth_label") or row.get("match_label") or row.get("result"))
        key = pair_key(source_id, target_id, directed=directed)
        if key in truth:
            prev = truth[key]
            if prev.label != label:
                anomalies.append(
                    {
                        "type": "truth_duplicate_conflicting_label",
                        "pair_key": "|".join(key),
                        "label_a": prev.label,
                        "label_b": label,
                        "row_number": str(idx),
                    }
                )
            continue
        truth[key] = TruthPair(key=key, source_id=source_id, target_id=target_id, label=label, row=dict(row))
    return truth, anomalies


def is_predicted_true(row: Mapping[str, Any], *, assume_all_predicted: bool) -> bool:
    if assume_all_predicted:
        return True
    for field in ("true_match", "prediction", "predicted_true_match", "match", "accepted"):
        value = truthy(row.get(field))
        if value is not None:
            return value
    judgement = label_key(row.get("judgement") or row.get("result") or row.get("decision"))
    if judgement in {"true_match", "accepted", "accept", "match", "positive"}:
        return True
    if judgement in {"rejected", "reject", "false_match", "not_match", "negative"}:
        return False
    # For judged manifests, unknown/missing prediction fields should not be counted as accepted.
    # true_matches_manifest.jsonl is handled by assume_all_predicted/auto_assume_all.
    return False


def coerce_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def compact_prediction_row(
    row: Mapping[str, Any],
    *,
    source_id: str,
    target_id: str,
    key: Tuple[str, str],
    predicted_true: bool,
    truth_pair: Optional[TruthPair],
    result: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source_id": source_id,
        "target_id": target_id,
        "pair_key": "|".join(key),
        "predicted_true_match": predicted_true,
        "result": result,
        "truth_label": truth_pair.label if truth_pair else "",
        "in_source_of_truth": truth_pair is not None,
        "truth_source_id": truth_pair.source_id if truth_pair else "",
        "truth_target_id": truth_pair.target_id if truth_pair else "",
    }
    for field in OPTIONAL_VGGT_FIELDS:
        if field in row:
            out[field] = coerce_for_csv(row[field])
    # Always preserve these if present, even if an ID alias resolved differently.
    for field in ("source_path", "target_path", "source_id", "target_id"):
        if field in row:
            out[f"vggt_{field}"] = coerce_for_csv(row[field])
    return out


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def make_report(summary: Mapping[str, Any], output_paths: Mapping[str, str]) -> str:
    ratios = summary.get("ratios", {})
    lines = [
        "# VGGT Judge vs Source-of-Truth Comparison",
        "",
        "## Headline stats",
        f"- Evaluated VGGT rows: {summary['counts']['vggt_rows_total']}",
        f"- Predicted true matches: {summary['counts']['predicted_true_matches']}",
        f"- Truth positives in comparison scope: {summary['counts']['truth_positive_in_scope']}",
        f"- True positives found: {summary['counts']['true_positives_found']}",
        f"- False positives: {summary['counts']['false_positives']}",
        f"- Missed truth positives: {summary['counts']['missed_truth_positives']}",
        f"- Precision: {pct(ratios.get('precision'))}",
        f"- Recall: {pct(ratios.get('recall'))}",
        f"- F1: {pct(ratios.get('f1'))}",
        "",
        "## Output files",
    ]
    for name, path in output_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    if summary.get("warnings"):
        lines.extend(["", "## Warnings"])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vggt_judge.py outputs against SOURCE_OF_TRUTH matches and generate stats."
    )
    parser.add_argument(
        "--vggt-manifest",
        required=True,
        type=Path,
        help="VGGT output manifest, usually vggt_judged_manifest.jsonl. true_matches_manifest.jsonl also works.",
    )
    parser.add_argument(
        "--truth-matches",
        required=True,
        type=Path,
        help="SOURCE_OF_TRUTH matches manifest (.csv, .jsonl, or .json).",
    )
    parser.add_argument(
        "--truth-images",
        type=Path,
        default=None,
        help="Optional SOURCE_OF_TRUTH images manifest for filename/path-to-image_id alias resolution.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("vggt_truth_comparison"))
    parser.add_argument(
        "--positive-labels",
        default=",".join(DEFAULT_POSITIVE_LABELS),
        help="Comma-separated truth labels counted as real matches.",
    )
    parser.add_argument(
        "--non-positive-labels",
        default=",".join(DEFAULT_NON_POSITIVE_LABELS),
        help="Comma-separated known non-positive labels for reporting breakdowns.",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat A->B and B->A as different pairs. Default is symmetric/unordered, matching earlier SOURCE_OF_TRUTH comparison.",
    )
    parser.add_argument(
        "--scope",
        choices=("evaluated", "all"),
        default="evaluated",
        help=(
            "Recall denominator. 'evaluated' scopes missed truth positives to sources that appear in the VGGT manifest; "
            "'all' uses every positive truth pair."
        ),
    )
    parser.add_argument(
        "--assume-all-vggt-rows-predicted",
        action="store_true",
        help="Treat every VGGT row as a predicted true match. Useful when passing true_matches_manifest.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_labels = {label_key(x) for x in args.positive_labels.split(",") if x.strip()}
    non_positive_labels = {label_key(x) for x in args.non_positive_labels.split(",") if x.strip()}

    aliases, image_alias_anomalies = build_image_aliases(args.truth_images)
    truth_pairs, truth_anomalies = load_truth_pairs(args.truth_matches, aliases, directed=args.directed)
    vggt_rows = read_table(args.vggt_manifest)

    auto_assume_all = args.assume_all_vggt_rows_predicted or args.vggt_manifest.stem in {"true_matches", "true_matches_manifest"}
    warnings: List[str] = []
    if auto_assume_all and args.vggt_manifest.stem in {"true_matches", "true_matches_manifest"}:
        warnings.append("Input manifest name contains 'true_matches'; every row is treated as a predicted match.")

    truth_positive_keys = {key for key, pair in truth_pairs.items() if pair.label in positive_labels}
    truth_label_counts = Counter(pair.label or "<blank>" for pair in truth_pairs.values())

    prediction_rows: List[Dict[str, Any]] = []
    true_positive_rows: List[Dict[str, Any]] = []
    false_positive_rows: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, str]] = list(image_alias_anomalies) + list(truth_anomalies)
    evaluated_image_ids: set[str] = set()
    evaluated_pair_keys: set[Tuple[str, str]] = set()
    predicted_keys: set[Tuple[str, str]] = set()
    duplicate_prediction_counts: Counter[Tuple[str, str]] = Counter()
    predicted_truth_label_counts: Counter[str] = Counter()

    for idx, row in enumerate(vggt_rows, 1):
        source_field, source_id, target_field, target_id = extract_pair(row, aliases)
        if not source_id or not target_id:
            anomalies.append(
                {
                    "type": "vggt_row_missing_pair",
                    "row_number": str(idx),
                    "source_field": source_field,
                    "target_field": target_field,
                    "row": json.dumps(row, ensure_ascii=False, sort_keys=True),
                }
            )
            continue
        evaluated_image_ids.update((source_id, target_id))
        key = pair_key(source_id, target_id, directed=args.directed)
        evaluated_pair_keys.add(key)
        predicted_true = is_predicted_true(row, assume_all_predicted=auto_assume_all)
        truth_pair = truth_pairs.get(key)
        if truth_pair:
            predicted_truth_label_counts[truth_pair.label or "<blank>"] += int(predicted_true)

        if predicted_true:
            duplicate_prediction_counts[key] += 1
            if duplicate_prediction_counts[key] > 1:
                anomalies.append(
                    {
                        "type": "duplicate_vggt_prediction",
                        "pair_key": "|".join(key),
                        "row_number": str(idx),
                        "kept": "first_prediction_for_pair",
                    }
                )
                continue
            predicted_keys.add(key)
            if key in truth_positive_keys:
                result = "true_positive"
            elif truth_pair and truth_pair.label in non_positive_labels:
                result = f"known_non_positive:{truth_pair.label}"
            elif truth_pair:
                result = f"known_other_label:{truth_pair.label or '<blank>'}"
            else:
                result = "new_or_unlabeled_false_positive"
            compact = compact_prediction_row(
                row,
                source_id=source_id,
                target_id=target_id,
                key=key,
                predicted_true=predicted_true,
                truth_pair=truth_pair,
                result=result,
            )
            prediction_rows.append(compact)
            if result == "true_positive":
                true_positive_rows.append(compact)
            else:
                false_positive_rows.append(compact)

    duplicate_predictions = [
        {"pair_key": "|".join(key), "count": str(count)}
        for key, count in sorted(duplicate_prediction_counts.items())
        if count > 1
    ]

    if args.scope == "all":
        scoped_truth_positive_keys = set(truth_positive_keys)
    else:
        # Match previous comparison behavior: do not penalize a partial VGGT run for sources never evaluated.
        scoped_truth_positive_keys = {
            key
            for key, pair in truth_pairs.items()
            if pair.label in positive_labels and (pair.source_id in evaluated_image_ids or pair.target_id in evaluated_image_ids)
        }

    missed_keys = scoped_truth_positive_keys - predicted_keys
    missed_rows: List[Dict[str, Any]] = []
    for key in sorted(missed_keys):
        pair = truth_pairs[key]
        missed_rows.append(
            {
                "source_id": pair.source_id,
                "target_id": pair.target_id,
                "pair_key": "|".join(key),
                "truth_label": pair.label,
                "result": "missed_truth_positive",
                "truth_row": json.dumps(pair.row, ensure_ascii=False, sort_keys=True),
            }
        )

    tp = len(true_positive_rows)
    fp = len(false_positive_rows)
    fn = len(missed_rows)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)

    output_paths = {
        "vggt_predictions_compared.csv": str(output_dir / "vggt_predictions_compared.csv"),
        "true_positives.csv": str(output_dir / "true_positives.csv"),
        "false_positives.csv": str(output_dir / "false_positives.csv"),
        "missed_truth_matches.csv": str(output_dir / "missed_truth_matches.csv"),
        "anomalies.csv": str(output_dir / "anomalies.csv"),
        "summary.json": str(output_dir / "summary.json"),
        "report.md": str(output_dir / "report.md"),
    }

    preferred_prediction_fields = (
        "source_id",
        "target_id",
        "pair_key",
        "result",
        "truth_label",
        "predicted_true_match",
        "in_source_of_truth",
        "global_similarity",
        "pose_shift_total",
        "rank",
        "similarity_score",
        "candidate_id",
        "vggt_source_path",
        "vggt_target_path",
    )
    write_csv(output_dir / "vggt_predictions_compared.csv", prediction_rows, preferred_prediction_fields)
    write_csv(output_dir / "true_positives.csv", true_positive_rows, preferred_prediction_fields)
    write_csv(output_dir / "false_positives.csv", false_positive_rows, preferred_prediction_fields)
    write_csv(
        output_dir / "missed_truth_matches.csv",
        missed_rows,
        ("source_id", "target_id", "pair_key", "truth_label", "result", "truth_row"),
    )
    write_csv(output_dir / "anomalies.csv", anomalies, ("type", "row_number", "pair_key", "alias", "count", "row"))

    summary = {
        "inputs": {
            "vggt_manifest": str(args.vggt_manifest),
            "truth_matches": str(args.truth_matches),
            "truth_images": str(args.truth_images) if args.truth_images else None,
        },
        "settings": {
            "directed": args.directed,
            "scope": args.scope,
            "positive_labels": sorted(positive_labels),
            "non_positive_labels": sorted(non_positive_labels),
            "assume_all_vggt_rows_predicted": auto_assume_all,
        },
        "counts": {
            "vggt_rows_total": len(vggt_rows),
            "vggt_pairs_resolved": len(evaluated_pair_keys),
            "predicted_true_matches": len(prediction_rows),
            "predicted_unique_pairs": len(predicted_keys),
            "truth_pairs_total": len(truth_pairs),
            "truth_positive_total": len(truth_positive_keys),
            "truth_positive_in_scope": len(scoped_truth_positive_keys),
            "true_positives_found": tp,
            "false_positives": fp,
            "missed_truth_positives": fn,
            "duplicate_prediction_pairs": len(duplicate_predictions),
            "anomalies": len(anomalies),
        },
        "breakdowns": {
            "truth_labels_all": dict(sorted(truth_label_counts.items())),
            "predicted_truth_labels": dict(sorted(predicted_truth_label_counts.items())),
            "false_positive_results": dict(sorted(Counter(row["result"] for row in false_positive_rows).items())),
        },
        "ratios": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate_among_predictions": safe_ratio(fp, tp + fp),
        },
        "warnings": warnings,
        "outputs": output_paths,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(make_report(summary, output_paths), encoding="utf-8")

    print(make_report(summary, output_paths))


if __name__ == "__main__":
    main()
