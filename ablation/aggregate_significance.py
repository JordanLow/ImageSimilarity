"""Final aggregation step — the ONE place Holm-Bonferroni correction happens for the
paper's statistics.

Collects RAW (uncorrected) McNemar results from every upstream script/notebook and
applies `ablation/significance.py`'s `holm_correct()` once per stage family, pooled
across both shards within a family (per explicit user decision — see
`ablation/STATISTICS_METHODOLOGY.md`):

  - **Stage 3 family** = B1, B2, B5, B6, B8, B11 (from `ablation/statistics.py`'s
    output, `statistics_results.json`) + B10 (from the MASt3R Colab notebook's
    synced output, `mast3r_stage3_results.json`, once it exists) — every row, both
    shards, corrected together as one family.
  - **Stage 2 family** = B14/C1, C2/RoMa (from `ablation/eval_stage2.py`'s output)
    — every row, both shards, corrected together as a separate family.

No individual upstream script computes its own Holm-adjusted values anymore — they
emit raw p-values only, tagged with a `family` label. This script is the single
place those get turned into the numbers the paper actually cites.

A missing upstream source (most likely: the MASt3R notebook hasn't been run/synced
yet) is reported loudly and that family is corrected over whatever *is* available —
but the summary explicitly flags the family as PROVISIONAL/INCOMPLETE rather than
silently correcting over a smaller family without saying so.

Outputs:
    D:/DINO OUTPUTS/statistics_final.json
    D:/DINO OUTPUTS/statistics_final.md

Usage:
    python ablation/aggregate_significance.py
    python ablation/aggregate_significance.py \\
        --stage2-results "D:/DINO OUTPUTS/stage2_geometry_results.json" \\
        --mast3r-results "D:/DINO OUTPUTS/mast3r_stage3_results.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from ablation_utils import DINO_ROOT
from significance import holm_correct, ALPHA

STATISTICS_JSON  = DINO_ROOT / "statistics_results.json"     # ablation/statistics.py's output
STAGE2_JSON      = DINO_ROOT / "stage2_geometry_results.json"  # ablation/eval_stage2.py's output
MAST3R_JSON      = DINO_ROOT / "mast3r_stage3_results.json"    # MASt3R notebook's synced output

OUTPUT_JSON = DINO_ROOT / "statistics_final.json"
OUTPUT_MD   = DINO_ROOT / "statistics_final.md"


def load_stage3_from_statistics(path: Path) -> list[dict[str, Any]]:
    """Flatten statistics.py's per-shard mcnemar dicts into a flat list of raw
    tests, each already tagged family="stage3" by statistics.py itself."""
    if not path.exists():
        print(f"[aggregate] WARNING: {path} not found -- run `python ablation/statistics.py` "
              f"first. Stage 3 family will be built without B1/B2/B5/B6/B8/B11.")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    tests: list[dict[str, Any]] = []
    for shard_name, shard_data in data.get("results", {}).items():
        for row_name, mc in shard_data.get("mcnemar", {}).items():
            t = dict(mc)
            t.setdefault("row", row_name)
            t["shard"] = shard_name
            t.setdefault("family", "stage3")
            tests.append(t)
    print(f"[aggregate] Loaded {len(tests)} raw Stage 3 tests from {path.name} "
          f"(B1/B2/B5/B6/B8/B11 x {len(data.get('results', {}))} shard(s))")
    return tests


def load_stage3_from_mast3r(path: Path) -> list[dict[str, Any]]:
    """Load B10/MASt3R's raw tests, synced from the Colab notebook. Expected
    schema: a JSON list of test dicts (same shape as significance.mcnemar_exact's
    return value), each with "row" (e.g. "B10") and "shard" set, "family" optional
    (defaulted to "stage3" here if absent)."""
    if not path.exists():
        print(f"[aggregate] WARNING: {path} not found -- B10/MASt3R has not been "
              f"synced yet (or hasn't been re-run since the STATISTICS_METHODOLOGY "
              f"overhaul). Stage 3 family will be PROVISIONAL/INCOMPLETE without it.")
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    tests: list[dict[str, Any]] = []
    for t in raw:
        t = dict(t)
        t.setdefault("family", "stage3")
        t.setdefault("row", "B10")
        tests.append(t)
    print(f"[aggregate] Loaded {len(tests)} raw Stage 3 tests from {path.name} (B10)")
    return tests


def load_stage2_from_eval(path: Path) -> list[dict[str, Any]]:
    """Flatten eval_stage2.py's per-variant, per-shard mcnemar dicts into a flat
    list of raw tests, each already tagged family="stage2" by eval_stage2.py."""
    if not path.exists():
        print(f"[aggregate] WARNING: {path} not found -- run `python ablation/eval_stage2.py` "
              f"first (needs the Colab-produced C1/C2 manifests). Stage 2 family will be empty.")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    tests: list[dict[str, Any]] = []
    for variant in data.get("variants", []):
        for shard_name, shard_data in variant.get("shards", {}).items():
            mc = shard_data.get("mcnemar")
            if mc is None:
                continue
            t = dict(mc)
            t.setdefault("row", variant.get("name", "?"))
            t.setdefault("shard", shard_name)
            t.setdefault("family", "stage2")
            tests.append(t)
    print(f"[aggregate] Loaded {len(tests)} raw Stage 2 tests from {path.name} (C1/C2)")
    return tests


def render_markdown(tests: list[dict[str, Any]], family_expected: dict[str, set[str]]) -> str:
    lines = [
        "# Table B (final) — Holm-Bonferroni-corrected significance, by stage family",
        "",
        f"Generated: {date.today()}  |  alpha={ALPHA}",
        "",
        "Every row here is corrected against every other row in the SAME stage family",
        "(pooled across both shards) — see ablation/STATISTICS_METHODOLOGY.md. This is",
        "the file to cite; ablation/statistics.py's and ablation/eval_stage2.py's own",
        "outputs carry raw, uncorrected p-values only.",
        "",
    ]

    by_family: dict[str, list[dict[str, Any]]] = {}
    for t in tests:
        by_family.setdefault(t["family"], []).append(t)

    for family in sorted(by_family):
        rows_seen = {t["row"] for t in by_family[family]}
        expected = family_expected.get(family, set())
        missing = expected - rows_seen
        lines.append(f"## {family}")
        if missing:
            lines.append("")
            lines.append(f"**PROVISIONAL — missing from this family: {sorted(missing)}.** "
                          f"Holm correction below is computed over only the "
                          f"{len(rows_seen)} row(s) actually available; re-run once the "
                          f"missing source(s) are synced.")
        lines.append("")
        lines.append("| Row | Shard | b | c | p (raw) | Holm p | Significant |")
        lines.append("|---|---|---|---|---|---|---|")
        for t in sorted(by_family[family], key=lambda x: (x["row"], x.get("shard", ""))):
            sig = "**SIG**" if t.get("holm_reject") else "n.s."
            lines.append(
                f"| {t['row']} | {t.get('shard', '?')} | {t['b']} | {t['c']} | "
                f"{t['p_value']:.4f} | {t.get('holm_p', float('nan')):.4f} | {sig} |"
            )
        lines.append("")

    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--statistics-results", default=str(STATISTICS_JSON))
    p.add_argument("--stage2-results", default=str(STAGE2_JSON))
    p.add_argument("--mast3r-results", default=str(MAST3R_JSON))
    p.add_argument("--output-json", default=str(OUTPUT_JSON))
    p.add_argument("--output-md", default=str(OUTPUT_MD))
    args = p.parse_args(argv)

    tests: list[dict[str, Any]] = []
    tests += load_stage3_from_statistics(Path(args.statistics_results))
    tests += load_stage3_from_mast3r(Path(args.mast3r_results))
    tests += load_stage2_from_eval(Path(args.stage2_results))

    if not tests:
        print("[aggregate] No raw tests loaded from any source -- nothing to correct. "
              "Run the upstream scripts first.")
        return

    # Expected family membership, per STATISTICS_METHODOLOGY.md -- used only to
    # print a loud, explicit "still missing" warning, never to silently proceed.
    family_expected = {
        "stage3": {"B1", "B2", "B5", "B6", "B8", "B11", "B10"},
        "stage2": {"C1 (LightGlue)", "C2 (RoMa)"},
    }

    holm_correct(tests, family_key="family")

    print("\n[aggregate] Family summary:")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for t in tests:
        by_family.setdefault(t["family"], []).append(t)
    for family, family_tests in sorted(by_family.items()):
        rows_seen = {t["row"] for t in family_tests}
        expected = family_expected.get(family, set())
        missing = expected - rows_seen
        status = "PROVISIONAL (missing: %s)" % sorted(missing) if missing else "complete"
        n_sig = sum(1 for t in family_tests if t.get("holm_reject"))
        print(f"  {family}: {len(family_tests)} tests, {len(rows_seen)} row(s) "
              f"({sorted(rows_seen)}), {n_sig} significant after Holm — {status}")

    md = render_markdown(tests, family_expected)
    print("\n" + md)

    output = {
        "metadata": {
            "date": str(date.today()),
            "alpha": ALPHA,
            "family_expected": {k: sorted(v) for k, v in family_expected.items()},
        },
        "tests": tests,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(f"\n[aggregate] Written: {out_json}")
    print(f"[aggregate] Written: {args.output_md}")


if __name__ == "__main__":
    main()
