"""Task 5 — Bootstrap 95% CIs and McNemar tests for Table B.

Re-derives per-pair predictions from stored manifests — no model rerun.
  B1/B2/B4/B5/B6  from aspan_all_manifest
  B8               via cv2.decomposeHomographyMat on stored homographies
  B11              via LR (vggt_aggr) retrained on Shard1, evaluated on Shard2

McNemar tests compare each ablation row vs. B4 on the intersection of
pair sets. Holm-Bonferroni correction controls family-wise error rate across
all rows within each shard.

Outputs:
    D:/DINO OUTPUTS/statistics_results.json
    D:/DINO OUTPUTS/statistics_table.md

Usage:
    python ablation/statistics.py
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import date
from math import comb
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))              # pose_scoring is at repo root
sys.path.insert(0, str(Path(__file__).parent))  # ablation_utils is in ablation/
sys.path.insert(0, str(_REPO / "_local"))   # classical_pose stays in _local/
from ablation_utils import (
    DINO_ROOT, SHARDS, load_ground_truth, load_aspan_all, load_judge_manifest,
    compute_metrics, pr_auc, EXCLUDE_LABELS,
)
from pose_scoring import score_row, INLIER_RATIO_THRESHOLD, POSE_COMPONENT_THRESHOLD
from classical_pose import decompose_H

OUTPUT_JSON = DINO_ROOT / "statistics_results.json"
OUTPUT_MD   = DINO_ROOT / "statistics_table.md"
B8_JSON     = DINO_ROOT / "b8_results.json"
ABL_JSON    = DINO_ROOT / "ablation_results.json"

N_BOOT    = 10_000
BOOT_SEED = 42
ALPHA     = 0.05

PairMap = dict[tuple[str, str], dict]  # {(sid, tid): {y_true, y_pred, score}}


# ── Statistical primitives ─────────────────────────────────────────────────────

def chi2_sf_1dof(stat: float) -> float:
    """Exact survival function of chi-squared(1): erfc(sqrt(stat/2))."""
    if stat <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(stat / 2.0))


def binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value (for b+c < 25)."""
    n = b + c
    hi = max(b, c)
    half_n = 0.5 ** n
    tail = sum(comb(n, k) * half_n for k in range(hi, n + 1))
    return min(2.0 * tail, 1.0)


def bootstrap_ci(
    y_true: list[int],
    y_pred: list[int],
    scores: list[float] | None = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Bootstrap 95% CIs for P, R, F1 (from binary preds) and PR-AUC (from scores)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt = np.array(y_true, dtype=np.int8)
    yp = np.array(y_pred, dtype=np.int8)
    sc = np.array(scores, dtype=np.float64) if scores is not None else None

    pres, recs, f1s, aucs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_b, yp_b = yt[idx], yp[idx]
        m = compute_metrics(yt_b.tolist(), yp_b.tolist())
        pres.append(m["precision"])
        recs.append(m["recall"])
        f1s.append(m["f1"])
        if sc is not None:
            sc_b = sc[idx]
            aucs.append(pr_auc(yt_b.tolist(), sc_b.tolist()))

    def ci(vals: list[float]) -> tuple[float, float]:
        v = [x for x in vals if not math.isnan(x)]
        if not v:
            return (float("nan"), float("nan"))
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    out: dict = {"precision_ci": ci(pres), "recall_ci": ci(recs), "f1_ci": ci(f1s)}
    if sc is not None:
        out["pr_auc_ci"] = ci(aucs)
    return out


def mcnemar_test(
    y_true: list[int],
    pred_a: list[int],   # ablation row (challenger)
    pred_b: list[int],   # B4 reference
) -> dict:
    """Two-sided McNemar's test: does the challenger differ from B4?

    b = challenger correct & B4 wrong  (challenger outperforms on this pair)
    c = challenger wrong  & B4 correct (B4 outperforms on this pair)
    """
    b = sum(1 for yt, pa, pb in zip(y_true, pred_a, pred_b) if pa == yt and pb != yt)
    c = sum(1 for yt, pa, pb in zip(y_true, pred_a, pred_b) if pa != yt and pb == yt)
    n_dis = b + c

    direction = "challenger_better" if b > c else ("b4_better" if c > b else "tie")

    if n_dis == 0:
        return {"b": 0, "c": 0, "stat": 0.0, "p_value": 1.0,
                "method": "equal", "direction": "equal"}

    if n_dis < 25:
        p = binom_two_sided_p(b, c)
        return {"b": b, "c": c, "stat": float(max(b, c)), "p_value": float(p),
                "method": "exact_binomial", "direction": direction}

    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = chi2_sf_1dof(stat)
    return {"b": b, "c": c, "stat": float(stat), "p_value": float(p),
            "method": "chi2_continuity", "direction": direction}


def holm_correct(tests: list[dict]) -> None:
    """Add holm_p and holm_reject to each McNemar result dict (in-place)."""
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i]["p_value"])
    max_adj = 0.0
    for rank, idx in enumerate(order):
        adj = min(tests[idx]["p_value"] * (m - rank), 1.0)
        adj = max(adj, max_adj)
        tests[idx]["holm_p"] = round(float(adj), 6)
        tests[idx]["holm_reject"] = bool(adj < ALPHA)
        max_adj = adj


# ── Per-pair prediction derivers ───────────────────────────────────────────────

def _y(label: str | None) -> int | None:
    if label is None or label.lower() in EXCLUDE_LABELS:
        return None
    if label == "Positive":
        return 1
    if label == "Negative":
        return 0
    return None


_SCORE_DEFAULTS = dict(
    inlier_ratio_threshold=INLIER_RATIO_THRESHOLD,
    pose_component_threshold=POSE_COMPONENT_THRESHOLD,
    global_sim_threshold=None,
    pose_components="all",
    keypoint_floor=0,
)


def pairs_b_variants(shard_name: str) -> dict[str, PairMap]:
    """B2/B4/B5/B6: per-pair predictions from aspan_all (VGGT-processed rows only)."""
    paths = SHARDS[shard_name]
    gt = load_ground_truth(paths["manifest_csv"])
    aspan_all = load_aspan_all(paths["aspan_all"])

    out: dict[str, PairMap] = {"B2": {}, "B4": {}, "B5": {}, "B6": {}}

    for (sid, tid), row in aspan_all.items():
        if "aspan_2d_inlier_ratio" not in row:
            continue
        yt = _y(gt.get((sid, tid)))
        if yt is None:
            continue

        ir = float(row.get("aspan_2d_inlier_ratio") or 0.0)
        ps = float(row.get("pose_component_score") or 9.0)
        soft_combined = ir - ps / 10.0

        for row_label, overrides, score in [
            ("B2", {"pose_component_threshold": 0.0},           ir),
            ("B4", {},                                           soft_combined),
            ("B5", {"pose_components": "rotation_xy"},          soft_combined),
            ("B6", {"pose_components": "fov_z"},                soft_combined),
        ]:
            pred, _ = score_row(row, **{**_SCORE_DEFAULTS, **overrides})
            out[row_label][(sid, tid)] = {"y_true": yt, "y_pred": int(pred), "score": score}

    return out


def pairs_b1(shard_name: str, kp_threshold: int) -> PairMap:
    """B1: keypoint-floor decision at fixed threshold from ablation_runner output."""
    paths = SHARDS[shard_name]
    gt = load_ground_truth(paths["manifest_csv"])
    aspan_all = load_aspan_all(paths["aspan_all"])

    result: PairMap = {}
    for (sid, tid), label in gt.items():
        yt = _y(label)
        if yt is None:
            continue
        row = aspan_all.get((sid, tid))
        if row is None:
            continue
        kp = int(row.get("raw_keypoint_count") or row.get("filtered_keypoint_count", 0))
        result[(sid, tid)] = {"y_true": yt, "y_pred": int(kp >= kp_threshold), "score": float(kp)}
    return result


def pairs_b8(shard_name: str, rot_threshold: float) -> PairMap:
    """B8: homography decomposition — predicted positive if rotation <= threshold."""
    paths = SHARDS[shard_name]
    gt = load_ground_truth(paths["manifest_csv"])
    judge = load_judge_manifest(paths["judge_jsonl"])

    result: PairMap = {}
    for (sid, tid), label in gt.items():
        yt = _y(label)
        if yt is None:
            continue
        row = judge.get((sid, tid))
        if row is None:
            continue
        H_list = row.get("alignment_homography")
        sz = row.get("alignment_source_resized_size")
        if not H_list or not sz:
            continue
        rot_deg = decompose_H(H_list, sz, focal_mult=1.0)
        if rot_deg is None:
            continue
        result[(sid, tid)] = {
            "y_true": yt,
            "y_pred": int(rot_deg <= rot_threshold),
            "score": -rot_deg,   # negated: higher score = more likely positive
        }
    return result


def pairs_b11_shard2() -> PairMap:
    """B11: LR (vggt_aggr) retrained on Shard1, evaluated on Shard2.

    Features: [kp_count, inlier_ratio, global_similarity, pose_component_score].
    Threshold: cross-validated F1-optimal on Shard1.
    Returns Shard2 per-pair predictions only.
    """
    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError:
        print("  [B11] scikit-learn not available — skipping B11")
        return {}

    def extract(row: dict) -> np.ndarray | None:
        try:
            return np.array([
                float(row["raw_keypoint_count"]),
                float(row["aspan_2d_inlier_ratio"]),
                float(row["global_similarity"]),
                float(row["pose_component_score"]),
            ], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None

    def load_shard(name: str):
        paths = SHARDS[name]
        gt = load_ground_truth(paths["manifest_csv"])
        aspan = load_aspan_all(paths["aspan_all"])
        keys, X_rows, ys = [], [], []
        for (sid, tid), row in aspan.items():
            if "aspan_2d_inlier_ratio" not in row or "global_similarity" not in row:
                continue
            yt = _y(gt.get((sid, tid)))
            if yt is None:
                continue
            f = extract(row)
            if f is None:
                continue
            keys.append((sid, tid))
            X_rows.append(f)
            ys.append(yt)
        return keys, np.array(X_rows), np.array(ys)

    # Build and fit on Shard1
    keys1, X1, y1 = load_shard("Shard1")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=cv, scoring="f1",
            class_weight="balanced", max_iter=1_000,
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X1, y1)
        cv_proba = cross_val_predict(pipe, X1, y1, cv=cv, method="predict_proba")[:, 1]

    # CV-optimal threshold from Shard1
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        yp = (cv_proba >= t).astype(int).tolist()
        m = compute_metrics(y1.tolist(), yp)
        if not math.isnan(m["f1"]) and m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)

    # Predict on Shard2
    keys2, X2, y2 = load_shard("Shard2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba2 = pipe.predict_proba(X2)[:, 1]

    result: PairMap = {}
    for key, yt, prob in zip(keys2, y2, proba2):
        result[key] = {
            "y_true": int(yt),
            "y_pred": int(prob >= best_t),
            "score": float(prob),
        }
    print(f"  [B11] cv_threshold={best_t:.2f}  "
          f"cv_S1_F1={best_f1:.3f}  n_S2={len(result)}")
    return result


# ── Per-shard evaluation ───────────────────────────────────────────────────────

def _flatten(pairs: PairMap) -> tuple[list, list, list]:
    yt = [v["y_true"] for v in pairs.values()]
    yp = [v["y_pred"] for v in pairs.values()]
    sc = [v["score"]  for v in pairs.values()]
    return yt, yp, sc


def _mcnemar_vs_b4(
    challenger: PairMap,
    b4: PairMap,
    row_name: str,
) -> dict:
    common = sorted(set(challenger) & set(b4))
    if not common:
        print(f"  [McNemar/{row_name}] 0 pairs in common with B4 — skipping")
        return {"b": 0, "c": 0, "stat": 0.0, "p_value": 1.0, "n_common": 0,
                "method": "no_common_pairs", "direction": "unknown"}
    yt  = [b4[k]["y_true"]          for k in common]
    pa  = [challenger[k]["y_pred"]   for k in common]
    pb  = [b4[k]["y_pred"]           for k in common]
    res = mcnemar_test(yt, pa, pb)
    res["n_common"] = len(common)
    return res


def eval_shard(
    shard_name: str,
    b1_threshold: int,
    b8_threshold: float,
    b11_pairs: PairMap,
) -> dict:
    print(f"\n[{shard_name}]")

    # Load all predictions
    variants  = pairs_b_variants(shard_name)
    b4_pairs  = variants["B4"]
    b1_pairs  = pairs_b1(shard_name, b1_threshold)
    b8_pairs  = pairs_b8(shard_name, b8_threshold)

    row_maps: dict[str, PairMap] = {
        "B1": b1_pairs,
        "B2": variants["B2"],
        "B4": b4_pairs,
        "B5": variants["B5"],
        "B6": variants["B6"],
        "B8": b8_pairs,
    }
    if shard_name == "Shard2" and b11_pairs:
        row_maps["B11"] = b11_pairs

    # Bootstrap CIs per row
    ci_results: dict[str, dict] = {}
    for row_name, pairs in row_maps.items():
        if not pairs:
            continue
        yt, yp, sc = _flatten(pairs)
        pt = compute_metrics(yt, yp)
        pt["pr_auc"] = pr_auc(yt, sc)
        ci = bootstrap_ci(yt, yp, sc)
        ci_results[row_name] = {"point": pt, "ci": ci, "n": len(yt)}
        print(f"  [{row_name}] n={len(yt):3d}  "
              f"F1={pt['f1']:.3f}  CI=[{ci['f1_ci'][0]:.3f}, {ci['f1_ci'][1]:.3f}]  "
              f"P={pt['precision']:.3f}  R={pt['recall']:.3f}")

    # McNemar tests: each row vs. B4
    mc_tests: list[dict] = []
    for row_name in ["B1", "B2", "B5", "B6", "B8", "B11"]:
        if row_name not in row_maps:
            continue
        test = _mcnemar_vs_b4(row_maps[row_name], b4_pairs, row_name)
        test["row"] = row_name
        mc_tests.append(test)
        sig = "**" if test["p_value"] < ALPHA else "n.s."
        print(f"  [McNemar/{row_name}vsB4]  "
              f"n_common={test['n_common']}  b={test['b']}  c={test['c']}  "
              f"p={test['p_value']:.4f} {sig}  dir={test.get('direction','?')}")

    holm_correct(mc_tests)
    for t in mc_tests:
        holm_sig = "SIG" if t.get("holm_reject") else "n.s."
        print(f"  [Holm/{t['row']}]  holm_p={t.get('holm_p','?'):.4f}  {holm_sig}")

    return {
        "bootstrap": ci_results,
        "mcnemar": {t["row"]: t for t in mc_tests},
    }


# ── Output ─────────────────────────────────────────────────────────────────────

def _fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if not math.isnan(v) else "—"


def write_markdown(results: dict, b1_thresholds: dict, b8_thresholds: dict, out_path: Path) -> None:
    shards = list(SHARDS.keys())
    lines = [
        "# Table B — Statistical Results",
        "",
        f"Generated: {date.today()}",
        f"Bootstrap: {N_BOOT:,} resamples, seed={BOOT_SEED}, α={ALPHA}",
        "",
        "## 1. Bootstrap 95% Confidence Intervals",
        "",
        "F1 point estimate [95% CI]; PR-AUC point estimate [95% CI]",
        "",
        "| Row | " + " | ".join(f"{s} F1 [95%CI] | {s} PR-AUC [95%CI] | {s} P [95%CI] | {s} R [95%CI]" for s in shards) + " |",
        "|---|" + "".join("|---|---|---|---|" for _ in shards),
    ]

    for row_name in ["B1", "B2", "B4", "B5", "B6", "B8", "B11"]:
        cells = [f"**{row_name}**"]
        for shard in shards:
            r = results.get(shard, {}).get("bootstrap", {}).get(row_name)
            if r is None:
                cells += ["—", "—", "—", "—"]
                continue
            pt, ci = r["point"], r["ci"]
            f1lo, f1hi   = ci["f1_ci"]
            plo, phi     = ci["precision_ci"]
            rlo, rhi     = ci["recall_ci"]
            auc_ci       = ci.get("pr_auc_ci", (float("nan"), float("nan")))

            cells.append(f"{_fmt(pt['f1'])} [{_fmt(f1lo)}, {_fmt(f1hi)}]")
            cells.append(f"{_fmt(pt['pr_auc'],4)} [{_fmt(auc_ci[0],4)}, {_fmt(auc_ci[1],4)}]")
            cells.append(f"{_fmt(pt['precision'])} [{_fmt(plo)}, {_fmt(phi)}]")
            cells.append(f"{_fmt(pt['recall'])} [{_fmt(rlo)}, {_fmt(rhi)}]")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"B1 thresholds used: {b1_thresholds} (keypoint floor, from ablation_results.json)",
        f"B8 thresholds used: {b8_thresholds} (rotation_deg <=, from b8_results.json)",
        "",
        "## 2. McNemar Tests vs. B4",
        "",
        "b = challenger correct & B4 wrong; c = B4 correct & challenger wrong",
        "",
        "| Row vs B4 | " + " | ".join(f"{s} b/c | {s} p | {s} holm-p | {s} sig | {s} direction" for s in shards) + " |",
        "|---|" + "".join("|---|---|---|---|---|" for _ in shards),
    ]

    for row_name in ["B1", "B2", "B5", "B6", "B8", "B11"]:
        cells = [f"**{row_name}**"]
        for shard in shards:
            mc = results.get(shard, {}).get("mcnemar", {}).get(row_name)
            if mc is None:
                cells += ["—", "—", "—", "—", "—"]
                continue
            bc   = f"{mc['b']}/{mc['c']} (n={mc['n_common']})"
            p    = f"{mc['p_value']:.4f}"
            holm = f"{mc.get('holm_p', float('nan')):.4f}"
            sig  = "SIG *" if mc.get("holm_reject") else "n.s."
            dirn = mc.get("direction", "?")
            cells += [bc, p, holm, sig, dirn]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Notes",
        f"- McNemar with continuity correction (chi-squared, n≥25) or exact binomial (n<25)",
        f"- Holm correction applied across B1/B2/B5/B6/B8[/B11] within each shard independently",
        "- B11 Shard1: omitted — LR is trained on Shard1; Shard2 is the out-of-sample evaluation",
        "- SIG * = significant after Holm correction",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown written to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load best thresholds from prior task outputs
    b1_thresholds: dict[str, int] = {"Shard1": 50, "Shard2": 50}
    if ABL_JSON.exists():
        abl = json.loads(ABL_JSON.read_text(encoding="utf-8"))
        for shard in SHARDS:
            kf = abl.get("rows", {}).get("B1", {}).get(shard, {}).get("best", {}).get("keypoint_floor")
            if kf is not None:
                b1_thresholds[shard] = int(kf)

    b8_thresholds: dict[str, float] = {"Shard1": 2.5, "Shard2": 3.0}
    if B8_JSON.exists():
        b8j = json.loads(B8_JSON.read_text(encoding="utf-8"))
        for shard in SHARDS:
            td = (b8j.get("results", {})
                      .get(shard, {})
                      .get("f1.0", {})
                      .get("best", {})
                      .get("threshold_deg"))
            if td is not None:
                b8_thresholds[shard] = float(td)

    print(f"B1 kp thresholds : {b1_thresholds}")
    print(f"B8 rot thresholds: {b8_thresholds}")

    # B11: retrain on Shard1, predict Shard2 (once, then pass to both shard evals)
    print("\n[B11] Retraining LR on Shard1 ...")
    b11_shard2 = pairs_b11_shard2()

    # Per-shard evaluation
    all_results: dict[str, dict] = {}
    for shard_name in SHARDS:
        all_results[shard_name] = eval_shard(
            shard_name,
            b1_threshold=b1_thresholds[shard_name],
            b8_threshold=b8_thresholds[shard_name],
            b11_pairs=b11_shard2,
        )

    # Serialise results
    output = {
        "metadata": {
            "date":            str(date.today()),
            "n_boot":          N_BOOT,
            "boot_seed":       BOOT_SEED,
            "alpha":           ALPHA,
            "b1_thresholds":   b1_thresholds,
            "b8_thresholds":   b8_thresholds,
            "inlier_threshold": INLIER_RATIO_THRESHOLD,
            "pose_threshold":   POSE_COMPONENT_THRESHOLD,
        },
        "results": all_results,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults JSON written to {OUTPUT_JSON}")

    write_markdown(all_results, b1_thresholds, b8_thresholds, OUTPUT_MD)
    print("Done.")


if __name__ == "__main__":
    main()
