"""Task 4 — B11: Learned classifier over stored pipeline signals.

Replaces the hand-set threshold pair (inlier_ratio >= 0.65, pose_score <= 2.13)
with learned decision boundaries. Answers: "Why a hand-tuned threshold?"

Protocol:
    Train  — Shard 1 (Positive/Negative; exclude Unsure/Unknown)
    Eval   — Shard 2 (held-out, frozen after training)
    Models — (a) L2-regularized logistic regression (primary, interpretable)
              (b) MLP 64→32→1 via sklearn (secondary, checks for non-linearity)

Feature groups:
    Geometric  — raw_keypoint_count, aspan_2d_inlier_ratio
    Global     — global_similarity
    Pose aggr. — pose_component_score
    Pose comps — pose_rotation_deg, pose_translation_xy_l2,
                  pose_translation_z_abs, pose_fov_l2
    Raw deltas — |pose_src[i] - pose_tgt[i]| for i=0..8  (9 dims)

Relocated from `_local/learned_classifier.py` 2026-07-16 (git-tracked from here on —
`_local/` is gitignored, which is how the previous run's model became unrecoverable;
see `ablation/STATISTICS_METHODOLOGY.md`). Two real bugs fixed at the same time:

  1. `LogisticRegressionCV(cv=CV_FOLDS)` used to pass a bare int. Per sklearn's
     `check_cv()`, a bare int silently defaults to `shuffle=False` and never receives
     the estimator's `random_state` — so the regularization-strength (C) selection
     was tuned on unshuffled, sequential-block folds, not a randomized split. Fixed:
     always pass an explicit `StratifiedKFold(shuffle=True, random_state=...)` object.
  2. Feature source switched from `Shard{N} Judge Manifest.jsonl` to
     `aspan_all_manifest_shard{N}.jsonl`, matching this project's canonical B4/raw-
     measurement source (see STATISTICS_METHODOLOGY.md) — empirically identical for
     today's labeled subset, but now enforced as one source, not two independently
     agreeing by coincidence.

Also fixed: `max_iter` unified to 2000 everywhere (was 1000 in the old inline retrain
that used to live in `statistics.py`); the `no_vggt` feature group — byte-identical to
`geom_only`, silently dropped from the table — removed rather than computed and discarded.

New in this version: every variant's per-pair predictions (Shard1-CV and Shard2-test,
keyed by (source_id, target_id)) are persisted to `b11_predictions/`, and every fitted
pipeline is joblib-dumped to `b11_models/` alongside its training metadata. This is what
was missing before — no downstream script should ever need to retrain B11 again; they
should load these artifacts instead.

Outputs:
    D:/DINO OUTPUTS/b11_results.json
    D:/DINO OUTPUTS/b11_table.md
    D:/DINO OUTPUTS/b11_predictions/{variant}.jsonl   (per-pair, both shards)
    D:/DINO OUTPUTS/b11_models/{variant}.joblib       (fitted pipeline)

Usage:
    python ablation/learned_classifier.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from ablation_utils import (
    DINO_ROOT, SHARDS, load_ground_truth, load_aspan_all,
    compute_metrics, pr_auc, roc_auc, EXCLUDE_LABELS,
)

OUTPUT_JSON = DINO_ROOT / "b11_results.json"
OUTPUT_MD   = DINO_ROOT / "b11_table.md"
PRED_DIR    = DINO_ROOT / "b11_predictions"
MODEL_DIR   = DINO_ROOT / "b11_models"

RANDOM_STATE = 42
CV_FOLDS = 5
MAX_ITER = 2000

# B4 reference (Task 1 / ablation_runner.py) — informational only, for the printed
# comparison table; not used in any computation here.
B4_REFERENCE = {
    "Shard1": {"precision": 0.867, "recall": 0.963, "f1": 0.913, "pr_auc": 0.9640},
    "Shard2": {"precision": 0.902, "recall": 0.984, "f1": 0.941, "pr_auc": 0.9932},
}

FEATURE_NAMES = [
    "kp_count",
    "inlier_ratio",
    "global_sim",
    "pose_score",
    "rot_deg",
    "xy_l2",
    "z_abs",
    "fov_l2",
    "delta_0", "delta_1", "delta_2",
    "delta_3", "delta_4", "delta_5",
    "delta_6", "delta_7", "delta_8",
]

# Feature groups for subset experiments. ("no_vggt" from the original script was a
# byte-identical duplicate of "geom_only" that never appeared in the table — removed.)
FEATURE_GROUPS = {
    "geom_only":   [0, 1],                  # kp_count, inlier_ratio
    "vggt_aggr":   [0, 1, 2, 3],            # + global_sim, pose_score  (the headline row)
    "vggt_comps":  [0, 1, 4, 5, 6, 7],      # + 4 pose components
    "all_feats":   list(range(17)),          # everything
}


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(row: dict) -> np.ndarray | None:
    """Extract feature vector from one manifest row. Returns None if any
    required field is missing."""
    try:
        kp    = float(row["raw_keypoint_count"])
        ir    = float(row["aspan_2d_inlier_ratio"])
        gs    = float(row["global_similarity"])
        ps    = float(row["pose_component_score"])
        rd    = float(row["pose_rotation_deg"])
        xy    = float(row["pose_translation_xy_l2"])
        z     = float(row["pose_translation_z_abs"])
        fov   = float(row["pose_fov_l2"])
        src   = np.array(row["pose_src"], dtype=float)
        tgt   = np.array(row["pose_tgt"], dtype=float)
        delta = np.abs(src - tgt)          # shape (9,)
        if delta.shape != (9,):
            return None
        return np.array([kp, ir, gs, ps, rd, xy, z, fov, *delta])
    except (KeyError, TypeError, ValueError):
        return None


def load_dataset(shard_name: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Returns (X, y, keys) for all Positive/Negative labeled pairs in the shard.

    keys[i] == (source_id, target_id) for X[i]/y[i] — needed so predictions can be
    written back out per-pair, not just as an anonymous array.
    """
    paths = SHARDS[shard_name]
    gt    = load_ground_truth(paths["manifest_csv"])
    aspan = load_aspan_all(paths["aspan_all"])

    X_rows, y_rows, keys = [], [], []
    n_skip = 0
    for (sid, tid), label in gt.items():
        if label.lower() in EXCLUDE_LABELS or label not in ("Positive", "Negative"):
            continue
        row = aspan.get((sid, tid))
        if row is None:
            n_skip += 1
            continue
        feats = extract_features(row)
        if feats is None:
            n_skip += 1
            continue
        X_rows.append(feats)
        y_rows.append(1 if label == "Positive" else 0)
        keys.append((sid, tid))

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    n_pos = y.sum()
    print(f"  [{shard_name}] n={len(y)} (pos={n_pos}, neg={len(y)-n_pos})  skipped={n_skip}")
    return X, y, keys


# ── Model training & evaluation ────────────────────────────────────────────────

def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Sweep decision threshold on predicted probabilities; return (best_t, best_f1)."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        y_pred = (proba >= t).astype(int)
        m = compute_metrics(y_true.tolist(), y_pred.tolist())
        f1 = m["f1"]
        if not np.isnan(f1) and f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (proba >= threshold).astype(int)
    m = compute_metrics(y_true.tolist(), y_pred.tolist())
    m["pr_auc"]   = pr_auc(y_true.tolist(), proba.tolist())
    m["roc_auc"]  = roc_auc(y_true.tolist(), proba.tolist())
    m["threshold"] = threshold
    return m


def _cv_splitter() -> StratifiedKFold:
    """The one CV-splitting object used everywhere in this script — always an
    explicit, shuffled, seeded StratifiedKFold, never a bare int (see module
    docstring, bug #1)."""
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def persist_predictions(
    variant: str,
    keys_train: list[tuple[str, str]], y_train: np.ndarray, cv_proba: np.ndarray, cv_threshold: float,
    keys_test: list[tuple[str, str]], y_test: np.ndarray, test_proba: np.ndarray,
) -> Path:
    """Write per-pair predictions for both shards to one JSONL sidecar. This is the
    artifact downstream scripts (statistics.py, aggregate_significance.py) should
    load instead of ever retraining B11 themselves."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"{variant}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for (sid, tid), yt, p in zip(keys_train, y_train, cv_proba):
            f.write(json.dumps({
                "role": "shard1_cv", "source_id": sid, "target_id": tid,
                "y_true": int(yt), "proba": float(p),
                "y_pred_default": int(p >= 0.5), "y_pred_tuned": int(p >= cv_threshold),
            }, ensure_ascii=False) + "\n")
        for (sid, tid), yt, p in zip(keys_test, y_test, test_proba):
            f.write(json.dumps({
                "role": "shard2_test", "source_id": sid, "target_id": tid,
                "y_true": int(yt), "proba": float(p),
                "y_pred_default": int(p >= 0.5), "y_pred_tuned": int(p >= cv_threshold),
            }, ensure_ascii=False) + "\n")
    return out_path


def run_lr(
    X_train: np.ndarray, y_train: np.ndarray, keys_train: list[tuple[str, str]],
    X_test: np.ndarray, y_test: np.ndarray, keys_test: list[tuple[str, str]],
    feat_indices: list[int], variant: str,
) -> dict:
    """Logistic regression with LRCV (picks best C via 5-fold CV on train set)."""
    Xtr = X_train[:, feat_indices]
    Xte = X_test[:,  feat_indices]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegressionCV(
            Cs=10, cv=_cv_splitter(), scoring="f1",
            random_state=RANDOM_STATE, max_iter=MAX_ITER,
            class_weight="balanced",
        )),
    ])
    pipe.fit(Xtr, y_train)

    # CV predictions on train set (already done internally by LRCV, but we want
    # the full cross-validated proba for train-set reporting). Reuses the SAME
    # splitter object as the LRCV fit above, not a second independently-constructed
    # one, so both levels of CV agree on fold membership.
    cv_proba = cross_val_predict(
        Pipeline([("scaler", StandardScaler()),
                  ("lr",     pipe.named_steps["lr"])]),
        Xtr, y_train,
        cv=_cv_splitter(),
        method="predict_proba",
    )[:, 1]
    cv_t, _ = best_f1_threshold(y_train, cv_proba)
    cv_metrics = evaluate(y_train, cv_proba, cv_t)

    # Held-out test set
    test_proba = pipe.predict_proba(Xte)[:, 1]
    test_default = evaluate(y_test, test_proba, 0.5)
    test_tuned   = evaluate(y_test, test_proba, cv_t)     # threshold from CV

    pred_path = persist_predictions(
        variant, keys_train, y_train, cv_proba, cv_t, keys_test, y_test, test_proba,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{variant}.joblib"
    joblib.dump(pipe, model_path)

    # Feature importances (standardised coefficients)
    lr_coef = pipe.named_steps["lr"].coef_[0]
    importances = sorted(
        zip([FEATURE_NAMES[i] for i in feat_indices], lr_coef),
        key=lambda t: abs(t[1]), reverse=True,
    )

    best_C = float(pipe.named_steps["lr"].C_[0])
    return {
        "best_C":           best_C,
        "cv_threshold":     round(cv_t, 2),
        "cv_metrics":       cv_metrics,
        "test_default":     test_default,   # threshold=0.5
        "test_tuned":       test_tuned,     # threshold from CV
        "importances":      importances,
        "predictions_path": str(pred_path),
        "model_path":       str(model_path),
    }


def run_mlp(
    X_train: np.ndarray, y_train: np.ndarray, keys_train: list[tuple[str, str]],
    X_test: np.ndarray, y_test: np.ndarray, keys_test: list[tuple[str, str]],
    feat_indices: list[int], variant: str,
) -> dict:
    """Small MLP (64→32→1) with L2 regularisation via sklearn."""
    Xtr = X_train[:, feat_indices]
    Xte = X_test[:,  feat_indices]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.1,              # L2 weight decay (higher = more regularised)
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(Xtr, y_train)

    cv_proba = cross_val_predict(
        Pipeline([("scaler", StandardScaler()),
                  ("mlp",    MLPClassifier(
                      hidden_layer_sizes=(64, 32), activation="relu",
                      alpha=0.1, max_iter=500, random_state=RANDOM_STATE))]),
        Xtr, y_train,
        cv=_cv_splitter(),
        method="predict_proba",
    )[:, 1]
    cv_t, _ = best_f1_threshold(y_train, cv_proba)
    cv_metrics = evaluate(y_train, cv_proba, cv_t)

    test_proba   = pipe.predict_proba(Xte)[:, 1]
    test_default = evaluate(y_test, test_proba, 0.5)
    test_tuned   = evaluate(y_test, test_proba, cv_t)

    pred_path = persist_predictions(
        variant, keys_train, y_train, cv_proba, cv_t, keys_test, y_test, test_proba,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{variant}.joblib"
    joblib.dump(pipe, model_path)

    return {
        "cv_threshold": round(cv_t, 2),
        "cv_metrics":   cv_metrics,
        "test_default": test_default,
        "test_tuned":   test_tuned,
        "predictions_path": str(pred_path),
        "model_path":       str(model_path),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[B11] Loading datasets")
    X_train, y_train, keys_train = load_dataset("Shard1")
    X_test,  y_test,  keys_test  = load_dataset("Shard2")

    results: dict = {}

    print("\n[B11] Logistic regression — feature group sweep")
    results["lr"] = {}
    for group_name, feat_idx in FEATURE_GROUPS.items():
        print(f"\n  --- LR / {group_name} ---")
        r = run_lr(X_train, y_train, keys_train, X_test, y_test, keys_test,
                   feat_idx, variant=f"lr_{group_name}")
        results["lr"][group_name] = r
        td = r["test_tuned"]
        print(f"  S2 (cv-tuned threshold={r['cv_threshold']:.2f}):  "
              f"P={td['precision']:.3f}  R={td['recall']:.3f}  F1={td['f1']:.3f}  "
              f"PR-AUC={td['pr_auc']:.4f}  [C={r['best_C']:.4f}]")
        print(f"  Top features: " +
              "  ".join(f"{n}={v:+.3f}" for n, v in r["importances"][:5]))

    print("\n[B11] MLP — all features only (§D2: primary result is LR)")
    mlp_r = run_mlp(X_train, y_train, keys_train, X_test, y_test, keys_test,
                     FEATURE_GROUPS["all_feats"], variant="mlp_all_feats")
    results["mlp"] = {"all_feats": mlp_r}
    td = mlp_r["test_tuned"]
    print(f"  S2 (cv-tuned threshold={mlp_r['cv_threshold']:.2f}):  "
          f"P={td['precision']:.3f}  R={td['recall']:.3f}  F1={td['f1']:.3f}  "
          f"PR-AUC={td['pr_auc']:.4f}")

    # Summary comparison
    print("\n" + "=" * 75)
    print("SUMMARY — Shard 2 (held-out), cv-tuned threshold")
    print("=" * 75)
    ref = B4_REFERENCE["Shard2"]
    print(f"  B4  (paper rule)        F1={ref['f1']:.3f}  PR-AUC={ref['pr_auc']:.4f}")
    for g in ["geom_only", "vggt_aggr", "vggt_comps", "all_feats"]:
        r = results["lr"][g]["test_tuned"]
        print(f"  LR  ({g:<12s})   F1={r['f1']:.3f}  PR-AUC={r['pr_auc']:.4f}")
    mlp_td = results["mlp"]["all_feats"]["test_tuned"]
    print(f"  MLP (all_feats)         F1={mlp_td['f1']:.3f}  PR-AUC={mlp_td['pr_auc']:.4f}")
    print()
    print("  Full importances (LR / all_feats, standardised coef):")
    for name, coef in results["lr"]["all_feats"]["importances"]:
        bar = "#" * int(abs(coef) * 8)
        sign = "+" if coef > 0 else "-"
        print(f"    {name:<12s}  {sign}{abs(coef):.4f}  {bar}")

    output = {
        "metadata": {
            "date": str(date.today()),
            "train_shard": "Shard1",
            "eval_shard": "Shard2",
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
            "feature_source": "aspan_all_manifest_shard{N}.jsonl",
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "b4_reference": B4_REFERENCE,
            "feature_names": FEATURE_NAMES,
            "predictions_dir": str(PRED_DIR),
            "models_dir": str(MODEL_DIR),
        },
        "results": results,
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x),
        encoding="utf-8",
    )
    print(f"\nResults written to {OUTPUT_JSON}")
    print(f"Per-pair predictions written to {PRED_DIR}/")
    print(f"Fitted models written to {MODEL_DIR}/")

    _write_markdown(results)


def _write_markdown(results: dict) -> None:
    ref2 = B4_REFERENCE["Shard2"]

    lines = [
        "# B11 — Learned Classifier (Logistic Regression + MLP)",
        "",
        f"Generated: {date.today()}",
        "Train: Shard 1 | Eval: Shard 2 (held-out) | Threshold: cross-validated on Shard 1",
        "",
        "## Results vs. B4 (Shard 2, cv-tuned threshold)",
        "",
        "| Model | Features | S2 P | S2 R | S2 F1 | S2 PR-AUC |",
        "|---|---|---|---|---|---|",
        f"| **B4** | inlier_ratio + pose_score (hand-tuned) "
        f"| {ref2['precision']:.3f} | {ref2['recall']:.3f} | {ref2['f1']:.3f} | {ref2['pr_auc']:.4f} |",
    ]
    group_labels = {
        "geom_only":  "kp_count + inlier_ratio only",
        "vggt_aggr":  "+ global_sim + pose_score",
        "vggt_comps": "+ 4 pose components",
        "all_feats":  "all 17 features",
    }
    for g, label in group_labels.items():
        r = results["lr"][g]["test_tuned"]
        lines.append(
            f"| LR | {label} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['pr_auc']:.4f} |"
        )
    mlp = results["mlp"]["all_feats"]["test_tuned"]
    lines.append(
        f"| MLP 64→32 | all 17 features "
        f"| {mlp['precision']:.3f} | {mlp['recall']:.3f} | {mlp['f1']:.3f} | {mlp['pr_auc']:.4f} |"
    )

    lines += ["", "## LR feature importances (all_feats, standardised coefficients)", ""]
    lines.append("| Feature | Coefficient | Direction |")
    lines.append("|---|---|---|")
    for name, coef in results["lr"]["all_feats"]["importances"]:
        direction = "positive (predicts match)" if coef > 0 else "negative (predicts non-match)"
        lines.append(f"| {name} | {coef:+.4f} | {direction} |")

    lines += [
        "",
        "## Reproducibility",
        "",
        "Every variant's per-pair Shard1-CV and Shard2-test predictions are persisted to "
        f"`{PRED_DIR.name}/<variant>.jsonl`, and every fitted pipeline is joblib-dumped to "
        f"`{MODEL_DIR.name}/<variant>.joblib`. No downstream script should retrain any B11 "
        "variant — load these artifacts instead. See `ablation/STATISTICS_METHODOLOGY.md`.",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown written to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
