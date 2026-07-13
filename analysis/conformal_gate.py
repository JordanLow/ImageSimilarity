#!/usr/bin/env python3
"""
Prototype: risk-controlled decision layer for image-match verification.

Pipeline
  0. Join judge manifests (features) with human labels (Positive/Negative),
     verify the join by reproducing the hand-rule confusion matrices.
  1. Calibrated fusion score: logistic regression on [inlier_ratio,
     pose_component_score (imputed), pose_missing indicator], isotonic
     calibration via 5-fold CV (CalibratedClassifierCV). Fit on Shard 1 ONLY.
  2. Split-conformal risk control for a recall (>=98%) guarantee.
  3. Three-way decision (accept / abstain / reject) with two conformal
     thresholds, at 2% and 1% risk targets.
  4. Risk-coverage curve: margin-based deferral at fixed abstention budgets.

Calibration set: Shard 1. Evaluation set: Shard 2 (never touched for fitting).
Guarantees are marginal (in expectation over exchangeable draws), not
conditional; finite-sample correction uses k = floor(alpha*(n+1)).
"""
import csv
import json
import math
import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.environ.get(
    "NCR_BASE",
    "/tmp/claude-0/-home-user-ImageSimilarity/"
    "8fd33d55-4d1e-5a82-89e1-8af39f5a13fb/scratchpad")
MANIF_DIR = os.environ.get(
    "NCR_MANIFEST_DIR", os.path.join(BASE, "jordan_drop/manifests/stage1_manifests"))
LABEL_DIR = os.environ.get(
    "NCR_LABEL_DIR", os.path.join(BASE, "jordan_drop/package/stage1_pkg/data"))
OUT_DIR = os.environ.get("NCR_OUT_DIR", os.path.join(BASE, "prototype_conformal"))

POSE_IMPUTE = 10.0  # "large" pose score for missing pose (worst pose ~ few units)
RNG_SEED = 0


# ---------------------------------------------------------------- data loading
def load_shard(shard):
    """Join manifest features with labels; drop Unsure. Returns X, y arrays."""
    feats = {}
    with open(os.path.join(MANIF_DIR, f"Shard{shard} Judge Manifest.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            feats[(r["source_id"], r["target_id"])] = (
                r.get("aspan_2d_inlier_ratio"), r.get("pose_component_score"))

    inl, pose, y = [], [], []
    with open(os.path.join(LABEL_DIR, f"match_manifest_shard{shard}.csv")) as f:
        for row in csv.DictReader(f):
            if row["classification"] == "Unsure":
                continue
            key = (os.path.splitext(row["source_image"])[0],
                   os.path.splitext(row["target_image"])[0])
            if key not in feats:
                raise KeyError(f"join miss: {key}")
            a, p = feats[key]
            inl.append(a)
            pose.append(p)
            y.append(1 if row["classification"] == "Positive" else 0)
    return np.array(inl, dtype=object), np.array(pose, dtype=object), np.array(y)


def hand_rule_confusion(inl, pose, y):
    """Hand rule: accept iff inlier>=0.65 AND pose<=2.13 (missing pose -> reject)."""
    acc = np.array([(a is not None and a >= 0.65 and p is not None and p <= 2.13)
                    for a, p in zip(inl, pose)])
    tp = int(np.sum(acc & (y == 1)));  fp = int(np.sum(acc & (y == 0)))
    fn = int(np.sum(~acc & (y == 1))); tn = int(np.sum(~acc & (y == 0)))
    return dict(TP=tp, FP=fp, TN=tn, FN=fn)


def featurize(inl, pose):
    """[inlier, pose (imputed), pose_missing indicator]."""
    n = len(inl)
    X = np.zeros((n, 3))
    for i in range(n):
        X[i, 0] = 0.0 if inl[i] is None else float(inl[i])
        if pose[i] is None:
            X[i, 1] = POSE_IMPUTE
            X[i, 2] = 1.0
        else:
            X[i, 1] = float(pose[i])
    return X


# ------------------------------------------------------------------- metrics
def ece(y, p, n_bins=10):
    """Expected calibration error, equal-width bins."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def prec_rec(y, accept):
    tp = int(np.sum(accept & (y == 1))); fp = int(np.sum(accept & (y == 0)))
    fn = int(np.sum(~accept & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return prec, rec, tp, fp, fn


# ---------------------------------------------------- conformal thresholds
def conformal_lower(pos_scores, alpha):
    """t = k-th smallest positive-class score, k = floor(alpha*(n+1)).
    Accepting p >= t gives marginal FNR (miss rate on positives) <= alpha."""
    n = len(pos_scores)
    k = math.floor(alpha * (n + 1))
    if k < 1:            # too few calibration points to certify alpha
        return -np.inf
    return float(np.sort(pos_scores)[k - 1])


def conformal_upper(neg_scores, alpha):
    """t = k-th largest negative-class score, k = floor(alpha*(n+1)).
    Accepting p >= t gives marginal false-accept rate on negatives <= alpha."""
    n = len(neg_scores)
    k = math.floor(alpha * (n + 1))
    if k < 1:
        return np.inf
    return float(np.sort(neg_scores)[::-1][k - 1])


def three_way(p2, y2, t_hi, t_lo):
    """Accept p>=t_hi, reject p<=t_lo, else abstain.

    If the bands cross (t_lo >= t_hi), points inside the overlap satisfy both
    rules; they ABSTAIN (previous behavior silently let accept win)."""
    acc = (p2 >= t_hi) & (p2 > t_lo)
    rej = (p2 <= t_lo) & (p2 < t_hi)
    abst = ~acc & ~rej
    n = len(p2)
    prec_acc = (float(np.sum(acc & (y2 == 1)) / acc.sum()) if acc.sum() else
                float("nan"))
    n_pos = int((y2 == 1).sum())
    fatal = int(np.sum(rej & (y2 == 1)))          # true positive auto-rejected
    recall_with_human = (n_pos - fatal) / n_pos    # abstain -> human = saved
    return dict(
        t_hi=float(t_hi), t_lo=float(t_lo),
        frac_accept=float(acc.mean()), frac_reject=float(rej.mean()),
        frac_abstain=float(abst.mean()),
        n_accept=int(acc.sum()), n_reject=int(rej.sum()), n_abstain=int(abst.sum()),
        precision_auto_accept=prec_acc,
        fatal_errors_TP_auto_rejected=fatal,
        recall_counting_abstain_as_human=float(recall_with_human),
        false_accepts_in_auto_accept=int(np.sum(acc & (y2 == 0))),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    # ---- 0. load + verify join --------------------------------------------
    inl1, pose1, y1 = load_shard(1)
    inl2, pose2, y2 = load_shard(2)
    ref = {1: dict(TP=313, FP=48, TN=268, FN=12),
           2: dict(TP=248, FP=27, TN=359, FN=4)}
    ver = {}
    for s, (i_, p_, y_) in {1: (inl1, pose1, y1), 2: (inl2, pose2, y2)}.items():
        cm = hand_rule_confusion(i_, p_, y_)
        ver[f"shard{s}"] = dict(computed=cm, reference=ref[s],
                                match=cm == ref[s], n=len(y_))
        assert cm == ref[s], f"shard {s} join verification FAILED: {cm}"
    results["join_verification"] = ver

    X1, X2 = featurize(inl1, pose1), featurize(inl2, pose2)

    # ---- 1. calibrated fusion score ----------------------------------------
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=RNG_SEED))
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X1, y1)
    p1 = clf.predict_proba(X1)[:, 1]   # calibration-set scores (Shard 1)
    p2 = clf.predict_proba(X2)[:, 1]   # evaluation scores (Shard 2)
    results["fusion"] = dict(
        shard2_pr_auc=float(average_precision_score(y2, p2)),
        shard2_ece_10bin=ece(y2, p2),
        shard1_ece_10bin=ece(y1, p1),
        n_pose_missing_shard1=int(X1[:, 2].sum()),
        n_pose_missing_shard2=int(X2[:, 2].sum()),
    )

    # ---- 2. conformal recall guarantee (FNR <= 2%) --------------------------
    alpha = 0.02
    pos1 = p1[y1 == 1]
    t = conformal_lower(pos1, alpha)
    accept = p2 >= t
    prec, rec, tp, fp, fn = prec_rec(y2, accept)
    results["conformal_recall_gate"] = dict(
        alpha=alpha, n_calib_pos=int(len(pos1)),
        k=int(math.floor(alpha * (len(pos1) + 1))), threshold=float(t),
        shard2=dict(precision=prec, recall=rec, TP=tp, FP=fp, FN=fn),
        hand_rule_shard2=dict(precision=0.902, recall=0.984),
    )

    # ---- 3. three-way decision with abstention ------------------------------
    neg1 = p1[y1 == 0]
    tw = {}
    for a in (0.02, 0.01):
        t_hi = conformal_upper(neg1, a)   # controls false-accepts
        t_lo = conformal_lower(pos1, a)   # controls false-rejects
        tw[f"alpha_{a}"] = three_way(p2, y2, t_hi, t_lo)
    tw["n_calib_pos"] = int(len(pos1)); tw["n_calib_neg"] = int(len(neg1))
    results["three_way"] = tw

    # ---- 4. risk-coverage curve (margin-based deferral) ---------------------
    margin = np.abs(p2 - 0.5)
    order = np.argsort(margin)            # least confident first
    curve = []
    n = len(p2)
    for budget in np.arange(0.0, 0.301, 0.05):
        n_abst = int(round(budget * n))
        abst_idx = set(order[:n_abst].tolist())
        keep = np.array([i not in abst_idx for i in range(n)])
        yk, pk = y2[keep], p2[keep]
        acc = pk >= 0.5
        prec, rec, tp, fp, fn = prec_rec(yk, acc)
        curve.append(dict(abstain_budget=round(float(budget), 2),
                          coverage=float(keep.mean()),
                          precision_auto=prec, recall_auto=rec,
                          TP=tp, FP=fp, FN=fn))
    results["risk_coverage"] = curve

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
