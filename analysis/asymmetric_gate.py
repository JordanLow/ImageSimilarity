#!/usr/bin/env python3
"""
Cost-derived asymmetric risk budgets for the three-way decision gate.

Extends conformal_gate.py: instead of a single symmetric risk target, the
accept and reject thresholds get separate conformal budgets

    alpha_reject  — bound on the rate of true positives auto-rejected
                    (the historian's catastrophic error; keep tiny)
    alpha_accept  — bound on the rate of negatives auto-accepted
                    (costs review time downstream; may be looser)

In Bayes decision-theoretic terms (Chow's rule with costs c_defer, c_FP,
c_FN): accept iff p >= 1 - c_defer/c_FP, reject iff p <= c_defer/c_FN,
defer otherwise. The conformal quantiles replace trust in the calibrated
posterior with realized-risk certificates. Certifiable floor:
alpha >= 1/(n_calibration_class + 1).

Calibration: Shard 1 only. Evaluation: Shard 2. All CPU, seeds fixed.
"""
import json
import os
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_gate import (RNG_SEED, conformal_lower, conformal_upper,
                            featurize, load_shard, three_way)

# (alpha_reject, alpha_accept, label)
BUDGETS = [
    (0.02, 0.02, "symmetric 2%/2% (baseline prototype)"),
    (0.01, 0.05, "reject-risk 1%, accept-risk 5%"),
    (0.005, 0.05, "reject-risk 0.5%, accept-risk 5%"),
    (0.01, 0.10, "reject-risk 1%, accept-risk 10%"),
]


def main():
    inl1, pose1, y1 = load_shard(1)
    inl2, pose2, y2 = load_shard(2)
    X1, X2 = featurize(inl1, pose1), featurize(inl2, pose2)

    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=RNG_SEED))
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X1, y1)
    p1 = clf.predict_proba(X1)[:, 1]
    p2 = clf.predict_proba(X2)[:, 1]
    pos1, neg1 = p1[y1 == 1], p1[y1 == 0]

    floor_rej = 1.0 / (len(pos1) + 1)
    floor_acc = 1.0 / (len(neg1) + 1)
    print(f"calibration: {len(pos1)} pos / {len(neg1)} neg | "
          f"certifiable floors: alpha_reject >= {floor_rej:.4f}, "
          f"alpha_accept >= {floor_acc:.4f}")

    results = {}
    for a_rej, a_acc, label in BUDGETS:
        t_lo = conformal_lower(pos1, a_rej)
        t_hi = conformal_upper(neg1, a_acc)
        r = three_way(p2, y2, t_hi, t_lo)
        r["alpha_reject"], r["alpha_accept"] = a_rej, a_acc
        results[label] = r
        print(f"\n== {label}")
        print(f"   t_lo={r['t_lo']:.4f}  t_hi={r['t_hi']:.4f}")
        print(f"   auto-accept {r['n_accept']} ({r['frac_accept']*100:.1f}%) | "
              f"auto-reject {r['n_reject']} ({r['frac_reject']*100:.1f}%) | "
              f"defer {r['n_abstain']} ({r['frac_abstain']*100:.1f}%)")
        print(f"   accept precision {r['precision_auto_accept']:.4f} | "
              f"false accepts {r['false_accepts_in_auto_accept']} | "
              f"fatal (TP auto-rejected) {r['fatal_errors_TP_auto_rejected']} | "
              f"recall w/ deferral {r['recall_counting_abstain_as_human']:.4f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "asymmetric_gate_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
