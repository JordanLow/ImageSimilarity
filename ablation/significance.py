"""Canonical significance-testing module for the paper's ablation statistics.

Single source of truth for McNemar's test and Holm-Bonferroni correction, used by
every ablation family (Stage 2: LightGlue/RoMa; Stage 3: B1/B2/B5/B6/B8/B11/MASt3R).
Replaces three independently-drifting implementations that used to live in
`statistics.py`, `eval_stage2.py`, and the MASt3R notebook — see
`ablation/STATISTICS_METHODOLOGY.md` for the full history and rationale.

McNemar: always exact two-sided binomial — no chi-squared/continuity-correction
approximation branch. Every discordant-pair count observed across this project's
ablations so far (max seen: b=21, c=41) is small enough that exact computation is
trivial, so there is no reason to fall back to an approximation for large samples.

Holm: callers group tests into families explicitly via a `family` label (see
`holm_correct`'s docstring) — correction happens once per family, pooled across
shards within that family. No individual ablation script should compute its own
Holm-adjusted values; that happens once, centrally, in
`ablation/aggregate_significance.py`, after every family member's raw p-value is
known.
"""
from __future__ import annotations

from math import comb
from typing import Any

ALPHA = 0.05


def mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    """Two-sided exact binomial McNemar's test.

    b = challenger correct & reference wrong (challenger outperforms on this pair)
    c = challenger wrong & reference correct (reference outperforms on this pair)

    Returns b, c, n_discordant, p_value, direction, and method (always
    "exact_binomial" — kept as an explicit field, not because there's a second
    method anymore, but so downstream tables/consumers don't need to special-case
    the absence of the field).
    """
    n = b + c
    if n == 0:
        direction = "equal"
    elif b > c:
        direction = "challenger_better"
    elif c > b:
        direction = "reference_better"
    else:
        direction = "equal"

    if n == 0:
        p_value = 1.0
    else:
        lo = min(b, c)
        # Two-sided exact binomial p-value under H0: b, c ~ Binomial(n, 0.5).
        # Summing the lower tail (0..lo) is equivalent by symmetry to summing the
        # upper tail (max(b,c)..n); lo is the smaller sum, cheaper to compute.
        tail = sum(comb(n, k) for k in range(lo + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)

    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": float(p_value),
        "method": "exact_binomial",
        "direction": direction,
    }


def mcnemar_from_predictions(
    y_true: list[int],
    pred_challenger: list[int],
    pred_reference: list[int],
) -> dict[str, Any]:
    """Convenience wrapper: derive b/c from three aligned, equal-length prediction
    lists (correctness-based, not raw-agreement-based — see mcnemar_exact's
    docstring for the b/c definition), then run mcnemar_exact.
    """
    b = sum(
        1 for yt, pc, pr in zip(y_true, pred_challenger, pred_reference)
        if pc == yt and pr != yt
    )
    c = sum(
        1 for yt, pc, pr in zip(y_true, pred_challenger, pred_reference)
        if pc != yt and pr == yt
    )
    return mcnemar_exact(b, c)


def holm_correct(tests: list[dict[str, Any]], family_key: str = "family") -> None:
    """Add holm_p and holm_reject to each test dict, in place, grouped by
    test[family_key].

    Every test dict must already carry a "p_value" (e.g. from mcnemar_exact) and a
    value under `family_key` identifying which family it belongs to (this project's
    convention: "stage2" or "stage3" — see STATISTICS_METHODOLOGY.md). Tests are
    corrected only against other tests in the *same* family; different families
    never affect each other's adjusted p-values.

    This is the one place Holm correction should ever run in this project — normally
    called once from `aggregate_significance.py` after every family member's raw
    p-value is known, never per-shard or per-script in isolation.
    """
    families: dict[Any, list[dict[str, Any]]] = {}
    for t in tests:
        families.setdefault(t[family_key], []).append(t)

    for family_tests in families.values():
        m = len(family_tests)
        order = sorted(range(m), key=lambda i: family_tests[i]["p_value"])
        max_adj = 0.0
        for rank, idx in enumerate(order):
            adj = min(family_tests[idx]["p_value"] * (m - rank), 1.0)
            adj = max(adj, max_adj)
            family_tests[idx]["holm_p"] = round(float(adj), 6)
            family_tests[idx]["holm_reject"] = bool(adj < ALPHA)
            max_adj = adj
