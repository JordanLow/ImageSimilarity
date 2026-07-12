#!/usr/bin/env python3
"""
Transitivity analysis of expert-labeled positive pairs.

"Same exposure" is an equivalence relation: if (a,b) and (b,c) are both
confirmed positives, (a,c) is logically implied. This script:

  1. Builds the graph of labeled positive pairs and reports its connected
     components ("exposure groups") and their size distribution.
  2. Enumerates implied-but-never-labeled pairs (open triangles closed by
     transitivity), checks them against existing labels for contradictions,
     and checks whether the pipeline ever scored them.
  3. Classifies implied pairs as archive-archive vs magazine-magazine and
     extracts cross-issue circulation groups (same negative reproduced in
     magazine issues with different publication dates).
  4. Writes closure_pairs.csv (the implied pairs, for geometric
     re-verification through pipeline stages 2-4) and closure_report.json.

Inputs: match_manifest_shard{1,2}.csv (labels) and the per-shard judge
manifests (JSONL) — same layout as conformal_gate.py expects.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

DEFAULT_BASE = ("/tmp/claude-0/-home-user-ImageSimilarity/"
                "8fd33d55-4d1e-5a82-89e1-8af39f5a13fb/scratchpad/jordan_drop")


def load_labels(label_dir, shards=(1, 2)):
    labels = {}
    for shard in shards:
        pattern = os.path.join(label_dir, "**", f"match_manifest_shard{shard}.csv")
        paths = glob.glob(pattern, recursive=True)
        if not paths:
            raise FileNotFoundError(pattern)
        with open(paths[0]) as f:
            for row in csv.DictReader(f):
                s = os.path.splitext(row["source_image"])[0]
                t = os.path.splitext(row["target_image"])[0]
                labels[frozenset((s, t))] = (shard, row["classification"])
    return labels


def load_scored_pairs(manifest_dir):
    scored = set()
    for path in glob.glob(os.path.join(manifest_dir, "**", "*.jsonl"), recursive=True):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s, t = r.get("source_id"), r.get("target_id")
                if s and t:
                    scored.add(frozenset((s, t)))
    return scored


def parse_issue_date(name):
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default=os.path.join(DEFAULT_BASE, "package"))
    ap.add_argument("--manifest-dir", default=os.path.join(DEFAULT_BASE, "manifests"))
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    labels = load_labels(args.label_dir)
    pos = [tuple(k) for k, (sh, c) in labels.items() if c == "Positive"]

    adj = collections.defaultdict(set)
    for s, t in pos:
        adj[s].add(t)
        adj[t].add(s)

    # connected components (union-find)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, t in pos:
        union(s, t)
    comps = collections.defaultdict(list)
    for n in adj:
        comps[find(n)].append(n)
    size_dist = collections.Counter(len(v) for v in comps.values())

    # implied pairs via open triangles
    implied = set()
    for b in list(adj):
        nb = sorted(adj[b])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                a, c = nb[i], nb[j]
                if c not in adj[a]:
                    implied.add(frozenset((a, c)))
    contradictions = [tuple(p) for p in implied
                      if labels.get(p, (None, None))[1] == "Negative"]

    scored = load_scored_pairs(args.manifest_dir)
    implied_scored = sum(1 for p in implied if p in scored)

    def side(x):
        return "magazine" if "Magazine" in x else "archive"

    kinds = collections.Counter(
        tuple(sorted(side(x) for x in p)) for p in implied)

    # cross-issue circulation groups
    circulation = []
    for nodes in comps.values():
        mags = sorted((n for n in nodes if side(n) == "magazine"),
                      key=lambda m: parse_issue_date(m) or (0, 0))
        dates = {parse_issue_date(m) for m in mags} - {None}
        if len(mags) >= 2 and len(dates) >= 2:
            circulation.append({
                "archive_prints": sorted(n for n in nodes if side(n) == "archive"),
                "magazine_appearances": mags,
            })

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "closure_pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "kind"])
        for p in sorted(implied, key=sorted):
            a, b = sorted(p)
            w.writerow([a, b, "-".join(sorted(side(x) for x in p))])

    report = {
        "n_positive_pairs": len(pos),
        "n_images_in_positive_graph": len(adj),
        "n_components": len(comps),
        "component_size_distribution": dict(sorted(size_dist.items())),
        "n_implied_pairs": len(implied),
        "implied_pair_kinds": {"-".join(k): v for k, v in kinds.items()},
        "n_label_contradictions": len(contradictions),
        "n_implied_pairs_ever_scored": implied_scored,
        "n_cross_issue_circulation_groups": len(circulation),
        "circulation_groups": circulation,
    }
    json_path = os.path.join(args.out_dir, "closure_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    for k, v in report.items():
        if k != "circulation_groups":
            print(f"{k}: {v}")
    print(f"wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
