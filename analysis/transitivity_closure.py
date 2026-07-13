#!/usr/bin/env python3
"""
Transitivity analysis of expert-labeled positive pairs.

"Same exposure" is an equivalence relation: if (a,b) and (b,c) are both
confirmed positives, (a,c) is logically implied. This script:

  1. Builds the graph of labeled positive pairs and reports its connected
     components ("exposure groups") and their size distribution.
  2. Enumerates ALL implied pairs via full component closure (every
     non-adjacent pair inside a positive component, any hop distance --
     triangle-only enumeration undercounts), checks them against existing
     labels for contradictions, and checks whether the pipeline scored them.
  3. Classifies implied pairs as archive-archive vs magazine-magazine and
     extracts cross-issue circulation groups (same negative reproduced in
     magazine issues with different publication dates).
  4. Quarantines every component that contains a label contradiction
     (the conflict may sit in the Negative label OR in one of the Positive
     edges connecting the endpoints), emits the shortest positive witness
     path for blind re-adjudication, and writes:
       closure_candidates_clean.csv  -- implied pairs from conflict-free
                                        components only (downstream default)
       closure_conflicts.csv         -- contradictions + witness paths
       closure_report.json           -- counts, lineage (input hashes,
                                        command line), quarantine detail

Inputs: match_manifest_shard{1,2}.csv (labels) and the per-shard judge
manifests (JSONL) — same layout as conformal_gate.py expects.
"""
import argparse
import collections
import csv
import glob
import hashlib
import itertools
import json
import os
import re
import sys

DEFAULT_BASE = os.environ.get(
    "NCR_BASE",
    "/tmp/claude-0/-home-user-ImageSimilarity/"
    "8fd33d55-4d1e-5a82-89e1-8af39f5a13fb/scratchpad/jordan_drop")


def load_labels(label_dir, shards=(1, 2)):
    """Exactly-one file per shard; fail fast on cross-file label conflicts."""
    labels = {}
    files = []
    for shard in shards:
        pattern = os.path.join(label_dir, "**", f"match_manifest_shard{shard}.csv")
        paths = sorted(glob.glob(pattern, recursive=True))
        if len(paths) != 1:
            raise SystemExit(f"expected exactly one file for {pattern}, "
                             f"found {len(paths)}: {paths}")
        files.append(paths[0])
        dup_same, conflicts = 0, []
        with open(paths[0]) as f:
            for row in csv.DictReader(f):
                s = os.path.splitext(row["source_image"])[0]
                t = os.path.splitext(row["target_image"])[0]
                key = frozenset((s, t))
                cls = row["classification"]
                if key in labels:
                    if labels[key][1] == cls:
                        dup_same += 1
                    else:
                        conflicts.append((sorted(key), labels[key], (shard, cls)))
                labels[key] = (shard, cls)
        if conflicts:
            raise SystemExit(f"conflicting duplicate labels across inputs: {conflicts}")
        if dup_same:
            print(f"note: {dup_same} same-label duplicate pairs in shard {shard}",
                  file=sys.stderr)
    return labels, files


def load_scored_pairs(manifest_dir, shards):
    """Only manifests whose filename names an allowed shard are read."""
    scored = set()
    allowed = tuple(f"shard{s}" for s in shards)
    for path in glob.glob(os.path.join(manifest_dir, "**", "*.jsonl"), recursive=True):
        if not any(a in os.path.basename(path).lower().replace(" ", "")
                   for a in allowed):
            continue
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shortest_positive_path(adj, a, b):
    """BFS witness path between contradiction endpoints over positive edges."""
    prev = {a: None}
    queue = collections.deque([a])
    while queue:
        x = queue.popleft()
        if x == b:
            path = [b]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return list(reversed(path))
        for y in adj[x]:
            if y not in prev:
                prev[y] = x
                queue.append(y)
    return []


def parse_issue_date(name):
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir",
                    default=os.environ.get("NCR_LABEL_DIR",
                                           os.path.join(DEFAULT_BASE, "package")))
    ap.add_argument("--manifest-dir",
                    default=os.environ.get("NCR_MANIFEST_DIR",
                                           os.path.join(DEFAULT_BASE, "manifests")))
    ap.add_argument("--out-dir", default=os.environ.get("NCR_OUT_DIR", "."))
    ap.add_argument("--shards", default="1,2",
                    help="comma-separated shard labels to load (gate work: --shards 1)")
    ap.add_argument("--allow-shard2", action="store_true",
                    help="explicit opt-in required to include shard 2 (frozen validation)")
    args = ap.parse_args()

    shards = tuple(int(s) for s in args.shards.split(","))
    if 2 in shards and not args.allow_shard2:
        raise SystemExit("shard 2 is the locked validation set; pass --allow-shard2 "
                         "only for descriptive (non-gate) analyses")
    labels, label_files = load_labels(args.label_dir, shards=shards)
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

    # implied pairs: FULL component closure (any hop distance), not just
    # open triangles -- a chain A-B-C-D implies A-D as well.
    implied = set()
    triangle_only = set()
    for b in list(adj):
        nb = sorted(adj[b])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                a, c = nb[i], nb[j]
                if c not in adj[a]:
                    triangle_only.add(frozenset((a, c)))
    for nodes in comps.values():
        for a, c in itertools.combinations(sorted(nodes), 2):
            k = frozenset((a, c))
            if labels.get(k, (None, None))[1] != "Positive":
                implied.add(k)
    contradictions = [tuple(sorted(p)) for p in implied
                      if labels.get(p, (None, None))[1] in ("Negative", "Unsure")]

    scored = load_scored_pairs(args.manifest_dir, shards)
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

    # quarantine: every component containing a contradiction is withheld from
    # the clean candidate list until blind re-adjudication (the error may live
    # in the Negative label OR in a Positive edge on the witness path).
    quarantined_roots = set()
    witness = {}
    for a, b in contradictions:
        quarantined_roots.add(find(a))
        witness[f"{a} | {b}"] = shortest_positive_path(adj, a, b)
    quarantined_nodes = {n for r, nodes in comps.items()
                         for n in nodes if r in quarantined_roots}
    clean = [p for p in implied if not (set(p) & quarantined_nodes)]
    quarantined_pairs = [p for p in implied if set(p) & quarantined_nodes]

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "closure_candidates_clean.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "kind"])
        for p in sorted(clean, key=sorted):
            a, b = sorted(p)
            w.writerow([a, b, "-".join(sorted(side(x) for x in p))])
    conflicts_path = os.path.join(args.out_dir, "closure_conflicts.csv")
    with open(conflicts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "label", "witness_path"])
        for a, b in contradictions:
            w.writerow([a, b, labels[frozenset((a, b))][1],
                        " -> ".join(witness[f"{a} | {b}"])])

    report = {
        "shards": list(shards),
        "command": " ".join(sys.argv),
        "label_file_sha256": {os.path.basename(f): sha256_file(f)
                              for f in label_files},
        "n_positive_pairs": len(pos),
        "n_images_in_positive_graph": len(adj),
        "n_components": len(comps),
        "component_size_distribution": dict(sorted(size_dist.items())),
        "n_implied_pairs_full_closure": len(implied),
        "n_implied_pairs_triangle_only": len(triangle_only),
        "implied_pair_kinds": {"-".join(k): v for k, v in kinds.items()},
        "n_label_contradictions": len(contradictions),
        "label_contradictions": contradictions,
        "n_implied_pairs_ever_scored": implied_scored,
        "n_quarantined_components": len(quarantined_roots),
        "n_quarantined_pairs": len(quarantined_pairs),
        "n_clean_candidates": len(clean),
        "contradiction_witness_paths": witness,
        "n_cross_issue_circulation_groups": len(circulation),
        "circulation_groups": circulation,
    }
    json_path = os.path.join(args.out_dir, "closure_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    for k, v in report.items():
        if k != "circulation_groups":
            print(f"{k}: {v}")
    print(f"wrote {csv_path}, {conflicts_path} and {json_path}")


if __name__ == "__main__":
    main()
