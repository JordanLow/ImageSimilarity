"""Graph assembly for NEG-Net (module J1 in the formulation note, §5.2).

Turns the frozen pipeline manifests into per-shard graph JSONL files that
NEG-Net trains on. Design constraints from the note:

- Evidence assembly is DECOUPLED from labels: `build` writes a graph with no
  labels; `join-labels` adds/overwrites labels as the final step. The negative
  audit then lands as a data update (rerun join-labels), not a rebuild.
- E_aug: same-side edges implied by the transitive closure of Positive edges
  are added with `missing_evidence: true`. Same-side pairs that were actually
  retrieved (magazine crops sit in the query pool by design) arrive as
  ordinary candidate edges with full evidence.
- `audit` checks the family-level shard-disjointness invariant: a connected
  component of Positive edges must live wholly inside one shard. Components
  that straddle shards are reported with a deterministic reassignment proposal
  (majority of labeled edges; ties to the lower-numbered shard).
- `folds` assigns family-masked cross-validation folds on the training shard:
  whole families go to one fold (greedy balance, largest family first).

Graph JSONL row types:
  {"type": "node", "node_id", "side", "path"}
  {"type": "edge", "source_id", "target_id", "origin": "candidate"|"closure",
   "missing_evidence": bool, "evidence": {...}, ["label": str]}

Usage:
    python graph_assembly.py build \
        --judge-manifest "D:/DINO OUTPUTS/Shard1 Judge Manifest.jsonl" \
        --shard Shard1 --output shard1_graph.jsonl
    python graph_assembly.py join-labels \
        --graph shard1_graph.jsonl \
        --labels "D:/DINO OUTPUTS/match_manifest_shard1.csv" \
        --output shard1_graph_labeled.jsonl
    python graph_assembly.py audit \
        --shard Shard1=shard1_graph_labeled.jsonl \
        --shard Shard2=shard2_graph_labeled.jsonl \
        --output shard_audit.json
    python graph_assembly.py folds \
        --graph shard1_graph_labeled.jsonl --k 5 --output shard1_folds.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

# Evidence keys copied verbatim from the vggt_signals.py judged-manifest schema.
# `alignment_homography` is handled separately (3x3 matrix).
EVIDENCE_KEYS = [
    "aspan_2d_raw_match_count",
    "aspan_2d_homography_inliers",
    "aspan_2d_inlier_ratio",
    "aspan_2d_mean_reprojection_error",
    "aspan_2d_median_reprojection_error",
    "aspan_2d_aligned_overlap_fraction",
    "alignment_keypoint_count",
    "global_similarity",
    "pose_rotation_deg",
    "pose_translation_xy_l2",
    "pose_translation_z_abs",
    "pose_fov_l2",
    "pose_zoom_depth_fraction",
    "pose_component_score",
    "pose_component_terms",
    "alignment_homography",
]

POSITIVE = "Positive"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Union-find over node ids ───────────────────────────────────────────────────

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # deterministic root

    def components(self) -> dict[str, list[str]]:
        comps: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            comps[self.find(x)].append(x)
        return {root: sorted(members) for root, members in comps.items()}


# ── build ──────────────────────────────────────────────────────────────────────

def cmd_build(args: argparse.Namespace) -> None:
    manifest = read_jsonl(Path(args.judge_manifest))
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    target_ids = {row["target_id"] for row in manifest}

    for row in manifest:
        sid, tid = row["source_id"], row["target_id"]
        # A node that ever appears on the target side is a magazine crop, even
        # when it also plays the query role (intended same-side retrieval).
        for nid, path_key in ((sid, "source_path"), (tid, "target_path")):
            side = "magazine" if nid in target_ids else "archive"
            nodes.setdefault(nid, {
                "type": "node", "node_id": nid, "side": side,
                "path": row.get(path_key),
            })
        evidence = {k: row.get(k) for k in EVIDENCE_KEYS}
        edges.append({
            "type": "edge", "source_id": sid, "target_id": tid,
            "origin": "candidate", "missing_evidence": False,
            "evidence": evidence,
        })

    rows = [{"type": "meta", "shard": args.shard,
             "judge_manifest": str(args.judge_manifest),
             "n_nodes": len(nodes), "n_candidate_edges": len(edges)}]
    rows += list(nodes.values()) + edges
    write_jsonl(Path(args.output), rows)
    print(f"{args.shard}: {len(nodes)} nodes, {len(edges)} candidate edges → {args.output}")


# ── join-labels (always the LAST step; rerun after every audit) ────────────────

def load_labels(csv_path: Path) -> dict[tuple[str, str], str]:
    """Same join key as ablation_utils.load_ground_truth:
    (source_folder, stem of target_image) == (source_id, target_id)."""
    labels: dict[tuple[str, str], str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["source_folder"], Path(row["target_image"]).stem)
            labels[key] = row["classification"].strip()
    return labels


def cmd_join_labels(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.graph))
    labels: dict[tuple[str, str], str] = {}
    for csv_path in args.labels:
        labels.update(load_labels(Path(csv_path)))

    n_labeled = 0
    out_rows = []
    for row in rows:
        if row.get("type") == "edge" and row.get("origin") == "candidate":
            row = dict(row)
            row.pop("label", None)
            label = labels.get((row["source_id"], row["target_id"]))
            if label is not None:
                row["label"] = label
                n_labeled += 1
        out_rows.append(row)

    # E_aug: same-side closure edges implied by Positive labels, evidence-free.
    # Derived from labels, so they are (re)generated here rather than in build.
    out_rows = [r for r in out_rows if r.get("origin") != "closure"]
    uf = UnionFind()
    scored = {frozenset((r["source_id"], r["target_id"]))
              for r in out_rows if r.get("type") == "edge"}
    for r in out_rows:
        if r.get("type") == "edge" and r.get("label") == POSITIVE:
            uf.union(r["source_id"], r["target_id"])
    n_closure = 0
    for members in uf.components().values():
        for a, b in combinations(members, 2):
            if frozenset((a, b)) not in scored:
                out_rows.append({
                    "type": "edge", "source_id": a, "target_id": b,
                    "origin": "closure", "missing_evidence": True,
                    "evidence": None, "label": POSITIVE,
                })
                n_closure += 1

    write_jsonl(Path(args.output), out_rows)
    print(f"labeled {n_labeled} candidate edges, added {n_closure} closure edges → {args.output}")


# ── audit: family-level shard disjointness ─────────────────────────────────────

def cmd_audit(args: argparse.Namespace) -> None:
    node_shards: dict[str, set[str]] = defaultdict(set)
    labeled_edge_shard: list[tuple[str, str, str]] = []  # (a, b, shard)
    uf = UnionFind()
    role_crossover: dict[str, set[str]] = defaultdict(set)

    for spec in args.shard:
        shard_name, path = spec.split("=", 1)
        for row in read_jsonl(Path(path)):
            if row.get("type") == "node":
                node_shards[row["node_id"]].add(shard_name)
                role_crossover[row["node_id"]].add(row["side"])
            elif row.get("type") == "edge":
                uf.find(row["source_id"])
                uf.find(row["target_id"])
                if row.get("label") == POSITIVE:
                    uf.union(row["source_id"], row["target_id"])
                if row.get("label"):
                    labeled_edge_shard.append((row["source_id"], row["target_id"], shard_name))

    straddlers = []
    for root, members in uf.components().items():
        shards = sorted({s for m in members for s in node_shards.get(m, ())})
        if len(shards) <= 1:
            continue
        votes: dict[str, int] = defaultdict(int)
        member_set = set(members)
        for a, b, shard_name in labeled_edge_shard:
            if a in member_set or b in member_set:
                votes[shard_name] += 1
        # Deterministic rule: majority of labeled edges, ties → lexicographically
        # lowest shard name (== lower-numbered shard for ShardN naming).
        proposal = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if votes else shards[0]
        straddlers.append({
            "component_root": root, "n_members": len(members),
            "members": members, "shards": shards,
            "labeled_edge_votes": dict(votes), "proposed_shard": proposal,
        })

    shared_nodes = sorted(n for n, s in node_shards.items() if len(s) > 1)
    report = {
        "n_nodes": len(node_shards),
        "n_components": len(uf.components()),
        "n_straddling_components": len(straddlers),
        "n_nodes_in_multiple_shards": len(shared_nodes),
        "nodes_in_multiple_shards": shared_nodes,
        "n_nodes_playing_both_roles": sum(1 for s in role_crossover.values() if len(s) > 1),
        "straddling_components": straddlers,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"components straddling shards: {len(straddlers)}; "
          f"nodes present in >1 shard: {len(shared_nodes)} → {args.output}")


# ── folds: family-masked K-fold on the training shard ──────────────────────────

def cmd_folds(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.graph))
    uf = UnionFind()
    labeled_edges = []
    for row in rows:
        if row.get("type") != "edge":
            continue
        uf.find(row["source_id"])
        uf.find(row["target_id"])
        if row.get("label") == POSITIVE:
            uf.union(row["source_id"], row["target_id"])
        if row.get("label"):
            labeled_edges.append(row)

    comps = uf.components()
    fold_load = [0] * args.k
    family_fold: dict[str, int] = {}
    # Largest family first, greedy to the lightest fold; load = labeled edges.
    edge_count: dict[str, int] = defaultdict(int)
    for row in labeled_edges:
        edge_count[uf.find(row["source_id"])] += 1
    for root in sorted(comps, key=lambda r: (-edge_count[r], r)):
        fold = min(range(args.k), key=lambda i: (fold_load[i], i))
        family_fold[root] = fold
        fold_load[fold] += edge_count[root]

    edge_fold = {
        f'{row["source_id"]}||{row["target_id"]}': family_fold[uf.find(row["source_id"])]
        for row in labeled_edges
    }
    out = {"k": args.k, "fold_labeled_edge_counts": fold_load,
           "family_fold": family_fold, "edge_fold": edge_fold}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(comps)} families over {args.k} folds, labeled-edge loads {fold_load} → {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="manifest → unlabeled graph JSONL")
    p.add_argument("--judge-manifest", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("join-labels", help="attach labels + closure edges (final step)")
    p.add_argument("--graph", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_join_labels)

    p = sub.add_parser("audit", help="family-level shard disjointness check")
    p.add_argument("--shard", action="append", required=True,
                   metavar="NAME=graph.jsonl")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("folds", help="family-masked K-fold assignment")
    p.add_argument("--graph", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_folds)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
