#!/usr/bin/env python3
"""check_negnet_exposure.py

Applies NEG-Net's leakage-prevention rule to a NEW graph (e.g. a future Shard 2/3+ evaluation
graph): any node already touched by a frozen checkpoint's training graph must be removed
entirely -- not just its edges -- before the new graph is evaluated. See
NCR/negnet_training_exposure/README.md for the full mechanism and rationale (decided over the
alternative of proactively resharding to keep shards disjoint in advance; see the session plan
"NEG-Net leakage prevention: audit-and-record, not audit-and-filter").

Why whole-node removal, not just edge removal: NEG-Net is transductive -- it forward-passes
(message-passes) over every node in the graph in one batch, so a node left in place as a
bystander still leaks into its neighbors' updated representations even if its own edges are
excluded from metrics. Removing the node necessarily removes every edge touching it.

Why no transitive/recursive closure is needed: exposure is exact node-id membership in a fixed,
already-computed registry, checked once per node in a single pass. A node adjacent to (but not
itself) a removed node is never removed on that basis alone -- contamination does not cascade
through graph adjacency.

Orphan pruning (a SEPARATE, non-leakage cleanup step): after removing touched nodes, any
remaining node left with zero edges contributes nothing to evaluation. These are pruned too,
but reported under a distinct label ("orphaned_after_leakage_removal") so they are never
confused with an actual contamination finding.

Usage:
    python check_negnet_exposure.py \
        --graph shard2_graph_labeled.jsonl \
        --exposure NCR/negnet_training_exposure/rawdino_shard1_exposure.json \
        --output shard2_graph_labeled_filtered.jsonl \
        --report shard2_exposure_check_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graph_assembly import read_jsonl, write_jsonl


def load_touched_nodes(exposure_paths: list[str]) -> dict[str, list[str]]:
    """node_id -> list of exposure-registry source labels that flagged it (usually one, but
    a node could in principle be touched by more than one prior training run)."""
    touched: dict[str, list[str]] = {}
    for path in exposure_paths:
        registry = json.loads(Path(path).read_text(encoding="utf-8"))
        label = f"{registry.get('backbone', '?')}/{registry.get('shard', '?')} ({Path(path).name})"
        for node_id in registry.get("nodes", {}):
            touched.setdefault(node_id, []).append(label)
    return touched


def apply_exposure_filter(
    rows: list[dict[str, Any]], touched_nodes: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (filtered_rows, report). Two-phase: (1) remove every node whose id is in
    touched_nodes plus every edge touching one of them; (2) among what's left, prune any node
    with zero remaining edges as a separate hygiene pass."""
    node_rows = [r for r in rows if r.get("type") == "node"]
    edge_rows = [r for r in rows if r.get("type") == "edge"]
    other_rows = [r for r in rows if r.get("type") not in ("node", "edge")]

    removed_leakage = {r["node_id"]: touched_nodes[r["node_id"]]
                        for r in node_rows if r["node_id"] in touched_nodes}

    kept_nodes = [r for r in node_rows if r["node_id"] not in removed_leakage]
    kept_edges = [r for r in edge_rows
                  if r["source_id"] not in removed_leakage and r["target_id"] not in removed_leakage]
    n_edges_dropped_leakage = len(edge_rows) - len(kept_edges)

    # Orphan pruning: nodes with zero remaining edges after leakage removal.
    degree: dict[str, int] = {r["node_id"]: 0 for r in kept_nodes}
    for r in kept_edges:
        degree[r["source_id"]] = degree.get(r["source_id"], 0) + 1
        degree[r["target_id"]] = degree.get(r["target_id"], 0) + 1
    orphaned = [nid for nid, deg in degree.items() if deg == 0]
    final_nodes = [r for r in kept_nodes if r["node_id"] not in set(orphaned)]

    filtered_rows = other_rows + final_nodes + kept_edges

    report = {
        "input_node_count": len(node_rows),
        "input_edge_count": len(edge_rows),
        "nodes_removed_leakage": removed_leakage,
        "n_nodes_removed_leakage": len(removed_leakage),
        "n_edges_dropped_as_result": n_edges_dropped_leakage,
        "orphaned_after_leakage_removal": orphaned,
        "n_orphaned_after_leakage_removal": len(orphaned),
        "note_orphans": (
            "Orphaned nodes are a separate graph-hygiene cleanup, not a leakage finding -- "
            "they simply have zero edges left after their touched neighbors were removed."
        ),
        "final_node_count": len(final_nodes),
        "final_edge_count": len(kept_edges),
    }
    return filtered_rows, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", required=True, help="New graph JSONL to check/filter.")
    parser.add_argument("--exposure", nargs="+", required=True,
                        help="One or more NCR/negnet_training_exposure/*.json registry files.")
    parser.add_argument("--output", required=True, help="Filtered graph JSONL, written here.")
    parser.add_argument("--report", required=True, help="JSON report of what was removed/why.")
    args = parser.parse_args(argv)

    touched_nodes = load_touched_nodes(args.exposure)
    print(f"Loaded {len(touched_nodes)} touched node(s) from {len(args.exposure)} exposure "
          f"registry file(s).")

    rows = read_jsonl(Path(args.graph))
    filtered_rows, report = apply_exposure_filter(rows, touched_nodes)

    write_jsonl(Path(args.output), filtered_rows)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input:  {report['input_node_count']} nodes, {report['input_edge_count']} edges")
    print(f"Removed (leakage): {report['n_nodes_removed_leakage']} node(s), "
          f"{report['n_edges_dropped_as_result']} edge(s) as a result")
    if report["nodes_removed_leakage"]:
        for nid, sources in list(report["nodes_removed_leakage"].items())[:10]:
            print(f"    {nid}  <-  {', '.join(sources)}")
    print(f"Pruned (orphaned after leakage removal, hygiene only): "
          f"{report['n_orphaned_after_leakage_removal']} node(s)")
    print(f"Final:  {report['final_node_count']} nodes, {report['final_edge_count']} edges "
          f"-> {args.output}")
    print(f"Report written -> {args.report}")


if __name__ == "__main__":
    main()
