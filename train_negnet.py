"""Tier-0 training loop for NEG-Net on assembled graph JSONL files.

Wires together graph_assembly.py outputs, dump_prehead_features.py caches,
negnet.py, and losses.py. Full-batch training (a shard graph is ~10^2-10^3
edges). The standardizer is fitted on Shard-1 TRAINING folds only and saved
next to the checkpoint; evaluation on the frozen validation graph applies the
prior-shift logit correction first.

Attribution rows for the main table come from --mp-rounds:
    --mp-rounds 0   → pair-MLP (Jordan's original proposal)
    --mp-rounds 3   → + message passing (default)
    --lambda-tri 0 --lambda-nest 0   → ablate the consistency losses

Usage:
    python train_negnet.py \
        --graph shard1_graph_labeled.jsonl --folds shard1_folds.json --val-fold 0 \
        --features output/prehead_cache_shard1.npz \
        --eval-graph shard2_graph_labeled.jsonl \
        --eval-features output/prehead_cache_shard2.npz \
        --save-dir output/negnet/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _load_ablation_utils():
    """Load ablation/ablation_utils.py by path. Deliberately avoids putting
    ablation/ on sys.path — its statistics.py would shadow the stdlib module."""
    path = Path(__file__).parent / "ablation" / "ablation_utils.py"
    spec = importlib.util.spec_from_file_location("ablation_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ablation_utils = _load_ablation_utils()
compute_metrics, pr_auc = _ablation_utils.compute_metrics, _ablation_utils.pr_auc

from graph_assembly import read_jsonl  # noqa: E402
from losses import prior_shift_logits, tier0_loss  # noqa: E402
from negnet import (  # noqa: E402
    NEGNet, Standardizer, count_parameters, extract_edge_features,
)

POS, NEG = "Positive", "Negative"


# ── Data assembly ──────────────────────────────────────────────────────────────

def load_feature_lookup(paths: list[str]) -> dict[str, np.ndarray]:
    """stem → feature vector, merged across one or more .npz caches."""
    lookup: dict[str, np.ndarray] = {}
    for cache_path in paths:
        with np.load(cache_path, allow_pickle=False) as data:
            for path_key, feat_key in (("source_paths", "source_features"),
                                       ("target_paths", "target_features")):
                feats = data[feat_key]
                for i, p in enumerate(data[path_key].tolist()):
                    lookup[Path(str(p)).stem] = feats[i]
    return lookup


def infer_node_dim(feature_lookup: dict[str, np.ndarray]) -> int:
    """Node feature width, taken from whatever cache was actually loaded rather
    than a hardcoded constant -- backbones with no fusion head (e.g. raw DINOv3,
    1024-d) produce a different width than the old 3-backbone pre-head concat
    (3024-d). Fails loudly on a mixed-width cache instead of silently truncating."""
    dims = {v.shape[0] for v in feature_lookup.values()}
    if len(dims) != 1:
        raise ValueError(f"Feature cache has inconsistent vector widths: {sorted(dims)}")
    return dims.pop()


class GraphData:
    def __init__(self, graph_path: str, feature_lookup: dict[str, np.ndarray], node_dim: int):
        rows = read_jsonl(Path(graph_path))
        node_rows = [r for r in rows if r.get("type") == "node"]
        edge_rows = [r for r in rows if r.get("type") == "edge"]

        self.node_ids = [r["node_id"] for r in node_rows]
        index = {nid: i for i, nid in enumerate(self.node_ids)}
        self.node_x = np.zeros((len(self.node_ids), node_dim), dtype=np.float32)
        misses = 0
        for i, r in enumerate(node_rows):
            stem = Path(r["path"]).stem if r.get("path") else r["node_id"]
            vec = feature_lookup.get(stem, feature_lookup.get(r["node_id"]))
            if vec is None:
                misses += 1
            else:
                self.node_x[i] = vec
        if misses:
            print(f"WARNING: {misses}/{len(self.node_ids)} nodes without cached "
                  f"features (zero-imputed) in {graph_path}")

        self.edge_keys = [(r["source_id"], r["target_id"]) for r in edge_rows]
        self.edge_index = np.array(
            [[index[s] for s, _ in self.edge_keys],
             [index[t] for _, t in self.edge_keys]], dtype=np.int64)

        vals, miss = zip(*(extract_edge_features(r.get("evidence")) for r in edge_rows))
        self.edge_values = np.stack(vals)
        self.edge_missing = np.stack(miss)
        self.edge_all_missing = np.array(
            [1.0 if r.get("missing_evidence") else 0.0 for r in edge_rows],
            dtype=np.float32)

        self.label_n = np.array(
            [1 if r.get("label") == POS else 0 for r in edge_rows], dtype=np.int64)
        self.mask_n = np.array(
            [r.get("label") in (POS, NEG) for r in edge_rows], dtype=bool)
        # Scene supervision: pre-audit, Positive ⇒ same scene; Negative is only
        # a scene-negative once the audit separates S\N from U. Until the audit
        # lands, scene BCE trains on positives vs negatives as-is (weak labels).
        self.label_s = self.label_n.copy()
        self.mask_s = self.mask_n.copy()

        self.triangles = self._enumerate_triangles(cap=200_000)
        n_pos = int(self.label_n[self.mask_n].sum())
        print(f"{graph_path}: {len(self.node_ids)} nodes, {len(edge_rows)} edges "
              f"({int(self.mask_n.sum())} labeled, {n_pos} positive), "
              f"{len(self.triangles)} triangles")

    def _enumerate_triangles(self, cap: int) -> np.ndarray:
        edge_idx = {frozenset(k): i for i, k in enumerate(self.edge_keys)}
        neighbors: dict[str, set[str]] = defaultdict(set)
        for s, t in self.edge_keys:
            neighbors[s].add(t)
            neighbors[t].add(s)
        triangles: list[tuple[int, int, int]] = []
        for k, nbrs in neighbors.items():
            nbrs_sorted = sorted(nbrs)
            for ai in range(len(nbrs_sorted)):
                for bi in range(ai + 1, len(nbrs_sorted)):
                    i, j = nbrs_sorted[ai], nbrs_sorted[bi]
                    ij = edge_idx.get(frozenset((i, j)))
                    if ij is None:
                        continue
                    triangles.append((ij,
                                      edge_idx[frozenset((i, k))],
                                      edge_idx[frozenset((k, j))]))
                    if len(triangles) >= cap:
                        return np.asarray(triangles, dtype=np.int64)
        return (np.asarray(triangles, dtype=np.int64)
                if triangles else np.zeros((0, 3), dtype=np.int64))

    def tensors(self, standardizer: Standardizer, device: torch.device) -> dict:
        edge_x = standardizer.transform(
            self.edge_values, self.edge_missing, self.edge_all_missing)
        return {
            "node_x": torch.from_numpy(
                np.ascontiguousarray(self.node_x)).to(device),
            "edge_x": torch.from_numpy(edge_x).to(device),
            "edge_index": torch.from_numpy(self.edge_index).to(device),
            "label_n": torch.from_numpy(self.label_n).to(device),
            "mask_n": torch.from_numpy(self.mask_n).to(device),
            "label_s": torch.from_numpy(self.label_s).to(device),
            "mask_s": torch.from_numpy(self.mask_s).to(device),
            "triangles": torch.from_numpy(self.triangles).to(device),
        }


# ── Train / eval ───────────────────────────────────────────────────────────────

def evaluate(model: NEGNet, batch: dict, train_pos_rate: float,
             eval_pos_rate: float | None, mask: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        _, logit_n = model(batch["node_x"], batch["edge_x"], batch["edge_index"])
        if eval_pos_rate is not None:
            logit_n = prior_shift_logits(logit_n, train_pos_rate, eval_pos_rate)
        probs = torch.sigmoid(logit_n)
    y_true = batch["label_n"][mask].cpu().tolist()
    scores = probs[mask].cpu().tolist()
    y_pred = [1 if s >= 0.5 else 0 for s in scores]
    metrics = compute_metrics(y_true, y_pred)
    metrics["pr_auc"] = pr_auc(y_true, scores)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--eval-graph", default=None)
    parser.add_argument("--eval-features", nargs="+", default=None)
    parser.add_argument("--eval-pos-rate", type=float, default=None,
                        help="Deployment/validation positive rate for the prior "
                             "shift; default: measured from eval-graph labels.")
    parser.add_argument("--mp-rounds", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-tri", type=float, default=1.0)
    parser.add_argument("--lambda-nest", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default="output/negnet")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lookup = load_feature_lookup(args.features)
    node_dim = infer_node_dim(lookup)
    train_graph = GraphData(args.graph, lookup, node_dim)

    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    edge_fold = folds["edge_fold"]
    fold_of = np.array([edge_fold.get(f"{s}||{t}", -1)
                        for s, t in train_graph.edge_keys], dtype=np.int64)
    train_sel = train_graph.mask_n & (fold_of != args.val_fold) & (fold_of >= 0)
    val_sel = train_graph.mask_n & (fold_of == args.val_fold)

    # Standardizer: Shard-1 TRAINING folds only, then frozen.
    standardizer = Standardizer().fit(
        train_graph.edge_values[train_sel], train_graph.edge_missing[train_sel])

    batch = train_graph.tensors(standardizer, device)
    train_mask = torch.from_numpy(train_sel).to(device)
    val_mask = torch.from_numpy(val_sel).to(device)
    batch_train_mask_n = batch["mask_n"] & train_mask
    batch_train_mask_s = batch["mask_s"] & train_mask

    train_pos_rate = float(train_graph.label_n[train_sel].mean())
    n_pos = train_graph.label_n[train_sel].sum()
    pos_weight = float((train_sel.sum() - n_pos) / max(n_pos, 1))

    model = NEGNet(node_dim=node_dim, hidden=args.hidden,
                   mp_rounds=args.mp_rounds).to(device)
    print(f"NEG-Net mp_rounds={args.mp_rounds}: node_dim={node_dim}, "
          f"{count_parameters(model):,} params; "
          f"train edges {int(train_sel.sum())} (pos rate {train_pos_rate:.3f}), "
          f"val edges {int(val_sel.sum())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    best = {"f1": -1.0, "epoch": -1}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logit_s, logit_n = model(batch["node_x"], batch["edge_x"], batch["edge_index"])
        losses = tier0_loss(
            logit_s, logit_n,
            batch["label_n"], batch_train_mask_n,
            batch["label_s"], batch_train_mask_s,
            batch["triangles"],
            lambda_tri=args.lambda_tri, lambda_nest=args.lambda_nest,
            pos_weight=pos_weight)
        losses["total"].backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            val = evaluate(model, batch, train_pos_rate, None, val_mask)
            print(f"epoch {epoch:4d}  loss {losses['total'].item():.4f} "
                  f"(bce_n {losses['bce_n'].item():.4f} tri {losses['triangle'].item():.4f} "
                  f"nest {losses['nesting'].item():.4f})  "
                  f"val P {val['precision']:.3f} R {val['recall']:.3f} F1 {val['f1']:.3f}")
            if val["f1"] > best["f1"]:
                best = {"f1": val["f1"], "epoch": epoch}
                torch.save(model.state_dict(), save_dir / "negnet_best.pt")

    standardizer.save(save_dir / "standardizer.json")
    (save_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "train_pos_rate": train_pos_rate,
                    "best_val": best}, indent=2), encoding="utf-8")
    print(f"best val F1 {best['f1']:.3f} @ epoch {best['epoch']}; "
          f"artifacts in {save_dir}")

    if args.eval_graph:
        eval_lookup = (load_feature_lookup(args.eval_features)
                       if args.eval_features else lookup)
        # Reuses the TRAINING node_dim deliberately (not re-inferred from
        # eval_lookup) -- the frozen model's input width is fixed at train time.
        eval_graph = GraphData(args.eval_graph, eval_lookup, node_dim)
        eval_batch = eval_graph.tensors(standardizer, device)
        eval_pos_rate = args.eval_pos_rate
        if eval_pos_rate is None and eval_graph.mask_n.any():
            eval_pos_rate = float(eval_graph.label_n[eval_graph.mask_n].mean())
        model.load_state_dict(torch.load(save_dir / "negnet_best.pt",
                                         map_location=device))
        metrics = evaluate(model, eval_batch, train_pos_rate, eval_pos_rate,
                           eval_batch["mask_n"])
        print(f"frozen eval ({args.eval_graph}, prior shift "
              f"{train_pos_rate:.3f}→{eval_pos_rate:.3f}): "
              f"P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
              f"F1 {metrics['f1']:.3f} PR-AUC {metrics['pr_auc']:.3f}")


if __name__ == "__main__":
    main()
