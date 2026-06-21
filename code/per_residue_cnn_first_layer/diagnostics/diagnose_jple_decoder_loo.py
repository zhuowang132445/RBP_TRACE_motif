#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402
from diagnostic_utils import (  # noqa: E402
    align,
    family_scores_from_profile,
    l2_normalize,
    load_h5_per_residue,
    load_motif_npz,
    ndcg_at_k,
    pearson,
    pool_embedding,
    row_l2_normalize,
    spearman,
    standardize_rows,
    top_overlap,
    true_top1_rank,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617")
    p.add_argument("--num-eigenvector", type=int, default=122)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--std", type=float, default=0.2)
    args = p.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    motif_ids, y_raw, kmers = load_motif_npz(ROOT / args.motif_npz)
    y_norm = row_l2_normalize(y_raw)
    x_map = load_h5_per_residue(ROOT / args.per_residue_h5)
    ids, x = pool_embedding(x_map, "mean")
    train_ids, x_train, y_train_norm = align(ids, x, motif_ids, y_norm)
    y_raw_by_id = {pid: y_raw[i] for i, pid in enumerate(motif_ids)}

    model = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    model.fit(l2_normalize(x_train), y_train_norm)
    w = model.w_train.astype(np.float32)
    y = model.y_train.astype(np.float32)

    rows = []
    pred_rows = []
    for i, pid in enumerate(train_ids):
        mask = np.ones(len(train_ids), dtype=bool)
        mask[i] = False
        tmp = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
        tmp.load(y_train=y[mask], x_train_mean=model.x_train_mean, y_train_mean=model.y_train_mean, w_train=w[mask], v_train=model.v_train)
        pred, dist, neigh = tmp.predict_protein(np.zeros((1, x_train.shape[1]), dtype=np.float32))
        # Override the query projection by directly decoding this protein's all-train latent.
        # This isolates the neighbor-weighted decoder fidelity from X->latent projection error.
        from scipy.spatial.distance import cdist

        d = cdist(w[i : i + 1], w[mask], "cosine")[0]
        sim = np.exp(-(d**2) / (args.std**2))
        idx = np.argwhere(sim >= args.threshold).flatten()
        if len(idx) == 0:
            idx = np.asarray([int(np.argmax(sim))])
        idx = idx[np.argsort(-sim[idx])]
        weights = sim[idx] / sim[idx].sum()
        pred_profile = np.sum(weights[:, None] * y[mask][idx], axis=0)
        pred_profile = standardize_rows(pred_profile[None, :])[0]
        true = y_raw_by_id[pid]
        fam_scores = family_scores_from_profile(true, kmers, 50)
        row = {
            "protein_id": pid,
            "pearson": pearson(pred_profile, true),
            "spearman": spearman(pred_profile, true),
            "top20_overlap": top_overlap(pred_profile, true, 20),
            "top50_overlap": top_overlap(pred_profile, true, 50),
            "ndcg20": ndcg_at_k(pred_profile, true, 20),
            "ndcg50": ndcg_at_k(pred_profile, true, 50),
            "true_top1_rank": true_top1_rank(pred_profile, true),
            "nearest_latent_distance": float(d.min()),
            "neighbor_count": int(len(idx)),
            "true_family_top50": max(fam_scores, key=fam_scores.get),
        }
        row.update({f"true_family_score_top50_{k}": v for k, v in fam_scores.items()})
        rows.append(row)
        order = np.argsort(-pred_profile)[:50]
        for rank, kidx in enumerate(order, start=1):
            pred_rows.append({"protein_id": pid, "rank": rank, "kmer": str(kmers[kidx]), "pred_score": float(pred_profile[kidx]), "true_score": float(true[kidx])})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "diagnostic1_jple_decoder_loo_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "diagnostic1_jple_decoder_loo_top50.tsv.gz", sep="\t", index=False, compression="gzip")
    summary = {
        "n": int(len(metrics)),
        "pearson_mean": float(metrics["pearson"].mean()),
        "pearson_median": float(metrics["pearson"].median()),
        "spearman_mean": float(metrics["spearman"].mean()),
        "top20_overlap_mean": float(metrics["top20_overlap"].mean()),
        "top50_overlap_mean": float(metrics["top50_overlap"].mean()),
        "ndcg20_mean": float(metrics["ndcg20"].mean()),
        "ndcg50_mean": float(metrics["ndcg50"].mean()),
        "true_top1_rank_median": float(metrics["true_top1_rank"].median()),
    }
    (out / "diagnostic1_jple_decoder_loo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
