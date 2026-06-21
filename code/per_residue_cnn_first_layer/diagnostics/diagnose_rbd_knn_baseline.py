#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnostic_utils import (  # noqa: E402
    align,
    l2_normalize,
    load_h5_per_residue,
    load_motif_npz,
    pool_embedding,
    row_l2_normalize,
    standardize_rows,
    summarize_profile,
    weighted_knn_predict,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617")
    p.add_argument("--neighbor-ks", default="1,5,20,50")
    p.add_argument("--std", type=float, default=0.2)
    args = p.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    motif_ids, y_raw, kmers = load_motif_npz(ROOT / args.motif_npz)
    y_norm = row_l2_normalize(y_raw)
    train_map = load_h5_per_residue(ROOT / args.train_per_residue_h5)
    q_map = load_h5_per_residue(ROOT / args.query_rice_per_residue_h5)
    q_map.update(load_h5_per_residue(ROOT / args.query_atptbp3_per_residue_h5))
    query_ids = sorted(q_map)
    ks = [int(x) for x in args.neighbor_ks.split(",") if x]

    summary_rows = []
    neighbor_rows = []
    for mode in ["mean", "max", "mean_max"]:
        train_ids0, train_x0 = pool_embedding(train_map, mode)
        train_ids, train_x, y_train = align(train_ids0, train_x0, motif_ids, y_norm)
        q_ids, q_x = pool_embedding(q_map, mode)
        q_order = [q_ids.index(qid) for qid in query_ids]
        q_x = q_x[np.asarray(q_order)]
        train_x = l2_normalize(train_x)
        q_x = l2_normalize(q_x)
        for k in ks:
            pred, neigh, weights = weighted_knn_predict(q_x, train_x, y_train, k, args.std)
            pred = standardize_rows(pred)
            for qi, qid in enumerate(query_ids):
                row = summarize_profile(qid, pred[qi], kmers, f"{mode}_knn{k}")
                summary_rows.append(row)
                for local_rank, train_idx in enumerate(neigh[qi], start=1):
                    true_profile = y_raw[motif_ids.index(train_ids[train_idx])]
                    top = np.argsort(-true_profile)[:5]
                    neighbor_rows.append(
                        {
                            "query_id": qid,
                            "pooling": mode,
                            "neighbor_k": k,
                            "neighbor_rank": local_rank,
                            "train_protein_id": train_ids[train_idx],
                            "weight": float(weights[qi][local_rank - 1]),
                            "neighbor_top1_kmer": str(kmers[top[0]]),
                            "neighbor_top5_kmers": ",".join(kmers[top].astype(str)),
                        }
                    )

    pd.DataFrame(summary_rows).to_csv(out / "diagnostic2_rbd_knn_query_summary.tsv", sep="\t", index=False)
    pd.DataFrame(neighbor_rows).to_csv(out / "diagnostic2_rbd_knn_neighbor_table.tsv.gz", sep="\t", index=False, compression="gzip")
    (out / "diagnostic2_rbd_knn_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(summary_rows).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
