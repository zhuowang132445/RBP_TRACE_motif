#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnostic_utils import (  # noqa: E402
    FAMILIES,
    align,
    assign_profile_family,
    family_scores_from_profile,
    kmer_family,
    l2_normalize,
    load_h5_per_residue,
    load_motif_npz,
    pool_embedding,
    row_l2_normalize,
    summarize_profile,
)


def load_cnn_jple_profiles(path: Path, kmers: np.ndarray) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    out = {}
    kmer_index = {k: i for i, k in enumerate(kmers.astype(str))}
    for qid, sub in df.groupby("query_id"):
        row = np.zeros(len(kmers), dtype=np.float32)
        for _, r in sub.iterrows():
            row[kmer_index[str(r["kmer"])]] = float(r["score"])
        out[str(qid)] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-jple-score-matrix", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_score_matrix.tsv.gz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617")
    p.add_argument("--top-n", type=int, default=50)
    args = p.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    motif_ids, y_raw, kmers = load_motif_npz(ROOT / args.motif_npz)
    y_norm = row_l2_normalize(y_raw)
    train_map = load_h5_per_residue(ROOT / args.train_per_residue_h5)
    q_map = load_h5_per_residue(ROOT / args.query_rice_per_residue_h5)
    q_map.update(load_h5_per_residue(ROOT / args.query_atptbp3_per_residue_h5))

    train_ids0, train_x0 = pool_embedding(train_map, "mean_max")
    train_ids, train_x, _ = align(train_ids0, train_x0, motif_ids, y_norm)
    q_ids, q_x = pool_embedding(q_map, "mean_max")
    train_x = l2_normalize(train_x)
    q_x = l2_normalize(q_x)
    dist = cdist(q_x, train_x, "cosine")
    cnn_profiles = load_cnn_jple_profiles(ROOT / args.cnn_jple_score_matrix, kmers)

    motif_by_id = {pid: y_raw[i] for i, pid in enumerate(motif_ids)}
    neighbor_rows = []
    composition_rows = []
    support_rows = []
    for qi, qid in enumerate(q_ids):
        idx = np.argsort(dist[qi])[: args.top_n]
        fams = []
        for rank, ti in enumerate(idx, start=1):
            pid = train_ids[ti]
            prof = motif_by_id[pid]
            order = np.argsort(-prof)
            fam = assign_profile_family(prof, kmers, 50)
            fams.append(fam)
            scores = family_scores_from_profile(prof, kmers, 50)
            row = {
                "query_id": qid,
                "neighbor_rank": rank,
                "train_protein_id": pid,
                "cosine_distance": float(dist[qi, ti]),
                "neighbor_assigned_family": fam,
                "neighbor_top1_kmer": str(kmers[order[0]]),
                "neighbor_top5_kmers": ",".join(kmers[order[:5]].astype(str)),
            }
            row.update({f"neighbor_family_score_top50_{k}": v for k, v in scores.items()})
            neighbor_rows.append(row)
        comp = {"query_id": qid, "top_neighbor_n": args.top_n}
        for fam in FAMILIES:
            comp[f"neighbor_fraction_{fam}"] = fams.count(fam) / float(len(fams))
        composition_rows.append(comp)
        if qid in cnn_profiles:
            pred = cnn_profiles[qid]
            pred_summary = summarize_profile(qid, pred, kmers, "current_cnn_jple")
            pred_family = pred_summary["assigned_family_top50"]
            support_rows.append(
                {
                    "query_id": qid,
                    "current_cnn_jple_top1": pred_summary["top1_kmer"],
                    "current_cnn_jple_top5": pred_summary["top5_kmers"],
                    "current_cnn_jple_assigned_family_top50": pred_family,
                    "neighbor_support_fraction_same_family": fams.count(pred_family) / float(len(fams)),
                    "neighbor_majority_family": max(FAMILIES, key=lambda f: fams.count(f)),
                    "neighbor_majority_fraction": max(fams.count(f) for f in FAMILIES) / float(len(fams)),
                }
            )

    pd.DataFrame(neighbor_rows).to_csv(out / "diagnostic3_query_top50_neighbors.tsv", sep="\t", index=False)
    pd.DataFrame(composition_rows).to_csv(out / "diagnostic3_query_neighbor_family_composition.tsv", sep="\t", index=False)
    pd.DataFrame(support_rows).to_csv(out / "diagnostic3_query_neighbor_support.tsv", sep="\t", index=False)
    print(pd.DataFrame(support_rows).to_string(index=False))


if __name__ == "__main__":
    main()
