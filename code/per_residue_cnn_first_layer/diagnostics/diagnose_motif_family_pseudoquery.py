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

from diagnostic_utils import FAMILIES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617")
    p.add_argument("--diagnostic1-metrics", default="diagnostic1_jple_decoder_loo_metrics.tsv")
    p.add_argument("--diagnostic2-summary", default="diagnostic2_rbd_knn_query_summary.tsv")
    p.add_argument("--diagnostic3-support", default="diagnostic3_query_neighbor_support.tsv")
    p.add_argument("--diagnostic3-composition", default="diagnostic3_query_neighbor_family_composition.tsv")
    args = p.parse_args()

    out = ROOT / args.output_dir
    metrics = pd.read_csv(out / args.diagnostic1_metrics, sep="\t")
    fam_col = "true_family_top50"
    rows = []
    for fam, sub in metrics.groupby(fam_col):
        rows.append(
            {
                "motif_family": fam,
                "n": len(sub),
                "pearson_mean": sub["pearson"].mean(),
                "pearson_median": sub["pearson"].median(),
                "spearman_mean": sub["spearman"].mean(),
                "top20_overlap_mean": sub["top20_overlap"].mean(),
                "top50_overlap_mean": sub["top50_overlap"].mean(),
                "ndcg20_mean": sub["ndcg20"].mean(),
                "ndcg50_mean": sub["ndcg50"].mean(),
                "true_top1_rank_median": sub["true_top1_rank"].median(),
                "recoverable_top20_fraction_ge_0.25": float((sub["top20_overlap"] >= 0.25).mean()),
            }
        )
    fam_summary = pd.DataFrame(rows).sort_values(["pearson_mean", "top20_overlap_mean"], ascending=False)
    fam_summary.to_csv(out / "diagnostic4_motif_family_pseudoquery_summary.tsv", sep="\t", index=False)

    query_support = pd.read_csv(out / args.diagnostic3_support, sep="\t") if (out / args.diagnostic3_support).exists() else pd.DataFrame()
    query_comp = pd.read_csv(out / args.diagnostic3_composition, sep="\t") if (out / args.diagnostic3_composition).exists() else pd.DataFrame()
    knn = pd.read_csv(out / args.diagnostic2_summary, sep="\t") if (out / args.diagnostic2_summary).exists() else pd.DataFrame()

    lines = []
    lines.append("# CNN+JPLE Diagnostic Summary\n")
    lines.append("## Scope\n")
    lines.append("Diagnostics only. No CNN retraining, no split modification, and no query-target motif tuning were used.\n")
    lines.append("The CNN+JPLE model is treated as the chosen motif prediction route. Diagnostics evaluate decoder fidelity, RBD kNN baselines, query neighbor support, and motif-family recoverability.\n")

    lines.append("## Diagnostic 1: JPLE Latent Decoder Fidelity\n")
    lines.append(f"- RNAcompete pseudo-query proteins: {len(metrics)}\n")
    lines.append(f"- Pearson mean/median: {metrics['pearson'].mean():.3f} / {metrics['pearson'].median():.3f}\n")
    lines.append(f"- Spearman mean/median: {metrics['spearman'].mean():.3f} / {metrics['spearman'].median():.3f}\n")
    lines.append(f"- top20/top50 overlap mean: {metrics['top20_overlap'].mean():.3f} / {metrics['top50_overlap'].mean():.3f}\n")
    lines.append(f"- NDCG@20/@50 mean: {metrics['ndcg20'].mean():.3f} / {metrics['ndcg50'].mean():.3f}\n")
    lines.append(f"- true top1 rank median: {metrics['true_top1_rank'].median():.1f}\n")
    lines.append("Interpretation: high correlation with weak top-k overlap means the JPLE decoder can recover broad profile shape better than exact top motif ordering.\n")

    lines.append("## Diagnostic 4: Motif-Family Recoverability\n")
    for _, r in fam_summary.iterrows():
        lines.append(
            f"- {r['motif_family']}: n={int(r['n'])}, Pearson={r['pearson_mean']:.3f}, "
            f"top20={r['top20_overlap_mean']:.3f}, NDCG20={r['ndcg20_mean']:.3f}, "
            f"median true-top1-rank={r['true_top1_rank_median']:.1f}\n"
        )

    if not query_support.empty:
        lines.append("## Diagnostic 3: Query Neighbor Support\n")
        for _, r in query_support.iterrows():
            qid = str(r["query_id"])
            short = "AtPTBP3" if qid.startswith("AtPTBP3") else (qid.split("|original=", 1)[1].split("|", 1)[0] if "|original=" in qid else qid.split("|", 1)[0])
            lines.append(
                f"- {short}: CNN+JPLE top1={r['current_cnn_jple_top1']}, "
                f"assigned_family={r['current_cnn_jple_assigned_family_top50']}, "
                f"same-family neighbor support={float(r['neighbor_support_fraction_same_family']):.2f}, "
                f"neighbor_majority={r['neighbor_majority_family']} ({float(r['neighbor_majority_fraction']):.2f}).\n"
            )

    lines.append("## Query-Level Trust Assessment\n")
    lines.append("- w1: low trust under CNN+JPLE if prediction is CAU-rich; RBD-neighbor support should be checked before displaying as U-rich.\n")
    lines.append("- w2: low trust; unstable across models and expected CUUCU-like family is poorly supported by nearest-neighbor structure.\n")
    lines.append("- w3: moderate/high trust when predicted as UGUGUG-like; this family is directly reflected in query-neighbor support.\n")
    lines.append("- w4: high trust for U-rich if top20/top50 are U-rich dominated.\n")
    lines.append("- w5: low trust; model outputs are unstable and neighbor support is mixed.\n")
    lines.append("- w6: high trust for U-rich if top20/top50 are U-rich dominated.\n")
    lines.append("- AtPTBP3: moderate/high trust when predicted as CUUCU/UCUCUC-like and supported by nearest neighbors; suitable to show as a CU/U-rich tendency rather than overclaiming exact top1.\n")

    lines.append("## Files\n")
    for name in [
        "diagnostic1_jple_decoder_loo_metrics.tsv",
        "diagnostic2_rbd_knn_query_summary.tsv",
        "diagnostic3_query_top50_neighbors.tsv",
        "diagnostic3_query_neighbor_support.tsv",
        "diagnostic4_motif_family_pseudoquery_summary.tsv",
    ]:
        lines.append(f"- {name}\n")

    (out / "final_cnn_jple_diagnostic_summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
