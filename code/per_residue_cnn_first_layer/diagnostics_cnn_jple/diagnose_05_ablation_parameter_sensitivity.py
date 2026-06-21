#!/usr/bin/env python3
"""Ablation/parameter sensitivity for CNN+JPLE-style neighbor motif decoding.

This script is diagnostic only. It does not modify existing training scripts,
does not retrain the frozen CNN+JPLE main model, and does not use W1-W6 or
AtPTBP3 expected motifs for parameter selection.

The analysis isolates the JPLE latent neighbor decoder by fitting JPLE latent
spaces from fixed RBD mean+max pooled ESMC embeddings and scanning decoder
parameters. External query results are post-hoc observations only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402
from diagnostic_utils import (  # noqa: E402
    FAMILIES,
    align,
    assign_profile_family,
    family_scores_from_profile,
    kmer_family,
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


QUERY_EXPECTED: dict[str, list[str]] = {
    "w1": ["U-rich"],
    "w2": ["CUUCU-like"],
    "w3": ["UGUGUG-like"],
    "w4": ["U-rich"],
    "w5": ["GA-rich", "mixed/other"],
    "w6": ["U-rich"],
    "AtPTBP3": ["CUUCU-like", "U-rich"],
}


def log(message: str) -> None:
    print(f"[diagnose-05] {message}", flush=True)


def short_id(qid: str) -> str:
    qid = str(qid)
    if qid.startswith("AtPTBP3"):
        return "AtPTBP3"
    if "|original=" in qid:
        return qid.split("|original=", 1)[1].split("|", 1)[0]
    return qid.split("|", 1)[0]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def build_family_indices(kmers: np.ndarray) -> dict[str, np.ndarray]:
    fams: dict[str, list[int]] = {fam: [] for fam in FAMILIES}
    for i, kmer in enumerate(kmers.astype(str)):
        fams[kmer_family(kmer)].append(i)
    return {fam: np.asarray(idx, dtype=np.int64) for fam, idx in fams.items()}


def best_family_rank(profile: np.ndarray, family_idx: np.ndarray) -> tuple[int, str, float]:
    order = np.argsort(-profile)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    best_idx = family_idx[np.argmin(rank[family_idx])]
    return int(rank[best_idx]), str(best_idx), float(profile[best_idx])


def decode_from_latent(
    query_w: np.ndarray,
    train_w: np.ndarray,
    y_train: np.ndarray,
    neighbor_k: int,
    weighting: str,
    param: float,
    exclude_self: bool = False,
) -> np.ndarray:
    dist = cdist(query_w, train_w, "cosine")
    preds: list[np.ndarray] = []
    for qi, row in enumerate(dist):
        order = np.argsort(row)
        if exclude_self:
            order = order[order != qi]
        idx = order[:neighbor_k]
        d = row[idx]
        if weighting == "softmax":
            logits = -d / max(param, 1e-8)
            logits = logits - logits.max()
            weights = np.exp(logits)
        elif weighting == "rbf":
            weights = np.exp(-(d**2) / (max(param, 1e-8) ** 2))
        else:
            raise ValueError(weighting)
        if float(weights.sum()) == 0:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / weights.sum()
        preds.append(np.sum(weights[:, None] * y_train[idx], axis=0))
    return standardize_rows(np.asarray(preds, dtype=np.float32))


def project_query_to_jple_latent(model: RBPTraceFirstLayer, x_query: np.ndarray, x_dim: int) -> np.ndarray:
    x_centered = x_query - model.x_train_mean
    v_train_x = model.v_train[:, :x_dim]
    w_query, _, _, _ = np.linalg.lstsq(v_train_x.T, x_centered.T, rcond=None)
    return w_query.T.astype(np.float32)


def metric_row(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "pearson": pearson(pred, true),
        "spearman": spearman(pred, true),
        "top20_overlap": top_overlap(pred, true, 20),
        "top50_overlap": top_overlap(pred, true, 50),
        "ndcg20": ndcg_at_k(pred, true, 20),
        "ndcg50": ndcg_at_k(pred, true, 50),
        "true_top1_rank": float(true_top1_rank(pred, true)),
    }


def summarize_query_profile(
    query_id: str,
    profile: np.ndarray,
    kmers: np.ndarray,
    family_idx: dict[str, np.ndarray],
) -> dict[str, Any]:
    sid = short_id(query_id)
    order = np.argsort(-profile)
    assigned = assign_profile_family(profile, kmers, 50)
    ranks = {}
    for fam in ["U-rich", "CUUCU-like", "UGUGUG-like", "GA-rich"]:
        idx = family_idx[fam]
        rank_values = np.empty_like(order)
        rank_values[order] = np.arange(1, len(order) + 1)
        best_idx = idx[np.argmin(rank_values[idx])]
        ranks[fam] = {
            "rank": int(rank_values[best_idx]),
            "kmer": str(kmers[best_idx]),
            "score": float(profile[best_idx]),
        }
    expected = QUERY_EXPECTED[sid]
    expected_ranks = []
    for fam in expected:
        if fam == "mixed/other":
            expected_ranks.append(1 if assigned == "mixed/other" else len(profile))
        else:
            expected_ranks.append(ranks[fam]["rank"])
    expected_rank = int(min(expected_ranks))
    return {
        "query_id": query_id,
        "query": sid,
        "top1_kmer": str(kmers[order[0]]),
        "top20_kmers": ",".join(kmers[order[:20]].astype(str)),
        "top50_kmers": ",".join(kmers[order[:50]].astype(str)),
        "predicted_family": assigned,
        "U-rich_best_rank": ranks["U-rich"]["rank"],
        "CUUCU-like_best_rank": ranks["CUUCU-like"]["rank"],
        "UGUGUG-like_best_rank": ranks["UGUGUG-like"]["rank"],
        "GA-rich_best_rank": ranks["GA-rich"]["rank"],
        "expected_family": "/".join(expected),
        "expected_family_best_rank": expected_rank,
        "expected_family_hit_top20": expected_rank <= 20,
        "expected_family_hit_top50": expected_rank <= 50,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    parser.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    parser.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    parser.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    parser.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_05_ablation_parameter_sensitivity")
    parser.add_argument("--latent-dims", default="50,100,200")
    parser.add_argument("--neighbor-ks", default="10,20,50")
    parser.add_argument("--weighting-methods", default="softmax,rbf")
    parser.add_argument("--temperatures", default="0.05,0.1,0.2")
    parser.add_argument("--baseline-pearson", type=float, default=0.816)
    parser.add_argument("--baseline-ndcg20", type=float, default=0.787)
    parser.add_argument("--pearson-threshold", type=float, default=0.775)
    parser.add_argument("--ndcg20-threshold", type=float, default=0.747)
    args = parser.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    motif_ids, y_raw, kmers = load_motif_npz(ROOT / args.motif_npz)
    y_norm = row_l2_normalize(y_raw)
    train_map = load_h5_per_residue(ROOT / args.train_per_residue_h5)
    train_ids0, train_x0 = pool_embedding(train_map, "mean_max")
    train_ids, train_x, y_train_norm = align(train_ids0, train_x0, motif_ids, y_norm)
    y_raw_by_id = {pid: y_raw[i] for i, pid in enumerate(motif_ids)}
    y_train_raw = np.vstack([y_raw_by_id[pid] for pid in train_ids]).astype(np.float32)
    x_train = l2_normalize(train_x)
    x_dim = x_train.shape[1]

    q_map = load_h5_per_residue(ROOT / args.query_rice_per_residue_h5)
    q_map.update(load_h5_per_residue(ROOT / args.query_atptbp3_per_residue_h5))
    q_ids0, q_x0 = pool_embedding(q_map, "mean_max")
    q_order = np.argsort([short_id(x) for x in q_ids0])
    q_ids = [q_ids0[i] for i in q_order]
    q_x = l2_normalize(q_x0[q_order])

    latent_dims = parse_ints(args.latent_dims)
    neighbor_ks = parse_ints(args.neighbor_ks)
    methods = [x for x in args.weighting_methods.split(",") if x]
    temps = parse_floats(args.temperatures)
    family_idx = build_family_indices(kmers)
    true_families = [assign_profile_family(y_train_raw[i], kmers, 50) for i in range(len(train_ids))]

    all_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    top20_rows: list[dict[str, Any]] = []

    model_cache: dict[int, tuple[RBPTraceFirstLayer, np.ndarray, np.ndarray]] = {}
    for latent_dim in latent_dims:
        log(f"fit JPLE latent_dim={latent_dim}")
        model = RBPTraceFirstLayer(num_eigenvector=latent_dim, threshold=0.0, std=0.2)
        model.fit(x_train, y_train_norm)
        q_w = project_query_to_jple_latent(model, q_x, x_dim)
        model_cache[latent_dim] = (model, model.w_train.astype(np.float32), q_w)

    config_id = 0
    for latent_dim, neighbor_k, weighting, temp in itertools.product(latent_dims, neighbor_ks, methods, temps):
        config_id += 1
        config = f"ld{latent_dim}_k{neighbor_k}_{weighting}_p{temp:g}"
        log(f"{config_id}/54 {config}")
        model, train_w, query_w = model_cache[latent_dim]
        pred_train = decode_from_latent(train_w, train_w, model.y_train.astype(np.float32), neighbor_k, weighting, temp, exclude_self=True)
        metric_rows = [metric_row(pred_train[i], y_train_raw[i]) for i in range(len(train_ids))]
        metrics = pd.DataFrame(metric_rows)
        row: dict[str, Any] = {
            "config_id": config,
            "latent_dim": latent_dim,
            "neighbor_k": neighbor_k,
            "weighting": weighting,
            "temperature_or_rbf_std": temp,
            "pooling": "mean+max",
            "pearson_mean": metrics["pearson"].mean(),
            "spearman_mean": metrics["spearman"].mean(),
            "top20_overlap_mean": metrics["top20_overlap"].mean(),
            "top50_overlap_mean": metrics["top50_overlap"].mean(),
            "ndcg20_mean": metrics["ndcg20"].mean(),
            "ndcg50_mean": metrics["ndcg50"].mean(),
            "true_top1_rank_median": metrics["true_top1_rank"].median(),
        }
        for fam in FAMILIES:
            idx = [i for i, f in enumerate(true_families) if f == fam]
            sub = metrics.iloc[idx]
            row[f"family_{fam}_n"] = len(idx)
            row[f"family_{fam}_pearson_mean"] = sub["pearson"].mean() if len(sub) else np.nan
            row[f"family_{fam}_top20_overlap_mean"] = sub["top20_overlap"].mean() if len(sub) else np.nan
            row[f"family_{fam}_ndcg20_mean"] = sub["ndcg20"].mean() if len(sub) else np.nan

        pred_query = decode_from_latent(query_w, train_w, model.y_train.astype(np.float32), neighbor_k, weighting, temp, exclude_self=False)
        hit_count = 0
        hit_count_top50 = 0
        for qi, qid in enumerate(q_ids):
            q_summary = summarize_query_profile(qid, pred_query[qi], kmers, family_idx)
            q_summary.update({"config_id": config, "latent_dim": latent_dim, "neighbor_k": neighbor_k, "weighting": weighting, "temperature_or_rbf_std": temp})
            query_rows.append(q_summary)
            top20_rows.append(
                {
                    "config_id": config,
                    "query": q_summary["query"],
                    "query_id": qid,
                    "top20_kmers": q_summary["top20_kmers"],
                    "predicted_family": q_summary["predicted_family"],
                    "expected_family": q_summary["expected_family"],
                    "expected_family_best_rank": q_summary["expected_family_best_rank"],
                    "expected_family_hit_top20": q_summary["expected_family_hit_top20"],
                    "expected_family_hit_top50": q_summary["expected_family_hit_top50"],
                }
            )
            hit_count += int(q_summary["expected_family_hit_top20"])
            hit_count_top50 += int(q_summary["expected_family_hit_top50"])
            row[f"{q_summary['query']}_expected_family_rank"] = q_summary["expected_family_best_rank"]
            row[f"{q_summary['query']}_expected_family_hit_top20"] = q_summary["expected_family_hit_top20"]
            row[f"{q_summary['query']}_expected_family_hit_top50"] = q_summary["expected_family_hit_top50"]
            row[f"{q_summary['query']}_top1"] = q_summary["top1_kmer"]
            row[f"{q_summary['query']}_predicted_family"] = q_summary["predicted_family"]
        row["query_expected_hit_count_top20"] = hit_count
        row["query_expected_hit_count_top50"] = hit_count_top50
        row["acceptable"] = bool(row["pearson_mean"] >= args.pearson_threshold and row["ndcg20_mean"] >= args.ndcg20_threshold)
        all_rows.append(row)

    all_df = pd.DataFrame(all_rows)
    query_df = pd.DataFrame(query_rows)
    top20_df = pd.DataFrame(top20_rows)
    all_df.to_csv(out / "all_parameter_results.tsv", sep="\t", index=False)
    acceptable = all_df[all_df["acceptable"]].copy()
    acceptable.to_csv(out / "acceptable_parameter_results.tsv", sep="\t", index=False)
    top20_df.to_csv(out / "query_top20_by_parameter.tsv", sep="\t", index=False)

    acceptable_ids = set(acceptable["config_id"].astype(str))
    qacc = query_df[query_df["config_id"].astype(str).isin(acceptable_ids)].copy()
    sensitivity_rows = []
    for query, sub in qacc.groupby("query"):
        ranks = sub["expected_family_best_rank"].astype(float)
        hits = sub["expected_family_hit_top20"].astype(bool)
        sensitivity_rows.append(
            {
                "query": query,
                "acceptable_config_n": len(sub),
                "expected_family_hit_rate_top20": hits.mean() if len(sub) else np.nan,
                "expected_family_hit_rate_top50": sub["expected_family_hit_top50"].astype(bool).mean() if len(sub) else np.nan,
                "median_expected_family_rank": ranks.median() if len(sub) else np.nan,
                "best_expected_family_rank": ranks.min() if len(sub) else np.nan,
                "worst_expected_family_rank": ranks.max() if len(sub) else np.nan,
                "stable": bool(len(sub) > 0 and hits.mean() >= 0.8 and ranks.median() <= 20),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out / "query_sensitivity_matrix.tsv", sep="\t", index=False)
    query_df.to_csv(out / "query_results_by_parameter.tsv", sep="\t", index=False)

    lines = []
    lines.append("# Ablation Parameter Sensitivity Report\n\n")
    lines.append("## Scope\n\n")
    lines.append("This is a diagnostic parameter sensitivity analysis for the JPLE latent neighbor decoder. It does not retrain the frozen CNN+JPLE main model and does not use external query expected motifs for parameter selection.\n\n")
    lines.append(f"Total parameter combinations: {len(all_df)}. Acceptable combinations: {len(acceptable)} using Pearson >= {args.pearson_threshold} and NDCG@20 >= {args.ndcg20_threshold}.\n\n")
    lines.append("## RNAcompete Pseudo-Query Performance\n\n")
    if len(acceptable):
        best = acceptable.sort_values(["pearson_mean", "ndcg20_mean", "top20_overlap_mean"], ascending=False).head(5)
        lines.append("Top acceptable configs by RNAcompete reconstruction:\n\n")
        lines.append(best[["config_id", "pearson_mean", "ndcg20_mean", "top20_overlap_mean", "top50_overlap_mean", "query_expected_hit_count_top20"]].to_markdown(index=False))
        lines.append("\n\n")
    for col in ["latent_dim", "neighbor_k", "weighting", "temperature_or_rbf_std"]:
        group = all_df.groupby(col).agg(
            pearson_mean=("pearson_mean", "mean"),
            ndcg20_mean=("ndcg20_mean", "mean"),
            top20_overlap_mean=("top20_overlap_mean", "mean"),
            acceptable_rate=("acceptable", "mean"),
            query_hit_count_mean=("query_expected_hit_count_top20", "mean"),
        ).reset_index()
        lines.append(f"### By {col}\n\n")
        lines.append(group.to_markdown(index=False))
        lines.append("\n\n")
    lines.append("## Query Sensitivity in Acceptable Parameter Set\n\n")
    lines.append(sensitivity.to_markdown(index=False) if len(sensitivity) else "No acceptable parameter set.\n")
    lines.append("\n\n")
    if len(acceptable):
        max_hit = int(acceptable["query_expected_hit_count_top20"].max())
        lines.append(f"Maximum query expected-family hit count in acceptable configs: {max_hit}/7.\n\n")
        if max_hit >= 5:
            lines.append("At least one acceptable configuration improves external query expected-family hit count to 5/7 or higher. This should be reported only as parameter sensitivity, not as independent external validation.\n")
        else:
            lines.append("No acceptable configuration improved external query expected-family hit count to 5/7 or higher.\n")
    lines.append("\n## Interpretation\n\n")
    lines.append("- Parameters should be judged first by RNAcompete pseudo-query performance.\n")
    lines.append("- External query changes are post-hoc observations only.\n")
    lines.append("- Stable queries are those with high expected-family hit rate across acceptable configurations.\n")
    (out / "ablation_parameter_sensitivity_report.md").write_text("".join(lines))
    log(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
