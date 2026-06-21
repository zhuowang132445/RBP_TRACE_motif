#!/usr/bin/env python3
"""Train CNN+JPLE encoder-alignment repair branches and predict query motifs.

This script keeps the JPLE decoder frozen and retrains only the per-residue
CNN encoder against exact JPLE anchor latents. It supports:

1. B-only cosine alignment sweep
2. B + very weak neighbor-preservation sweep

Each config is trained independently on the full RNAcompete set and evaluated
with self-excluded RNAcompete pseudo-query decoding plus W1-W6 / AtPTBP3 query
prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics_cnn_jple"
sys.path.insert(0, str(DIAG_DIR))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize  # noqa: E402
from diagnose_07_cnn_vs_jple_latent_shift import load_cnn, resolve, short_id  # noqa: E402
from diagnose_14_encoder_alignment_benchmark import (  # noqa: E402
    decode_threshold,
    evaluate_strategy,
    family_indices,
    predict_encoder,
    seed_all,
    train_encoder_config,
)


def log(msg: str) -> None:
    print(f"[cnn-jple-encoder-sweep] {msg}", flush=True)


def parse_float_list(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def config_name(cos_lambda: float, neighbor_lambda: float) -> str:
    if neighbor_lambda <= 0:
        return f"B_cos_{cos_lambda:g}"
    return f"Bcos_{cos_lambda:g}_neighbor_{neighbor_lambda:g}"


def write_top100(
    out_path: Path,
    strategy_id: str,
    query_ids: list[str],
    pred_query: np.ndarray,
    kmers: np.ndarray,
) -> None:
    rows = []
    for qid, profile in zip(query_ids, pred_query):
        order = np.argsort(-profile)[:100]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "query": short_id(qid),
                    "query_id": qid,
                    "rank": rank,
                    "kmer": str(kmers[idx]),
                    "score": float(profile[idx]),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/cnn_jple_encoder_alignment_sweep_20260617")
    p.add_argument("--cosine-lambdas", default="0.3,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--neighbor-lambdas", default="0,0.002,0.005,0.01,0.02,0.05")
    p.add_argument("--positive-neighbors", type=int, default=10)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--nce-temperature", type=float, default=0.1)
    p.add_argument("--decoder-threshold", type=float, default=0.01)
    p.add_argument("--decoder-std", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.2)
    p.add_argument("--torch-num-threads", type=int, default=1)
    args = p.parse_args()

    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if args.device == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cos_lambdas = parse_float_list(args.cosine_lambdas)
    neighbor_lambdas = parse_float_list(args.neighbor_lambdas)

    log("loading data")
    motif_ids, y_raw, kmers = load_motif_npz(resolve(args.motif_npz))
    y_all = row_l2_normalize(y_raw)
    motif_index = {pid: i for i, pid in enumerate(motif_ids)}
    train_map, _ = load_h5_features(resolve(args.train_per_residue_h5), None)
    rice_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    at_map, _ = load_h5_features(resolve(args.query_atptbp3_per_residue_h5), None)
    query_map = dict(rice_map)
    query_map.update(at_map)
    query_ids = sorted(query_map.keys(), key=short_id)

    anchor = np.load(resolve(args.jple_anchor_npz), allow_pickle=True)
    anchor_ids = np.asarray(anchor["train_protein_id_list"]).astype(str).tolist()
    anchor_w = np.asarray(anchor["w_train"], dtype=np.float32)
    keep = [i for i, pid in enumerate(anchor_ids) if pid in train_map and pid in motif_index]
    train_ids = [anchor_ids[i] for i in keep]
    z_true = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    train_family_list = [train_family[pid] for pid in train_ids]
    fam_idx = family_indices(kmers)
    log(f"train_n={len(train_ids)} query_n={len(query_ids)}")

    positive_mask = np.zeros((len(train_ids), len(train_ids)), dtype=np.float32)
    exact_dist = cdist(z_true, z_true, "cosine")
    np.fill_diagonal(exact_dist, np.inf)
    for i in range(len(train_ids)):
        positive_mask[i, np.argsort(exact_dist[i])[: args.positive_neighbors]] = 1.0

    log("evaluating frozen baseline")
    baseline_dir = out / "baseline_current_cnn"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_model = load_cnn(resolve(args.cnn_checkpoint), device)
    z_base_train = predict_encoder(baseline_model, train_ids, train_map, device, args.batch_size)
    z_base_query = predict_encoder(baseline_model, query_ids, query_map, device, args.batch_size)
    baseline_metrics, baseline_q = evaluate_strategy(
        "baseline_current_cnn",
        "baseline",
        z_base_train,
        z_base_query,
        z_true,
        y_train,
        train_ids,
        train_family_list,
        train_family,
        query_ids,
        kmers,
        fam_idx,
        baseline_dir,
        args,
    )
    baseline_pred_query = decode_threshold(z_base_query, z_true, y_train, args.decoder_threshold, args.decoder_std, exclude_self=False)
    write_top100(baseline_dir / "query_top100_motifs.tsv", "baseline_current_cnn", query_ids, baseline_pred_query, kmers)

    all_metrics: list[dict[str, Any]] = [baseline_metrics]
    all_queries = [baseline_q]
    grid_rows: list[dict[str, Any]] = []

    configs = []
    for cos_lambda in cos_lambdas:
        configs.append({"cos": cos_lambda, "neighbor": 0.0})
    for cos_lambda in cos_lambdas:
        for neighbor_lambda in neighbor_lambdas:
            if neighbor_lambda <= 0:
                continue
            configs.append({"cos": cos_lambda, "neighbor": neighbor_lambda})

    log(f"training {len(configs)} encoder configs")
    for cfg in configs:
        strategy_id = config_name(cfg["cos"], cfg["neighbor"])
        route = "B_cosine_alignment" if cfg["neighbor"] <= 0 else "B_cosine_plus_weak_neighbor"
        sdir = out / strategy_id
        sdir.mkdir(parents=True, exist_ok=True)
        train_cfg = {"strategy_id": strategy_id, "route": route, "cos": cfg["cos"]}
        if cfg["neighbor"] > 0:
            train_cfg["neighbor"] = cfg["neighbor"]
        log(f"training {strategy_id}")
        model = train_encoder_config(train_cfg, train_ids, train_map, z_true, positive_mask, device, args, sdir)
        z_train = predict_encoder(model, train_ids, train_map, device, args.batch_size)
        z_query = predict_encoder(model, query_ids, query_map, device, args.batch_size)
        metrics, q = evaluate_strategy(
            strategy_id,
            route,
            z_train,
            z_query,
            z_true,
            y_train,
            train_ids,
            train_family_list,
            train_family,
            query_ids,
            kmers,
            fam_idx,
            sdir,
            args,
        )
        pred_query = decode_threshold(z_query, z_true, y_train, args.decoder_threshold, args.decoder_std, exclude_self=False)
        write_top100(sdir / "query_top100_motifs.tsv", strategy_id, query_ids, pred_query, kmers)
        all_metrics.append(metrics)
        all_queries.append(q)
        grid_rows.append(
            {
                "strategy_id": strategy_id,
                "route": route,
                "cosine_lambda": cfg["cos"],
                "neighbor_lambda": cfg["neighbor"],
            }
        )

    metrics_df = pd.DataFrame(all_metrics)
    query_df = pd.concat(all_queries, ignore_index=True)
    grid_df = pd.DataFrame(grid_rows)
    metrics_df.to_csv(out / "strategy_metrics.tsv", sep="\t", index=False)
    query_df.to_csv(out / "query_predictions_all_strategies.tsv", sep="\t", index=False)
    grid_df.to_csv(out / "sweep_grid.tsv", sep="\t", index=False)

    baseline_pearson = float(metrics_df.loc[metrics_df["strategy_id"] == "baseline_current_cnn", "pearson_mean"].iloc[0])
    baseline_ndcg20 = float(metrics_df.loc[metrics_df["strategy_id"] == "baseline_current_cnn", "ndcg20_mean"].iloc[0])
    summary_rows = []
    for _, m in metrics_df.iterrows():
        qsub = query_df[query_df["strategy_id"] == m["strategy_id"]].set_index("query")
        stable = bool(
            qsub.loc["w3", "expected_family_rank"] <= 20
            and qsub.loc["w4", "expected_family_rank"] <= 20
            and qsub.loc["w6", "expected_family_rank"] <= 20
            and qsub.loc["AtPTBP3", "expected_family_rank"] <= 20
        )
        pearson_ok = bool(m["pearson_mean"] >= 0.98 * baseline_pearson)
        ndcg_ok = bool(m["ndcg20_mean"] >= 0.98 * baseline_ndcg20)
        rncmpt_ok = bool(qsub.loc["w1", "RNCMPT00434_rank"] < 50)
        w1_ok = bool(qsub.loc["w1", "U-rich_rank"] <= 20)
        summary_rows.append(
            {
                "strategy_id": m["strategy_id"],
                "route": m["route"],
                "w1_U-rich_rank": int(qsub.loc["w1", "U-rich_rank"]),
                "w1_UUUUUUU_rank": int(qsub.loc["w1", "UUUUUUU_rank"]),
                "w1_top1": qsub.loc["w1", "top1_motif"],
                "w1_top50_U-rich_fraction": float(qsub.loc["w1", "top50_U-rich_fraction"]),
                "w1_RNCMPT00434_rank": int(qsub.loc["w1", "RNCMPT00434_rank"]),
                "core_queries_stable": stable,
                "pearson_mean": float(m["pearson_mean"]),
                "spearman_mean": float(m["spearman_mean"]),
                "top20_overlap_mean": float(m["top20_overlap_mean"]),
                "ndcg20_mean": float(m["ndcg20_mean"]),
                "neighbor_top50_overlap": float(m["neighbor_top50_overlap"]),
                "family_preservation": float(m["family_preservation"]),
                "pearson_ge_98pct_baseline": pearson_ok,
                "ndcg20_ge_98pct_baseline": ndcg_ok,
                "success_w1_top20": w1_ok,
                "success_rncmpt00434_top50": rncmpt_ok,
                "success_primary": bool(w1_ok and stable and pearson_ok and ndcg_ok),
                "success_full": bool(w1_ok and stable and pearson_ok and ndcg_ok and rncmpt_ok),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "encoder_alignment_sweep_summary.tsv", sep="\t", index=False)

    best = summary.sort_values(
        [
            "success_full",
            "success_primary",
            "w1_U-rich_rank",
            "w1_RNCMPT00434_rank",
            "pearson_mean",
            "ndcg20_mean",
        ],
        ascending=[False, False, True, True, False, False],
    ).iloc[0]

    report = [
        "# CNN+JPLE Encoder Alignment Sweep",
        "",
        "Decoder is frozen. Only the CNN encoder is retrained.",
        "",
        f"Baseline Pearson: {baseline_pearson:.6f}",
        f"Baseline NDCG@20: {baseline_ndcg20:.6f}",
        f"Pearson accept threshold (98% baseline): {0.98 * baseline_pearson:.6f}",
        f"NDCG@20 accept threshold (98% baseline): {0.98 * baseline_ndcg20:.6f}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## RNAcompete Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        f"Best config by stated criteria: `{best['strategy_id']}`.",
        "Primary success requires: w1 U-rich rank <= 20, core query stability, Pearson >= 98% baseline, NDCG@20 >= 98% baseline.",
        "Full success additionally requires: RNCMPT00434 rank < 50.",
        "",
        "## Files",
        "",
        "- `sweep_grid.tsv`",
        "- `strategy_metrics.tsv`",
        "- `query_predictions_all_strategies.tsv`",
        "- `encoder_alignment_sweep_summary.tsv`",
        "- per-config `query_prediction.tsv` and `query_top100_motifs.tsv`",
    ]
    (out / "encoder_alignment_sweep_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
