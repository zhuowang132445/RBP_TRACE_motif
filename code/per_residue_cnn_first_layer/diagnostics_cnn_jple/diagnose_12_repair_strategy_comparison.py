#!/usr/bin/env python3
"""Compare frozen-CNN latent repair strategies for CNN+JPLE.

Independent diagnostic branches only. Existing checkpoints are not modified.

Routes:
  A: linear / ridge calibration from frozen CNN latent to exact JPLE latent.
  B: small MLP latent alignment calibration.
  C: neighbor-preserving projection head with triplet loss.

All routes decode through the same fixed top-k softmax JPLE neighbor decoder:
latent_dim=100, neighbor_k=10, temperature=0.1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from diagnose_07_cnn_vs_jple_latent_shift import cnn_latent, l2_normalize, load_cnn, mean_pool, resolve, short_id  # noqa: E402
from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, ndcg_at_k, pearson, row_l2_normalize, spearman, standardize_rows, top_overlap, true_top1_rank  # noqa: E402
from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402


FAMILY_ORDER = ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich", "mixed/other"]
QUERY_EXPECTED = {
    "w1": ["U-rich"],
    "w2": ["CUUCU-like"],
    "w3": ["UGUGUG-like"],
    "w4": ["U-rich"],
    "w5": ["GA-rich"],
    "w6": ["U-rich"],
    "AtPTBP3": ["CUUCU-like", "UCUCUC-like", "U-rich"],
}


def log(msg: str) -> None:
    print(f"[diagnose-12] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def seed_like(kmer: str, seeds: list[str]) -> bool:
    for seed in seeds:
        n = len(seed)
        if n > len(kmer):
            continue
        for start in range(len(kmer) - n + 1):
            if hamming(kmer[start : start + n], seed) <= 1:
                return True
    return False


def kmer_family(kmer: str) -> str:
    kmer = str(kmer)
    if kmer.count("U") >= 5 or "UUUUU" in kmer:
        return "U-rich"
    if seed_like(kmer, ["CUUCU", "UCUUC", "CUUCUC", "UCUUCU", "CUUCUU"]):
        return "CUUCU-like"
    if seed_like(kmer, ["UCUCUC", "CUCUCU", "UCUCU", "CUCUC"]):
        return "UCUCUC-like"
    if seed_like(kmer, ["UGUGUG", "GUGUGU", "UGUGU", "GUGUG"]):
        return "UGUGUG-like"
    if (kmer.count("G") + kmer.count("A")) >= 5 or seed_like(kmer, ["GAGGA", "GGAGG", "GAAGA", "AGGAG", "GGAUG"]):
        return "GA-rich"
    return "mixed/other"


def family_indices(kmers: np.ndarray) -> dict[str, np.ndarray]:
    out = {fam: [] for fam in FAMILY_ORDER}
    for i, kmer in enumerate(kmers.astype(str)):
        out[kmer_family(kmer)].append(i)
    return {fam: np.asarray(idx, dtype=np.int64) for fam, idx in out.items()}


def best_rank(profile: np.ndarray, idx: np.ndarray, kmers: np.ndarray) -> tuple[int, str, float]:
    order = np.argsort(-profile)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    best = idx[np.argmin(ranks[idx])]
    return int(ranks[best]), str(kmers[best]), float(profile[best])


def standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def row_cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0] = np.nan
    return float(np.nanmean(np.sum(a * b, axis=1) / denom))


def row_pearson_mean(a: np.ndarray, b: np.ndarray) -> float:
    vals = [pearson(x, y) for x, y in zip(a, b)]
    return float(np.nanmean(vals))


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float32)])
    reg = ridge * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return w[:-1].astype(np.float32), w[-1].astype(np.float32)


def predict_linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (x @ w + b).astype(np.float32)


class CalibrationMlp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim * 2), nn.GELU(), nn.Linear(out_dim * 2, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_alignment_mlp(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, seed: int) -> CalibrationMlp:
    seed_all(seed)
    model = CalibrationMlp(x.shape[1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(xt)
        mse = torch.mean((pred - yt) ** 2)
        cos = 1.0 - torch.nn.functional.cosine_similarity(pred, yt, dim=1).mean()
        loss = mse + cos
        loss.backward()
        opt.step()
    return model


def train_neighbor_projection(
    x: np.ndarray,
    y: np.ndarray,
    exact_top20: list[list[int]],
    exact_top50_sets: list[set[int]],
    epochs: int,
    lr: float,
    seed: int,
) -> CalibrationMlp:
    seed_all(seed)
    model = CalibrationMlp(x.shape[1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    triplet = nn.TripletMarginLoss(margin=0.2, p=2)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        anchors, positives, negatives = [], [], []
        for i in range(n):
            pos = int(rng.choice(exact_top20[i]))
            while True:
                neg = int(rng.integers(0, n))
                if neg != i and neg not in exact_top50_sets[i]:
                    break
            anchors.append(i)
            positives.append(pos)
            negatives.append(neg)
        opt.zero_grad(set_to_none=True)
        pred = torch.nn.functional.normalize(model(xt[anchors]), dim=1)
        pos = torch.nn.functional.normalize(yt[positives], dim=1)
        neg = torch.nn.functional.normalize(yt[negatives], dim=1)
        loss = triplet(pred, pos, neg)
        loss.backward()
        opt.step()
    return model


def decode_topk_softmax(
    z_query: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    temperature: float,
    exclude_self: bool = False,
) -> np.ndarray:
    dist = cdist(z_query, z_train, "cosine")
    if exclude_self and z_query.shape[0] == z_train.shape[0]:
        np.fill_diagonal(dist, np.inf)
    preds = []
    for row in dist:
        idx = np.argsort(row)[:k]
        logits = -row[idx] / temperature
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / weights.sum()
        preds.append(np.sum(weights[:, None] * y_train[idx], axis=0))
    return standardize_rows(np.asarray(preds, dtype=np.float32))


def profile_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    rows = []
    for p, t in zip(pred, true):
        rows.append(
            {
                "pearson": pearson(p, t),
                "spearman": spearman(p, t),
                "top20": top_overlap(p, t, 20),
                "top50": top_overlap(p, t, 50),
                "ndcg20": ndcg_at_k(p, t, 20),
                "ndcg50": ndcg_at_k(p, t, 50),
                "true_top1_rank": true_top1_rank(p, t),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "profile_pearson_mean": float(df["pearson"].mean()),
        "profile_spearman_mean": float(df["spearman"].mean()),
        "top20_overlap_mean": float(df["top20"].mean()),
        "top50_overlap_mean": float(df["top50"].mean()),
        "ndcg20_mean": float(df["ndcg20"].mean()),
        "ndcg50_mean": float(df["ndcg50"].mean()),
        "true_top1_rank_median": float(df["true_top1_rank"].median()),
    }


def neighbor_overlap_metrics(z_pred: np.ndarray, z_exact: np.ndarray) -> dict[str, float]:
    d_pred = cdist(z_pred, z_exact, "cosine")
    d_exact = cdist(z_exact, z_exact, "cosine")
    np.fill_diagonal(d_pred, np.inf)
    np.fill_diagonal(d_exact, np.inf)
    vals20, vals50 = [], []
    for i in range(z_exact.shape[0]):
        a20 = set(np.argsort(d_exact[i])[:20])
        b20 = set(np.argsort(d_pred[i])[:20])
        a50 = set(np.argsort(d_exact[i])[:50])
        b50 = set(np.argsort(d_pred[i])[:50])
        vals20.append(len(a20 & b20) / 20.0)
        vals50.append(len(a50 & b50) / 50.0)
    return {"neighbor_top20_overlap_mean": float(np.mean(vals20)), "neighbor_top50_overlap_mean": float(np.mean(vals50))}


def summarize_query(strategy_id: str, query_ids: list[str], profiles: np.ndarray, kmers: np.ndarray, fam_idx: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for qid, profile in zip(query_ids, profiles):
        sid = short_id(qid)
        order = np.argsort(-profile)
        row: dict[str, Any] = {
            "strategy_id": strategy_id,
            "query": sid,
            "query_id": qid,
            "top1_motif": str(kmers[order[0]]),
            "top5_motifs": ",".join(kmers[order[:5]].astype(str)),
            "top10_motifs": ",".join(kmers[order[:10]].astype(str)),
        }
        for fam in ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich"]:
            rank, kmer, score = best_rank(profile, fam_idx[fam], kmers)
            row[f"{fam}_rank"] = rank
            row[f"{fam}_best_kmer"] = kmer
            row[f"{fam}_best_score"] = score
        expected = QUERY_EXPECTED.get(sid, [])
        row["expected_family_rank"] = min([row[f"{fam}_rank"] for fam in expected]) if expected else np.nan
        row["expected_family_hit_top20"] = bool(expected and row["expected_family_rank"] <= 20)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_strategy(
    strategy_id: str,
    route: str,
    z_train_pred: np.ndarray,
    z_query_pred: np.ndarray,
    z_exact: np.ndarray,
    y_train: np.ndarray,
    query_ids: list[str],
    train_ids: list[str],
    train_family: dict[str, str],
    kmers: np.ndarray,
    fam_idx: dict[str, np.ndarray],
    out_dir: Path,
    decoder_k: int,
    temperature: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pred_profiles = decode_topk_softmax(z_train_pred, z_exact, y_train, decoder_k, temperature, exclude_self=True)
    metrics = profile_metrics(pred_profiles, y_train)
    metrics.update(neighbor_overlap_metrics(z_train_pred, z_exact))
    metrics.update(
        {
            "strategy_id": strategy_id,
            "route": route,
            "latent_cosine_mean": row_cosine_mean(z_train_pred, z_exact),
            "latent_mse": float(np.mean((z_train_pred - z_exact) ** 2)),
            "latent_pearson_mean": row_pearson_mean(z_train_pred, z_exact),
        }
    )
    q_profiles = decode_topk_softmax(z_query_pred, z_exact, y_train, decoder_k, temperature, exclude_self=False)
    q_df = summarize_query(strategy_id, query_ids, q_profiles, kmers, fam_idx)
    q_df.to_csv(out_dir / "query_prediction.tsv", sep="\t", index=False)
    w1_idx = [i for i, qid in enumerate(query_ids) if short_id(qid) == "w1"][0]
    dist = cdist(z_query_pred[[w1_idx]], z_exact, "cosine")[0]
    order = np.argsort(dist)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    candidate_idx = train_ids.index("RNCMPT00434") if "RNCMPT00434" in train_ids else None
    top50 = order[:50]
    metrics["w1_top50_U-rich_fraction"] = float(sum(train_family[train_ids[int(i)]] == "U-rich" for i in top50) / 50.0)
    metrics["w1_top50_mixed_fraction"] = float(sum(train_family[train_ids[int(i)]] == "mixed/other" for i in top50) / 50.0)
    metrics["w1_top50_GA-rich_fraction"] = float(sum(train_family[train_ids[int(i)]] == "GA-rich" for i in top50) / 50.0)
    metrics["w1_top1_neighbor_id"] = train_ids[int(order[0])]
    metrics["w1_top1_neighbor_family"] = train_family[train_ids[int(order[0])]]
    if candidate_idx is not None:
        metrics["w1_RNCMPT00434_rank"] = int(ranks[candidate_idx])
        metrics["w1_RNCMPT00434_distance"] = float(dist[candidate_idx])
    return metrics, q_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_12_repair_strategy_comparison")
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--neighbor-k", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-num-threads", type=int, default=1)
    args = p.parse_args()

    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

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
    train_ids = [pid for pid in anchor_ids if pid in train_map and pid in motif_index]
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    log(f"train_n={len(train_ids)} query_n={len(query_ids)}")

    log("computing exact JPLE100 target latents")
    x_train_mean = l2_normalize(mean_pool(train_ids, train_map))
    jple = RBPTraceFirstLayer(args.latent_dim, threshold=0.01, std=0.2)
    jple.fit(x_train_mean, y_train)
    z_exact = jple.w_train.astype(np.float32)

    log("extracting frozen CNN latents")
    cnn_model = load_cnn(resolve(args.cnn_checkpoint), device)
    z_cnn_train_raw = cnn_latent(cnn_model, train_ids, train_map, device, args.batch_size)
    z_cnn_query_raw = cnn_latent(cnn_model, query_ids, query_map, device, args.batch_size)
    z_cnn_train, x_mean, x_std = standardize_features(z_cnn_train_raw)
    z_cnn_query = apply_standardize(z_cnn_query_raw, x_mean, x_std)

    fam_idx = family_indices(kmers)
    exact_dist = cdist(z_exact, z_exact, "cosine")
    np.fill_diagonal(exact_dist, np.inf)
    exact_top20 = [np.argsort(exact_dist[i])[:20].tolist() for i in range(len(train_ids))]
    exact_top50_sets = [set(np.argsort(exact_dist[i])[:50].tolist()) for i in range(len(train_ids))]

    all_metrics: list[dict[str, Any]] = []
    all_queries: list[pd.DataFrame] = []

    # Route A linear and ridge.
    for strategy_id, ridge in [("routeA_linear_ols", 0.0), ("routeA_ridge_1e-2", 1e-2)]:
        log(f"fitting {strategy_id}")
        route_dir = out / "routeA_linear" / strategy_id
        route_dir.mkdir(parents=True, exist_ok=True)
        w, b = fit_linear(z_cnn_train, z_exact, ridge)
        np.savez_compressed(route_dir / "calibrator.npz", w=w, b=b, x_mean=x_mean, x_std=x_std, ridge=ridge)
        z_train_pred = predict_linear(z_cnn_train, w, b)
        z_query_pred = predict_linear(z_cnn_query, w, b)
        metrics, q_df = evaluate_strategy(strategy_id, "A_linear_calibration", z_train_pred, z_query_pred, z_exact, y_train, query_ids, train_ids, train_family, kmers, fam_idx, route_dir, args.neighbor_k, args.temperature)
        all_metrics.append(metrics)
        all_queries.append(q_df)

    # Route B alignment MLP.
    log("training routeB alignment MLP")
    route_b_dir = out / "routeB_alignment"
    route_b_dir.mkdir(parents=True, exist_ok=True)
    mlp_b = train_alignment_mlp(z_cnn_train, z_exact, args.epochs, args.lr, args.seed)
    torch.save(mlp_b.state_dict(), route_b_dir / "alignment_mlp.pt")
    with torch.no_grad():
        z_train_b = mlp_b(torch.from_numpy(z_cnn_train)).numpy().astype(np.float32)
        z_query_b = mlp_b(torch.from_numpy(z_cnn_query)).numpy().astype(np.float32)
    metrics, q_df = evaluate_strategy("routeB_alignment_mlp", "B_latent_alignment", z_train_b, z_query_b, z_exact, y_train, query_ids, train_ids, train_family, kmers, fam_idx, route_b_dir, args.neighbor_k, args.temperature)
    all_metrics.append(metrics)
    all_queries.append(q_df)

    # Route C neighbor preservation.
    log("training routeC neighbor-preserving projection")
    route_c_dir = out / "routeC_neighbor"
    route_c_dir.mkdir(parents=True, exist_ok=True)
    mlp_c = train_neighbor_projection(z_cnn_train, z_exact, exact_top20, exact_top50_sets, args.epochs, args.lr, args.seed)
    torch.save(mlp_c.state_dict(), route_c_dir / "neighbor_projection.pt")
    with torch.no_grad():
        z_train_c = mlp_c(torch.from_numpy(z_cnn_train)).numpy().astype(np.float32)
        z_query_c = mlp_c(torch.from_numpy(z_cnn_query)).numpy().astype(np.float32)
    metrics, q_df = evaluate_strategy("routeC_neighbor_triplet", "C_neighbor_preservation", z_train_c, z_query_c, z_exact, y_train, query_ids, train_ids, train_family, kmers, fam_idx, route_c_dir, args.neighbor_k, args.temperature)
    all_metrics.append(metrics)
    all_queries.append(q_df)

    metrics_df = pd.DataFrame(all_metrics)
    queries_df = pd.concat(all_queries, ignore_index=True)
    metrics_df.to_csv(out / "repair_strategy_summary.tsv", sep="\t", index=False)
    queries_df.to_csv(out / "query_predictions_all_strategies.tsv", sep="\t", index=False)

    # Add concise recommendation.
    summary_rows = []
    for _, m in metrics_df.iterrows():
        q = queries_df[queries_df["strategy_id"] == m["strategy_id"]].set_index("query")
        stable = bool(
            q.loc["w3", "expected_family_rank"] <= 20
            and q.loc["w4", "expected_family_rank"] <= 20
            and q.loc["w6", "expected_family_rank"] <= 20
            and q.loc["AtPTBP3", "expected_family_rank"] <= 20
        )
        summary_rows.append(
            {
                "strategy_id": m["strategy_id"],
                "route": m["route"],
                "w1_urich_rank": int(q.loc["w1", "U-rich_rank"]),
                "w1_top1": q.loc["w1", "top1_motif"],
                "w2_cuucu_rank": int(q.loc["w2", "CUUCU-like_rank"]),
                "w3_expected_rank": int(q.loc["w3", "expected_family_rank"]),
                "w4_expected_rank": int(q.loc["w4", "expected_family_rank"]),
                "w6_expected_rank": int(q.loc["w6", "expected_family_rank"]),
                "atptbp3_expected_rank": int(q.loc["AtPTBP3", "expected_family_rank"]),
                "keeps_core_queries_stable": stable,
                "w1_top50_U-rich_fraction": float(m["w1_top50_U-rich_fraction"]),
                "w1_top50_mixed_fraction": float(m["w1_top50_mixed_fraction"]),
                "w1_RNCMPT00434_rank": int(m["w1_RNCMPT00434_rank"]),
                "profile_pearson_mean": float(m["profile_pearson_mean"]),
                "top20_overlap_mean": float(m["top20_overlap_mean"]),
                "neighbor_top50_overlap_mean": float(m["neighbor_top50_overlap_mean"]),
            }
        )
    rec_df = pd.DataFrame(summary_rows)
    rec_df.to_csv(out / "repair_strategy_query_stability_summary.tsv", sep="\t", index=False)
    best = rec_df.sort_values(["keeps_core_queries_stable", "w1_urich_rank", "profile_pearson_mean"], ascending=[False, True, False]).iloc[0]
    lines = [
        "# Diagnose 12 Repair Strategy Comparison",
        "",
        "Unified decoder: latent_dim=100, neighbor_k=10, weighting=softmax, temperature=0.1.",
        "",
        "## Strategy Summary",
        "",
        rec_df.to_markdown(index=False),
        "",
        "## RNAcompete Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        f"Best diagnostic strategy by stability then w1 rank: `{best['strategy_id']}`.",
        "This is an independent repair branch, not a replacement for external validation.",
        "If this route improves w1 while keeping w3/w4/w6/AtPTBP3 stable, the next model target should be encoder latent alignment/retrieval geometry rather than decoder tuning.",
        "",
        "## Output Files",
        "",
        "- `repair_strategy_summary.tsv`",
        "- `repair_strategy_query_stability_summary.tsv`",
        "- `query_predictions_all_strategies.tsv`",
        "- route-specific `query_prediction.tsv` files",
    ]
    (out / "repair_strategy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
