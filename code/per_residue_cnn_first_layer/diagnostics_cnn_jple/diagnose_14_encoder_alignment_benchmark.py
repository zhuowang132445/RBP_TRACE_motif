#!/usr/bin/env python3
"""Encoder alignment / neighbor preservation benchmark for CNN+JPLE.

Independent diagnostic branches only. The original checkpoint and decoder are
never modified. New encoders/residual heads are trained in output subfolders.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from diagnose_07_cnn_vs_jple_latent_shift import load_cnn, resolve, short_id  # noqa: E402
from cnn_model_utils import PerResidueCnn, collate_batch, load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, ndcg_at_k, pearson, row_l2_normalize, spearman, standardize_rows, top_overlap  # noqa: E402


QUERY_EXPECTED = {
    "w1": ["U-rich"],
    "w2": ["CUUCU-like"],
    "w3": ["UGUGUG-like"],
    "w4": ["U-rich"],
    "w5": ["GA-rich"],
    "w6": ["U-rich"],
    "AtPTBP3": ["CUUCU-like", "UCUCUC-like", "U-rich"],
}
FAMILIES = ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich", "mixed/other"]


def log(msg: str) -> None:
    print(f"[diagnose-14] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    out = {fam: [] for fam in FAMILIES}
    for i, kmer in enumerate(kmers.astype(str)):
        out[kmer_family(kmer)].append(i)
    return {fam: np.asarray(v, dtype=np.int64) for fam, v in out.items()}


def best_rank(profile: np.ndarray, idx: np.ndarray, kmers: np.ndarray) -> tuple[int, str, float]:
    order = np.argsort(-profile)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    best = idx[np.argmin(ranks[idx])]
    return int(ranks[best]), str(kmers[best]), float(profile[best])


def exact_kmer_rank(profile: np.ndarray, kmer: str, kmers: np.ndarray) -> int:
    idx = int(np.where(kmers.astype(str) == kmer)[0][0])
    order = np.argsort(-profile)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return int(ranks[idx])


class LatentDataset(Dataset):
    def __init__(self, ids: list[str], x_map: dict[str, np.ndarray], targets: np.ndarray):
        self.ids = ids
        self.x_map = x_map
        self.targets = targets.astype(np.float32)
        self.id_to_idx = {pid: i for i, pid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        return pid, torch.from_numpy(self.x_map[pid].astype(np.float32)), torch.from_numpy(self.targets[idx])


def predict_encoder(model: nn.Module, ids: list[str], x_map: dict[str, np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    dummy = np.zeros((len(ids), 1), dtype=np.float32)
    loader = DataLoader(LatentDataset(ids, x_map, dummy), batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    rows = []
    model.eval()
    with torch.no_grad():
        for _, x, mask, _ in loader:
            rows.append(model(x.to(device), mask.to(device)).detach().cpu().numpy().astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def decode_threshold(z_query: np.ndarray, z_train: np.ndarray, y_train: np.ndarray, threshold: float, std: float, exclude_self: bool = False) -> np.ndarray:
    dist = cdist(z_query, z_train, "cosine")
    if exclude_self and z_query.shape[0] == z_train.shape[0]:
        np.fill_diagonal(dist, np.inf)
    sim = np.exp(-(dist**2) / (std**2))
    preds = []
    for row in sim:
        idx = np.argwhere(row >= threshold).flatten()
        if len(idx) == 0:
            idx = np.asarray([int(np.argmax(row))])
        idx = idx[np.argsort(-row[idx])]
        w = row[idx]
        w = w / w.sum()
        preds.append(np.sum(w[:, None] * y_train[idx], axis=0))
    return standardize_rows(np.asarray(preds, dtype=np.float32))


def profile_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    vals = []
    for p, t in zip(pred, true):
        vals.append({"pearson": pearson(p, t), "spearman": spearman(p, t), "top20": top_overlap(p, t, 20), "ndcg20": ndcg_at_k(p, t, 20)})
    df = pd.DataFrame(vals)
    return {
        "pearson_mean": float(df["pearson"].mean()),
        "spearman_mean": float(df["spearman"].mean()),
        "top20_overlap_mean": float(df["top20"].mean()),
        "ndcg20_mean": float(df["ndcg20"].mean()),
    }


def neighbor_metrics(z_pred: np.ndarray, z_true: np.ndarray, train_family: list[str]) -> dict[str, float]:
    d_pred = cdist(z_pred, z_true, "cosine")
    d_true = cdist(z_true, z_true, "cosine")
    np.fill_diagonal(d_pred, np.inf)
    np.fill_diagonal(d_true, np.inf)
    top50, recall10, fam_pres = [], [], []
    for i in range(z_true.shape[0]):
        true50 = np.argsort(d_true[i])[:50]
        pred50 = np.argsort(d_pred[i])[:50]
        true10 = set(np.argsort(d_true[i])[:10])
        true_set = set(true50)
        pred_set = set(pred50)
        top50.append(len(true_set & pred_set) / 50.0)
        recall10.append(len(true10 & pred_set) / 10.0)
        true_major = pd.Series([train_family[j] for j in true50]).value_counts().index[0]
        pred_major = pd.Series([train_family[j] for j in pred50]).value_counts().index[0]
        fam_pres.append(true_major == pred_major)
    return {
        "neighbor_top50_overlap": float(np.mean(top50)),
        "neighbor_recall_true_top10_in_pred_top50": float(np.mean(recall10)),
        "family_preservation": float(np.mean(fam_pres)),
    }


def summarize_queries(strategy_id: str, query_ids: list[str], profiles: np.ndarray, kmers: np.ndarray, fam_idx: dict[str, np.ndarray], z_query: np.ndarray, z_train: np.ndarray, train_ids: list[str], train_family: dict[str, str]) -> pd.DataFrame:
    dist = cdist(z_query, z_train, "cosine")
    rows = []
    for qi, (qid, profile) in enumerate(zip(query_ids, profiles)):
        sid = short_id(qid)
        order = np.argsort(-profile)
        row: dict[str, Any] = {
            "strategy_id": strategy_id,
            "query": sid,
            "query_id": qid,
            "top1_motif": str(kmers[order[0]]),
            "top5_motifs": ",".join(kmers[order[:5]].astype(str)),
            "UUUUUUU_rank": exact_kmer_rank(profile, "UUUUUUU", kmers),
        }
        for fam in ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich"]:
            rank, kmer, score = best_rank(profile, fam_idx[fam], kmers)
            row[f"{fam}_rank"] = rank
            row[f"{fam}_best_kmer"] = kmer
            row[f"{fam}_best_score"] = score
        expected = QUERY_EXPECTED.get(sid, [])
        row["expected_family_rank"] = min([row[f"{fam}_rank"] for fam in expected]) if expected else np.nan
        norder = np.argsort(dist[qi])
        top50 = norder[:50]
        row["top50_U-rich_fraction"] = sum(train_family[train_ids[int(i)]] == "U-rich" for i in top50) / 50.0
        if "RNCMPT00434" in train_ids:
            idx = train_ids.index("RNCMPT00434")
            ranks = np.empty_like(norder)
            ranks[norder] = np.arange(1, len(norder) + 1)
            row["RNCMPT00434_rank"] = int(ranks[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def info_nce_loss(pred: torch.Tensor, target_all: torch.Tensor, positive_mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    pred_n = torch.nn.functional.normalize(pred, dim=1)
    target_n = torch.nn.functional.normalize(target_all, dim=1)
    logits = pred_n @ target_n.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits)
    pos = (exp_logits * positive_mask).sum(dim=1).clamp_min(1e-8)
    denom = exp_logits.sum(dim=1).clamp_min(1e-8)
    return -torch.log(pos / denom).mean()


def train_encoder_config(
    config: dict[str, Any],
    train_ids: list[str],
    train_map: dict[str, np.ndarray],
    target: np.ndarray,
    positive_mask_np: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> PerResidueCnn:
    seed_all(args.seed)
    model = PerResidueCnn(1152, args.hidden_dim, target.shape[1], args.kernel_size, args.num_blocks, args.dropout).to(device)
    loader = DataLoader(LatentDataset(train_ids, train_map, target), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    smooth = nn.SmoothL1Loss()
    target_all = torch.from_numpy(target.astype(np.float32)).to(device)
    positive_mask = torch.from_numpy(positive_mask_np.astype(np.float32)).to(device)
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            pred = model(x, mask)
            loss = smooth(pred, y)
            if config.get("mse", 0.0):
                loss = loss + config["mse"] * torch.mean((pred - y) ** 2)
            if config.get("cos", 0.0):
                cos = 1.0 - torch.nn.functional.cosine_similarity(pred, y, dim=1).mean()
                loss = loss + config["cos"] * cos
            if config.get("neighbor", 0.0):
                batch_idx = torch.tensor([train_ids.index(pid) for pid in ids], dtype=torch.long, device=device)
                loss = loss + config["neighbor"] * info_nce_loss(pred, target_all, positive_mask[batch_idx], args.nce_temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % 25 == 0:
            rows.append({"epoch": epoch, "loss": float(np.mean(losses))})
    pd.DataFrame(rows).to_csv(out_dir / "training_log.tsv", sep="\t", index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "args": vars(args)}, out_dir / "encoder.pt")
    return model


class ResidualHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def train_residual_head(z_cnn: np.ndarray, target: np.ndarray, args: argparse.Namespace, out_dir: Path) -> ResidualHead:
    seed_all(args.seed)
    model = ResidualHead(target.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.from_numpy(z_cnn.astype(np.float32))
    y = torch.from_numpy(target.astype(np.float32))
    for _ in range(args.residual_epochs):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2) + 0.2 * (1.0 - torch.nn.functional.cosine_similarity(pred, y, dim=1).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), out_dir / "residual_head.pt")
    return model


def evaluate_strategy(strategy_id: str, route: str, z_train: np.ndarray, z_query: np.ndarray, z_true: np.ndarray, y_train: np.ndarray, train_ids: list[str], train_family_list: list[str], train_family: dict[str, str], query_ids: list[str], kmers: np.ndarray, fam_idx: dict[str, np.ndarray], out_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    pred_train = decode_threshold(z_train, z_true, y_train, args.decoder_threshold, args.decoder_std, exclude_self=True)
    metrics = profile_metrics(pred_train, y_train)
    metrics.update(neighbor_metrics(z_train, z_true, train_family_list))
    metrics.update({"strategy_id": strategy_id, "route": route})
    pred_query = decode_threshold(z_query, z_true, y_train, args.decoder_threshold, args.decoder_std, exclude_self=False)
    q = summarize_queries(strategy_id, query_ids, pred_query, kmers, fam_idx, z_query, z_true, train_ids, train_family)
    q.to_csv(out_dir / "query_prediction.tsv", sep="\t", index=False)
    return metrics, q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_14_encoder_alignment_benchmark")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--residual-epochs", type=int, default=800)
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

    exact_dist = cdist(z_true, z_true, "cosine")
    np.fill_diagonal(exact_dist, np.inf)
    positive_mask = np.zeros((len(train_ids), len(train_ids)), dtype=np.float32)
    for i in range(len(train_ids)):
        positive_mask[i, np.argsort(exact_dist[i])[:10]] = 1.0

    all_metrics: list[dict[str, Any]] = []
    all_queries: list[pd.DataFrame] = []

    # Baseline current CNN checkpoint.
    log("evaluating baseline")
    baseline_model = load_cnn(resolve(args.cnn_checkpoint), device)
    z_base_train = predict_encoder(baseline_model, train_ids, train_map, device, args.batch_size)
    z_base_query = predict_encoder(baseline_model, query_ids, query_map, device, args.batch_size)
    bdir = out / "baseline_current_cnn"
    bdir.mkdir(parents=True, exist_ok=True)
    metrics, q = evaluate_strategy("baseline_current_cnn", "baseline", z_base_train, z_base_query, z_true, y_train, train_ids, train_family_list, train_family, query_ids, kmers, fam_idx, bdir, args)
    all_metrics.append(metrics)
    all_queries.append(q)

    configs: list[dict[str, Any]] = []
    for lam in [0.05, 0.1, 0.2, 0.5]:
        configs.append({"strategy_id": f"A_mse_{lam}", "route": "A_latent_alignment_mse", "mse": lam})
    for lam in [0.05, 0.1, 0.2, 0.5]:
        configs.append({"strategy_id": f"B_cos_{lam}", "route": "B_cosine_alignment", "cos": lam})
    configs.append({"strategy_id": "C_neighbor_infonce_0.1", "route": "C_neighbor_preservation", "neighbor": 0.1})
    configs.append({"strategy_id": "D_mse0.1_neighbor0.1", "route": "D_alignment_plus_neighbor", "mse": 0.1, "neighbor": 0.1})

    for cfg in configs:
        log(f"training {cfg['strategy_id']}")
        sdir = out / cfg["strategy_id"]
        sdir.mkdir(parents=True, exist_ok=True)
        model = train_encoder_config(cfg, train_ids, train_map, z_true, positive_mask, device, args, sdir)
        z_train = predict_encoder(model, train_ids, train_map, device, args.batch_size)
        z_query = predict_encoder(model, query_ids, query_map, device, args.batch_size)
        metrics, q = evaluate_strategy(cfg["strategy_id"], cfg["route"], z_train, z_query, z_true, y_train, train_ids, train_family_list, train_family, query_ids, kmers, fam_idx, sdir, args)
        all_metrics.append(metrics)
        all_queries.append(q)

    # Strategy E residual correction head on current CNN latent.
    log("training E residual head")
    edir = out / "E_residual_head"
    edir.mkdir(parents=True, exist_ok=True)
    head = train_residual_head(z_base_train, z_true, args, edir)
    with torch.no_grad():
        z_train_e = head(torch.from_numpy(z_base_train.astype(np.float32))).numpy().astype(np.float32)
        z_query_e = head(torch.from_numpy(z_base_query.astype(np.float32))).numpy().astype(np.float32)
    metrics, q = evaluate_strategy("E_residual_head", "E_residual_correction", z_train_e, z_query_e, z_true, y_train, train_ids, train_family_list, train_family, query_ids, kmers, fam_idx, edir, args)
    all_metrics.append(metrics)
    all_queries.append(q)

    metrics_df = pd.DataFrame(all_metrics)
    query_df = pd.concat(all_queries, ignore_index=True)
    metrics_df.to_csv(out / "strategy_metrics.tsv", sep="\t", index=False)
    query_df.to_csv(out / "query_predictions_all_strategies.tsv", sep="\t", index=False)

    baseline_pearson = float(metrics_df.loc[metrics_df["strategy_id"] == "baseline_current_cnn", "pearson_mean"].iloc[0])
    summary_rows = []
    for _, m in metrics_df.iterrows():
        qsub = query_df[query_df["strategy_id"] == m["strategy_id"]].set_index("query")
        stable = bool(qsub.loc["w3", "expected_family_rank"] <= 20 and qsub.loc["w4", "expected_family_rank"] <= 20 and qsub.loc["w6", "expected_family_rank"] <= 20 and qsub.loc["AtPTBP3", "expected_family_rank"] <= 20)
        pearson_ok = bool(m["pearson_mean"] >= 0.98 * baseline_pearson)
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
                "pearson_ge_98pct_baseline": pearson_ok,
                "top20_overlap_mean": float(m["top20_overlap_mean"]),
                "ndcg20_mean": float(m["ndcg20_mean"]),
                "neighbor_top50_overlap": float(m["neighbor_top50_overlap"]),
                "family_preservation": float(m["family_preservation"]),
                "success_level1_w1_top20_and_core_stable": bool(w1_ok and stable),
                "success_level2_pearson_ok": pearson_ok,
                "success_level3_RNCMPT00434_top50": rncmpt_ok,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "encoder_alignment_strategy_summary.tsv", sep="\t", index=False)
    best = summary.sort_values(
        ["success_level1_w1_top20_and_core_stable", "success_level2_pearson_ok", "success_level3_RNCMPT00434_top50", "w1_U-rich_rank", "pearson_mean"],
        ascending=[False, False, False, True, False],
    ).iloc[0]
    lines = [
        "# Diagnose 14 Encoder Alignment Strategy Benchmark",
        "",
        f"Baseline Pearson: {baseline_pearson:.6f}. Success threshold: {0.98 * baseline_pearson:.6f}.",
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
        f"Best strategy by stated criteria: `{best['strategy_id']}`.",
        "A strategy is only a formal success if it satisfies w1 U-rich<=20, core query stability, RNAcompete Pearson >=98% baseline, and RNCMPT00434 rank<50.",
        "",
        "## Output Files",
        "",
        "- `encoder_alignment_strategy_summary.tsv`",
        "- `strategy_metrics.tsv`",
        "- `query_predictions_all_strategies.tsv`",
        "- per-strategy `query_prediction.tsv` and model files",
    ]
    (out / "encoder_alignment_strategy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
