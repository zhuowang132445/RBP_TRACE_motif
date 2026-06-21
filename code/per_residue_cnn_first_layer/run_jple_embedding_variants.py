#!/usr/bin/env python3
"""Run JPLE-style all-train motif prediction with pooled and per-residue CNN inputs."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(name, "1")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))

from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402
from cnn_model_utils import PerResidueCnn, collate_batch, load_h5_features, setup_threads  # noqa: E402


ID_KEYS = ["protein_ids", "profile_ids", "ids", "names"]
EMB_KEYS = ["embeddings", "X", "embedding"]
MOTIF_SCORE_KEYS = ["zscores", "scores", "Y", "profiles"]


def log(msg: str) -> None:
    print(f"[jple-embedding-variants] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    q = ROOT / p
    if q.exists():
        return q
    raise FileNotFoundError(path)


def first_key(npz: np.lib.npyio.NpzFile, keys: list[str]) -> str:
    for key in keys:
        if key in npz.files:
            return key
    raise KeyError(f"None of {keys} found in {npz.files}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def row_l2_normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    denom = np.sqrt(np.sum(y * y, axis=1, keepdims=True))
    denom[denom == 0] = 1.0
    return y / denom


def load_npz_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    z = np.load(path, allow_pickle=True)
    ids = np.asarray(z[first_key(z, ID_KEYS)]).astype(str).tolist()
    x = np.asarray(z[first_key(z, EMB_KEYS)], dtype=np.float32)
    return ids, x


def load_motif(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    ids = np.asarray(z[first_key(z, ID_KEYS)]).astype(str).tolist()
    y = np.asarray(z[first_key(z, MOTIF_SCORE_KEYS)], dtype=np.float32)
    kmers = np.asarray(z["kmers"]).astype(str)
    return ids, y, kmers


def load_h5_mean_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    rows: list[np.ndarray] = []
    with h5py.File(path, "r") as h5:
        keys = list(h5["embeddings"].keys())
        for key in keys:
            arr = np.asarray(h5["embeddings"][key], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            ids.append(key.replace("__slash__", "/"))
            rows.append(arr.mean(axis=0))
    return ids, np.vstack(rows).astype(np.float32)


def align_train(
    x_ids: list[str], x: np.ndarray, motif_ids: list[str], y: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    y_index = {pid: i for i, pid in enumerate(motif_ids)}
    keep_x: list[int] = []
    keep_y: list[int] = []
    keep_ids: list[str] = []
    for i, pid in enumerate(x_ids):
        if pid in y_index and np.isfinite(x[i]).all() and np.isfinite(y[y_index[pid]]).all():
            keep_x.append(i)
            keep_y.append(y_index[pid])
            keep_ids.append(pid)
    if not keep_ids:
        raise RuntimeError("No overlapping finite training proteins")
    return keep_ids, x[np.asarray(keep_x)], y[np.asarray(keep_y)]


def top_indices(row: np.ndarray, n: int) -> np.ndarray:
    n = min(int(n), row.shape[0])
    idx = np.argpartition(-row, n - 1)[:n]
    return idx[np.argsort(-row[idx])]


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def seed_like(kmer: str, seeds: list[str]) -> bool:
    for seed in seeds:
        k = len(seed)
        if k > len(kmer):
            continue
        for start in range(0, len(kmer) - k + 1):
            if hamming(kmer[start : start + k], seed) <= 1:
                return True
    return False


def motif_sets(kmers: np.ndarray) -> dict[str, np.ndarray]:
    cuucu = ["CUUCU", "UCUUC", "CUUCUC", "UCUUCU", "CUUCUU"]
    ucucuc = ["UCUCUC", "CUCUCU", "UCUCU", "CUCUC"]
    sets: dict[str, list[int]] = {"CUUCU_like": [], "UCUCUC_like": [], "U_rich": []}
    for i, kmer in enumerate(kmers.astype(str)):
        if seed_like(kmer, cuucu):
            sets["CUUCU_like"].append(i)
        if seed_like(kmer, ucucuc):
            sets["UCUCUC_like"].append(i)
        if kmer.count("U") >= 5 or "UUUUU" in kmer:
            sets["U_rich"].append(i)
    return {k: np.asarray(v, dtype=int) for k, v in sets.items()}


def best_rank(row: np.ndarray, idx: np.ndarray, kmers: np.ndarray) -> tuple[int, str, float]:
    order = np.argsort(-row)
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(1, len(order) + 1)
    local = idx[np.argmin(rank_of[idx])]
    return int(rank_of[local]), str(kmers[local]), float(row[local])


def short_id(pid: str) -> str:
    if pid.startswith("AtPTBP3"):
        return "AtPTBP3"
    if "|original=" in pid:
        return pid.split("|original=", 1)[1].split("|", 1)[0]
    return pid.split("|", 1)[0]


def write_predictions(
    out: Path,
    variant: str,
    query_ids: list[str],
    train_ids: list[str],
    kmers: np.ndarray,
    pred: np.ndarray,
    min_dist: np.ndarray,
    neighbor_df: pd.DataFrame,
    top_n: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sets = motif_sets(kmers)
    with gzip.open(out / f"{variant}_score_matrix.tsv.gz", "wt") as handle:
        handle.write("query_id\tshort_id\tkmer\tscore\n")
        for pid, row in zip(query_ids, pred):
            sid = short_id(pid)
            for kmer, score in zip(kmers, row):
                handle.write(f"{pid}\t{sid}\t{kmer}\t{float(score)}\n")

    top_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for qi, (pid, row) in enumerate(zip(query_ids, pred)):
        order = top_indices(row, top_n)
        sid = short_id(pid)
        for rank, idx in enumerate(order, start=1):
            top_rows.append(
                {
                    "query_id": pid,
                    "short_id": sid,
                    "variant": variant,
                    "rank": rank,
                    "kmer": str(kmers[idx]),
                    "score": float(row[idx]),
                    "min_latent_distance": float(min_dist[qi]),
                }
            )
        top20 = set(kmers[order[:20]].astype(str))
        cu_rank, cu_kmer, cu_score = best_rank(row, sets["CUUCU_like"], kmers)
        ucu_rank, ucu_kmer, ucu_score = best_rank(row, sets["UCUCUC_like"], kmers)
        ur_rank, ur_kmer, ur_score = best_rank(row, sets["U_rich"], kmers)
        summary_rows.append(
            {
                "query_id": pid,
                "short_id": sid,
                "variant": variant,
                "top1_kmer": str(kmers[order[0]]),
                "top5_kmers": ",".join(kmers[order[:5]].astype(str)),
                "top10_kmers": ",".join(kmers[order[:10]].astype(str)),
                "best_CUUCU_like_rank": cu_rank,
                "best_CUUCU_like_kmer": cu_kmer,
                "best_CUUCU_like_score": cu_score,
                "best_UCUCUC_like_rank": ucu_rank,
                "best_UCUCUC_like_kmer": ucu_kmer,
                "best_UCUCUC_like_score": ucu_score,
                "best_U_rich_rank": ur_rank,
                "best_U_rich_kmer": ur_kmer,
                "best_U_rich_score": ur_score,
                "contains_CUUCU_like_top20": any(seed_like(k, ["CUUCU", "UCUUC", "CUUCUC", "UCUUCU", "CUUCUU"]) for k in top20),
                "contains_UCUCUC_like_top20": any(seed_like(k, ["UCUCUC", "CUCUCU", "UCUCU", "CUCUC"]) for k in top20),
                "contains_U_rich_top20": any(k.count("U") >= 5 or "UUUUU" in k for k in top20),
                "min_latent_distance": float(min_dist[qi]),
            }
        )
    pd.DataFrame(top_rows).to_csv(out / f"{variant}_top_predicted_7mers.tsv", sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(out / f"{variant}_query_summary.tsv", sep="\t", index=False)
    if not neighbor_df.empty:
        ndf = neighbor_df.copy()
        ndf["query_id"] = ndf["test_idx"].map(dict(enumerate(query_ids)))
        ndf["train_protein_id"] = ndf["train_idx"].map(dict(enumerate(train_ids)))
        ndf.to_csv(out / f"{variant}_neighbor_table.tsv", sep="\t", index=False)


class LatentTargetDataset(Dataset):
    def __init__(self, ids: list[str], x_map: dict[str, np.ndarray], targets: np.ndarray):
        self.ids = ids
        self.x_map = x_map
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        return pid, torch.from_numpy(self.x_map[pid].astype(np.float32)), torch.from_numpy(self.targets[idx])


def train_cnn_to_jple_latent(
    x_map: dict[str, np.ndarray],
    train_ids: list[str],
    target_w: np.ndarray,
    args: argparse.Namespace,
    out: Path,
) -> PerResidueCnn:
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = LatentTargetDataset(train_ids, x_map, target_w)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    model = PerResidueCnn(
        input_dim=next(iter(x_map.values())).shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=target_w.shape[1],
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    rows: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for _, x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_smooth_l1_to_jple_latent": float(np.mean(losses))}
        rows.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            log(f"per_residue_CNN epoch={epoch} loss={row['train_smooth_l1_to_jple_latent']:.6g}")
    pd.DataFrame(rows).to_csv(out / "per_residue_cnn_jple_training_log.tsv", sep="\t", index=False)
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": {
                "input_dim": next(iter(x_map.values())).shape[1],
                "hidden_dim": args.hidden_dim,
                "latent_dim": int(target_w.shape[1]),
                "kernel_size": args.kernel_size,
                "num_blocks": args.num_blocks,
                "dropout": args.dropout,
            },
            "train_ids": train_ids,
            "target": "JPLE_w_train",
            "args": vars(args),
        },
        out / "per_residue_cnn_jple_checkpoint.pt",
    )
    return model


def predict_from_latent(
    w_query: np.ndarray,
    w_train: np.ndarray,
    y_train: np.ndarray,
    threshold: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dist_mat = cdist(w_query, w_train, "cosine")
    sim_mat = np.exp(-(dist_mat**2) / (std**2))
    preds: list[np.ndarray] = []
    min_dist: list[float] = []
    rows: list[dict[str, Any]] = []
    for qi, (dist, sim) in enumerate(zip(dist_mat, sim_mat)):
        idx = np.argwhere(sim >= threshold).flatten()
        idx = idx[sim[idx].argsort()][::-1]
        if len(idx) == 0:
            idx = np.asarray([int(np.argmax(sim))])
        weights = sim[idx]
        weights = weights / weights.sum()
        preds.append(np.sum(weights[:, None] * y_train[idx], axis=0))
        min_dist.append(float(np.min(dist)))
        for ti, d, w in zip(idx, dist[idx], weights):
            rows.append({"test_idx": qi, "train_idx": int(ti), "dist": float(d), "contribution": float(w * 100)})
    return np.asarray(preds), np.asarray(min_dist), pd.DataFrame(rows)


def standardize_pred(pred: np.ndarray) -> np.ndarray:
    std = np.std(pred, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (pred / std).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-pooled-npz", default="data/embeddings/rnacompete_domain_merged_esmc_embeddings.npz")
    p.add_argument("--query-pooled-rice-npz", default="results/final_rice_prediction/rice_inputs/rice_w1_w6_domain_merged_esmc_embeddings.npz")
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617")
    p.add_argument("--num-eigenvector", type=int, default=122)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--std", type=float, default=0.2)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260617)
    args = p.parse_args()

    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)

    out = resolve(args.output_dir) if Path(args.output_dir).exists() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    motif_ids, y_raw, kmers = load_motif(resolve(args.motif_npz))
    y_norm_all = row_l2_normalize(y_raw)

    # Variant 1: exact JPLE with pooled ESMC embeddings.
    pooled_ids, pooled_x = load_npz_embeddings(resolve(args.train_pooled_npz))
    pooled_train_ids, pooled_x_train_raw, pooled_y_train = align_train(pooled_ids, pooled_x, motif_ids, y_norm_all)
    pooled_x_train = l2_normalize(pooled_x_train_raw)
    pooled_model = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    pooled_model.fit(pooled_x_train, pooled_y_train)
    np.savez_compressed(
        out / "pooled_jple_all348_model.npz",
        train_protein_id_list=np.asarray(pooled_train_ids),
        y_train=pooled_model.y_train,
        x_train_mean=pooled_model.x_train_mean,
        y_train_mean=pooled_model.y_train_mean,
        w_train=pooled_model.w_train,
        v_train=pooled_model.v_train,
        num_eigenvector=args.num_eigenvector,
        threshold=args.threshold,
        std=args.std,
        model_type="exact_JPLE_pooled_ESMC_all348",
    )
    q_ids, q_x = load_npz_embeddings(resolve(args.query_pooled_rice_npz))
    at_ids, at_x = load_h5_mean_embeddings(resolve(args.query_atptbp3_per_residue_h5))
    pooled_query_ids = q_ids + at_ids
    pooled_query_x = l2_normalize(np.vstack([q_x, at_x]))
    pooled_pred, pooled_dist, pooled_neigh = pooled_model.predict_protein(pooled_query_x)
    pooled_pred = standardize_pred(pooled_pred)
    write_predictions(out / "pooled", "pooled_jple", pooled_query_ids, pooled_train_ids, kmers, pooled_pred, pooled_dist, pooled_neigh, args.top_n)
    log("finished pooled JPLE prediction")

    # Variant 2: per-residue CNN predicts JPLE latent coordinates, then uses JPLE neighbor reconstruction.
    per_x_map, _ = load_h5_features(resolve(args.train_per_residue_h5), None)
    per_mean_ids, per_mean_x = load_h5_mean_embeddings(resolve(args.train_per_residue_h5))
    per_train_ids, per_mean_x_train_raw, per_y_train = align_train(per_mean_ids, per_mean_x, motif_ids, y_norm_all)
    per_mean_x_train = l2_normalize(per_mean_x_train_raw)
    per_model = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    per_model.fit(per_mean_x_train, per_y_train)
    target_w = per_model.w_train.astype(np.float32)
    train_ids_for_cnn = [pid for pid in per_train_ids if pid in per_x_map]
    target_index = {pid: i for i, pid in enumerate(per_train_ids)}
    target_w = np.vstack([target_w[target_index[pid]] for pid in train_ids_for_cnn]).astype(np.float32)
    y_for_cnn = np.vstack([per_y_train[target_index[pid]] for pid in train_ids_for_cnn]).astype(np.float32)
    w_train_for_cnn = target_w
    cnn = train_cnn_to_jple_latent(per_x_map, train_ids_for_cnn, target_w, args, out / "per_residue_cnn")
    np.savez_compressed(
        out / "per_residue_mean_anchor_jple_all348_model.npz",
        train_protein_id_list=np.asarray(train_ids_for_cnn),
        y_train=y_for_cnn,
        w_train=w_train_for_cnn,
        num_eigenvector=args.num_eigenvector,
        threshold=args.threshold,
        std=args.std,
        model_type="per_residue_CNN_to_JPLE_latent_all348",
    )

    q_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    at_map, _ = load_h5_features(resolve(args.query_atptbp3_per_residue_h5), None)
    q_map.update(at_map)
    query_ids = list(q_map.keys())
    query_dataset = LatentTargetDataset(query_ids, q_map, np.zeros((len(query_ids), w_train_for_cnn.shape[1]), dtype=np.float32))
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    cnn.eval()
    pred_w: list[np.ndarray] = []
    with torch.no_grad():
        for _, x, mask, _ in query_loader:
            pred_w.append(cnn(x.to(args.device), mask.to(args.device)).cpu().numpy())
    pred_w_arr = np.vstack(pred_w)
    per_pred, per_dist, per_neigh = predict_from_latent(pred_w_arr, w_train_for_cnn, y_for_cnn, args.threshold, args.std)
    per_pred = standardize_pred(per_pred)
    write_predictions(out / "per_residue_cnn", "per_residue_cnn_jple", query_ids, train_ids_for_cnn, kmers, per_pred, per_dist, per_neigh, args.top_n)
    log("finished per-residue CNN -> JPLE latent prediction")

    config = vars(args)
    config["train_n_pooled"] = len(pooled_train_ids)
    config["train_n_per_residue_cnn"] = len(train_ids_for_cnn)
    (out / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
