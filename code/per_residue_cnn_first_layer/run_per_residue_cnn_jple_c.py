#!/usr/bin/env python3
"""Train per-residue CNN to JPLE latent with C-group profile/ranking losses."""

from __future__ import annotations

import argparse
import json
import os
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))

from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402
from cnn_model_utils import PerResidueCnn, collate_batch, load_h5_features, setup_threads  # noqa: E402
from run_jple_embedding_variants import (  # noqa: E402
    align_train,
    l2_normalize,
    load_h5_mean_embeddings,
    load_motif,
    predict_from_latent,
    resolve,
    row_l2_normalize,
    seed_all,
    standardize_pred,
    write_predictions,
)


def log(msg: str) -> None:
    print(f"[per-residue-cnn-jple-C] {msg}", flush=True)


class CGroupDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        x_map: dict[str, np.ndarray],
        target_w: np.ndarray,
        true_profile: np.ndarray,
        topk: np.ndarray,
        decoy: np.ndarray,
    ):
        self.ids = ids
        self.x_map = x_map
        self.target_w = target_w.astype(np.float32)
        self.true_profile = true_profile.astype(np.float32)
        self.topk = topk.astype(np.int64)
        self.decoy = decoy.astype(np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        return (
            pid,
            torch.from_numpy(self.x_map[pid].astype(np.float32)),
            torch.from_numpy(self.target_w[idx]),
            torch.from_numpy(self.true_profile[idx]),
            torch.from_numpy(self.topk[idx]),
            torch.from_numpy(self.decoy[idx]),
        )


def collate_cgroup(batch):
    ids = [b[0] for b in batch]
    simple = [(b[0], b[1], b[2]) for b in batch]
    _, padded, mask, target_w = collate_batch(simple)
    true_profile = torch.stack([b[3] for b in batch])
    topk = torch.stack([b[4] for b in batch])
    decoy = torch.stack([b[5] for b in batch])
    return ids, padded, mask, target_w, true_profile, topk, decoy


def build_topk_decoys(y: np.ndarray, top_k: int, decoy_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    top_rows: list[np.ndarray] = []
    decoy_rows: list[np.ndarray] = []
    all_idx = np.arange(y.shape[1])
    for row in y:
        order = np.argsort(-row)
        top_rows.append(order[:top_k].astype(np.int64))
        banned = set(order[: max(200, top_k)].tolist())
        candidates = np.asarray([i for i in all_idx if i not in banned], dtype=np.int64)
        replace = len(candidates) < decoy_n
        decoy_rows.append(rng.choice(candidates, size=decoy_n, replace=replace).astype(np.int64))
    return np.vstack(top_rows), np.vstack(decoy_rows)


def soft_jple_profile_torch(pred_w: torch.Tensor, train_w: torch.Tensor, y_train: torch.Tensor, std: float) -> torch.Tensor:
    pred_norm = F.normalize(pred_w, dim=1)
    train_norm = F.normalize(train_w, dim=1)
    cos = pred_norm @ train_norm.T
    dist = 1.0 - cos
    sim = torch.exp(-(dist * dist) / (std * std))
    weights = sim / sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights @ y_train


def pearson_corr_loss(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    true_c = true - true.mean(dim=1, keepdim=True)
    numer = (pred_c * true_c).sum(dim=1)
    denom = torch.sqrt((pred_c * pred_c).sum(dim=1) * (true_c * true_c).sum(dim=1)).clamp_min(eps)
    return (1.0 - numer / denom).mean()


def topk_ranking_loss(pred: torch.Tensor, topk: torch.Tensor, decoy: torch.Tensor, margin: float) -> torch.Tensor:
    top_scores = pred.gather(1, topk).unsqueeze(2)
    decoy_scores = pred.gather(1, decoy).unsqueeze(1)
    return torch.relu(margin - top_scores + decoy_scores).mean()


def predict_soft_from_latent(
    w_query: np.ndarray,
    w_train: np.ndarray,
    y_train: np.ndarray,
    std: float,
    neighbor_top_n: int = 50,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dist = cdist(w_query, w_train, "cosine")
    sim = np.exp(-(dist**2) / (std**2))
    weights = sim / np.clip(sim.sum(axis=1, keepdims=True), 1e-12, None)
    pred = weights @ y_train
    rows: list[dict[str, Any]] = []
    for qi in range(weights.shape[0]):
        idx = np.argsort(-weights[qi])[:neighbor_top_n]
        for ti in idx:
            rows.append(
                {
                    "test_idx": qi,
                    "train_idx": int(ti),
                    "dist": float(dist[qi, ti]),
                    "contribution": float(weights[qi, ti] * 100),
                }
            )
    return pred, dist.min(axis=1), pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn_jple_C")
    p.add_argument("--num-eigenvector", type=int, default=122)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--std", type=float, default=0.2)
    p.add_argument("--latent-loss-weight", type=float, default=1.0)
    p.add_argument("--profile-loss-weight", type=float, default=0.5)
    p.add_argument("--ranking-loss-weight", type=float, default=0.1)
    p.add_argument("--ranking-margin", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--decoy-n", type=int, default=200)
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
    device = torch.device(args.device)

    out = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    motif_ids, y_raw, kmers = load_motif(resolve(args.motif_npz))
    y_norm = row_l2_normalize(y_raw)
    per_mean_ids, per_mean_x = load_h5_mean_embeddings(resolve(args.train_per_residue_h5))
    train_ids, mean_x_train_raw, y_train = align_train(per_mean_ids, per_mean_x, motif_ids, y_norm)
    mean_x_train = l2_normalize(mean_x_train_raw)

    anchor = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    anchor.fit(mean_x_train, y_train)
    target_w_all = anchor.w_train.astype(np.float32)

    x_map, _ = load_h5_features(resolve(args.train_per_residue_h5), None)
    train_ids = [pid for pid in train_ids if pid in x_map]
    idx = {pid: i for i, pid in enumerate(per_mean_ids)}
    anchor_idx = {pid: i for i, pid in enumerate(anchor.y_train.shape[0] and train_ids)}
    original_idx = {pid: i for i, pid in enumerate([pid for pid in per_mean_ids if pid in set(motif_ids)])}
    # Rebuild by the anchor's actual train order before filtering to CNN-available IDs.
    anchor_train_ids, _, _ = align_train(per_mean_ids, per_mean_x, motif_ids, y_norm)
    anchor_lookup = {pid: i for i, pid in enumerate(anchor_train_ids)}
    target_w = np.vstack([target_w_all[anchor_lookup[pid]] for pid in train_ids]).astype(np.float32)
    true_profile = np.vstack([y_train[anchor_lookup[pid]] for pid in train_ids]).astype(np.float32)
    train_w_for_reconstruct = target_w.copy()
    y_for_reconstruct = true_profile.copy()
    topk, decoy = build_topk_decoys(true_profile, args.top_k, args.decoy_n, args.seed)

    dataset = CGroupDataset(train_ids, x_map, target_w, true_profile, topk, decoy)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_cgroup)
    model = PerResidueCnn(
        input_dim=next(iter(x_map.values())).shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=target_w.shape[1],
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    latent_loss_fn = nn.SmoothL1Loss()
    train_w_t = torch.tensor(train_w_for_reconstruct, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_for_reconstruct, dtype=torch.float32, device=device)

    rows: list[dict[str, float]] = []
    log(f"train_n={len(train_ids)} epochs={args.epochs} loss=latent+{args.profile_loss_weight}*profile+{args.ranking_loss_weight}*ranking")
    for epoch in range(1, args.epochs + 1):
        model.train()
        vals: list[dict[str, float]] = []
        for _, x, mask, target, profile, topk_b, decoy_b in loader:
            x = x.to(device)
            mask = mask.to(device)
            target = target.to(device)
            profile = profile.to(device)
            topk_b = topk_b.to(device)
            decoy_b = decoy_b.to(device)
            opt.zero_grad(set_to_none=True)
            pred_w = model(x, mask)
            pred_profile = soft_jple_profile_torch(pred_w, train_w_t, y_train_t, args.std)
            latent_loss = latent_loss_fn(pred_w, target)
            profile_loss = pearson_corr_loss(pred_profile, profile)
            ranking_loss = topk_ranking_loss(pred_profile, topk_b, decoy_b, args.ranking_margin)
            total = (
                args.latent_loss_weight * latent_loss
                + args.profile_loss_weight * profile_loss
                + args.ranking_loss_weight * ranking_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            vals.append(
                {
                    "total_loss": float(total.detach().cpu()),
                    "latent_loss": float(latent_loss.detach().cpu()),
                    "profile_corr_loss": float(profile_loss.detach().cpu()),
                    "ranking_loss": float(ranking_loss.detach().cpu()),
                }
            )
        row = {"epoch": epoch}
        for key in vals[0]:
            row[f"train_{key}"] = float(np.mean([v[key] for v in vals]))
        rows.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            log(
                "epoch=%d total=%.6g latent=%.6g profile=%.6g ranking=%.6g"
                % (
                    epoch,
                    row["train_total_loss"],
                    row["train_latent_loss"],
                    row["train_profile_corr_loss"],
                    row["train_ranking_loss"],
                )
            )

    pd.DataFrame(rows).to_csv(out / "training_log.tsv", sep="\t", index=False)
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
            "train_w": train_w_for_reconstruct,
            "y_train": y_for_reconstruct,
            "kmers": kmers,
            "args": vars(args),
        },
        out / "per_residue_cnn_jple_C_checkpoint.pt",
    )

    q_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    at_map, _ = load_h5_features(resolve(args.query_atptbp3_per_residue_h5), None)
    q_map.update(at_map)
    query_ids = list(q_map.keys())
    dummy = np.zeros((len(query_ids), target_w.shape[1]), dtype=np.float32)
    from run_jple_embedding_variants import LatentTargetDataset  # local import avoids circular type noise

    query_loader = DataLoader(LatentTargetDataset(query_ids, q_map, dummy), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    model.eval()
    pred_w_rows: list[np.ndarray] = []
    with torch.no_grad():
        for _, x, mask, _ in query_loader:
            pred_w_rows.append(model(x.to(device), mask.to(device)).cpu().numpy())
    query_w = np.vstack(pred_w_rows)
    pred, dist, neigh = predict_soft_from_latent(query_w, train_w_for_reconstruct, y_for_reconstruct, args.std)
    pred = standardize_pred(pred)
    write_predictions(out, "per_residue_cnn_jple_C", query_ids, train_ids, kmers, pred, dist, neigh, args.top_n)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    log("finished prediction")


if __name__ == "__main__":
    main()
