#!/usr/bin/env python3
"""Train per-residue RBD CNN motif model from a fixed split.

Enhanced objective:
  latent regression + differentiable profile correlation loss + top-k ranking loss.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from cnn_model_utils import PerResidueCnn, first_key, load_h5_features, resolve_path, setup_threads  # noqa: E402


def log(msg: str) -> None:
    print(f"[profile-ranking-cnn] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_motif(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    z = np.load(path, allow_pickle=True)
    ids = np.asarray(z[first_key(z, ["profile_ids", "protein_ids", "ids", "names"])]).astype(str)
    y = np.asarray(z[first_key(z, ["zscores", "scores", "Y", "profiles"])], dtype=np.float32)
    kmers = np.asarray(z["kmers"]).astype(str)
    return ids, y, kmers, {pid: i for i, pid in enumerate(ids)}


def latent_to_profile_torch(
    pred_latent: torch.Tensor,
    svd_components: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
) -> torch.Tensor:
    pred_profile_scaled = pred_latent @ svd_components
    return pred_profile_scaled * scaler_scale + scaler_mean


def pearson_corr_loss(pred_profile: torch.Tensor, true_profile: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_c = pred_profile - pred_profile.mean(dim=1, keepdim=True)
    true_c = true_profile - true_profile.mean(dim=1, keepdim=True)
    num = (pred_c * true_c).sum(dim=1)
    den = torch.sqrt((pred_c.square().sum(dim=1) * true_c.square().sum(dim=1)).clamp_min(eps))
    corr = num / den
    return (1.0 - corr).mean()


def topk_ranking_loss(
    pred_profile: torch.Tensor,
    topk_indices: torch.Tensor,
    decoy_indices: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    top_scores = pred_profile.gather(1, topk_indices).unsqueeze(2)
    decoy_scores = pred_profile.gather(1, decoy_indices).unsqueeze(1)
    return torch.relu(margin - top_scores + decoy_scores).mean()


class MotifProfileDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        x_map: dict[str, np.ndarray],
        latent_by_id: dict[str, np.ndarray],
        profile_by_id: dict[str, np.ndarray],
        topk_by_id: dict[str, np.ndarray],
        decoy_by_id: dict[str, np.ndarray],
        max_len: int | None = None,
    ):
        self.ids = ids
        self.x_map = x_map
        self.latent_by_id = latent_by_id
        self.profile_by_id = profile_by_id
        self.topk_by_id = topk_by_id
        self.decoy_by_id = decoy_by_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        x = self.x_map[pid]
        if self.max_len is not None and x.shape[0] > self.max_len:
            x = x[: self.max_len]
        return (
            pid,
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(self.latent_by_id[pid].astype(np.float32)),
            torch.from_numpy(self.profile_by_id[pid].astype(np.float32)),
            torch.from_numpy(self.topk_by_id[pid].astype(np.int64)),
            torch.from_numpy(self.decoy_by_id[pid].astype(np.int64)),
        )


def collate_profile_batch(batch):
    ids, xs, latents, profiles, topks, decoys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    dim = int(xs[0].shape[1])
    padded = torch.zeros((len(xs), max_len, dim), dtype=torch.float32)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)
    for i, x in enumerate(xs):
        length = x.shape[0]
        padded[i, :length] = x
        mask[i, :length] = True
    return list(ids), padded, mask, torch.stack(latents), torch.stack(profiles), torch.stack(topks), torch.stack(decoys)


def ids_from_split(split_tsv: Path, split_name: str, motif_ids: set[str], x_map: dict[str, np.ndarray]) -> list[str]:
    df = pd.read_csv(split_tsv, sep="\t")
    if "protein_id" not in df.columns or "split" not in df.columns:
        raise ValueError("split TSV must contain protein_id and split columns")
    ids = df.loc[df["split"].astype(str) == split_name, "protein_id"].astype(str).tolist()
    return [pid for pid in ids if pid in motif_ids and pid in x_map]


def filter_finite_ids(ids: list[str], y: np.ndarray, id2y: dict[str, int]) -> tuple[list[str], list[str]]:
    keep: list[str] = []
    drop: list[str] = []
    for pid in ids:
        row = y[id2y[pid]]
        if np.isfinite(row).all():
            keep.append(pid)
        else:
            drop.append(pid)
    return keep, drop


def build_latent_and_profile_maps(
    ids: list[str],
    y: np.ndarray,
    id2y: dict[str, int],
    scaler: StandardScaler,
    svd: TruncatedSVD,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    profiles = {pid: y[id2y[pid]].astype(np.float32) for pid in ids}
    scaled = scaler.transform(np.stack([profiles[pid] for pid in ids])).astype(np.float32)
    latent = svd.transform(scaled).astype(np.float32)
    latent_by_id = {pid: row for pid, row in zip(ids, latent)}
    return latent_by_id, profiles


def build_topk_decoys(
    ids: list[str],
    profile_by_id: dict[str, np.ndarray],
    top_k: int,
    decoy_n: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    topk_by_id: dict[str, np.ndarray] = {}
    decoy_by_id: dict[str, np.ndarray] = {}
    for pid in ids:
        profile = profile_by_id[pid]
        order = np.argsort(-profile)
        topk = order[:top_k].astype(np.int64)
        decoy_pool = order[min(200, len(order)) :]
        if len(decoy_pool) < decoy_n:
            topk_set = set(int(v) for v in topk)
            decoy_pool = np.array([idx for idx in order if int(idx) not in topk_set], dtype=np.int64)
        replace = len(decoy_pool) < decoy_n
        decoy = rng.choice(decoy_pool, size=decoy_n, replace=replace).astype(np.int64)
        topk_by_id[pid] = topk
        decoy_by_id[pid] = decoy
    return topk_by_id, decoy_by_id


def compute_loss_parts(
    model: nn.Module,
    batch,
    device: torch.device,
    latent_criterion: nn.Module,
    svd_components_t: torch.Tensor,
    scaler_mean_t: torch.Tensor,
    scaler_scale_t: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    _, x, mask, true_latent, true_profile, topk_idx, decoy_idx = batch
    x = x.to(device)
    mask = mask.to(device)
    true_latent = true_latent.to(device)
    true_profile = true_profile.to(device)
    topk_idx = topk_idx.to(device)
    decoy_idx = decoy_idx.to(device)
    pred_latent = model(x, mask)
    pred_profile = latent_to_profile_torch(pred_latent, svd_components_t, scaler_mean_t, scaler_scale_t)
    latent_loss = latent_criterion(pred_latent, true_latent)
    profile_loss = pearson_corr_loss(pred_profile, true_profile)
    ranking_loss = topk_ranking_loss(pred_profile, topk_idx, decoy_idx, margin=args.ranking_margin)
    total_loss = (
        args.latent_loss_weight * latent_loss
        + args.profile_loss_weight * profile_loss
        + args.ranking_loss_weight * ranking_loss
    )
    parts = {
        "total_loss": float(total_loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
        "profile_corr_loss": float(profile_loss.detach().cpu()),
        "ranking_loss": float(ranking_loss.detach().cpu()),
    }
    return total_loss, parts, pred_profile


def mean_loss_parts(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return out
    for key in rows[0]:
        out[f"{prefix}_{key}"] = float(np.mean([r[key] for r in rows]))
    return out


def profile_metrics_from_arrays(pred_rows: list[np.ndarray], true_rows: list[np.ndarray]) -> dict[str, float]:
    pearsons = []
    spearmans = []
    top20 = []
    top50 = []
    top1_ranks = []
    top5_best_ranks = []
    for pred, true in zip(pred_rows, true_rows):
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        pred_c = pred - pred.mean()
        true_c = true - true.mean()
        den = np.sqrt(np.sum(pred_c * pred_c) * np.sum(true_c * true_c))
        pearsons.append(float(np.sum(pred_c * true_c) / den) if den > 0 else np.nan)
        pred_rank = rankdata(pred)
        true_rank = rankdata(true)
        pred_rank_c = pred_rank - pred_rank.mean()
        true_rank_c = true_rank - true_rank.mean()
        rank_den = np.sqrt(np.sum(pred_rank_c * pred_rank_c) * np.sum(true_rank_c * true_rank_c))
        spearmans.append(float(np.sum(pred_rank_c * true_rank_c) / rank_den) if rank_den > 0 else np.nan)
        true_order = np.argsort(-true)
        pred_order = np.argsort(-pred)
        pred_rank_pos = np.empty_like(pred_order)
        pred_rank_pos[pred_order] = np.arange(1, len(pred_order) + 1)
        true_top20 = set(true_order[:20].tolist())
        pred_top20 = set(pred_order[:20].tolist())
        true_top50 = set(true_order[:50].tolist())
        pred_top50 = set(pred_order[:50].tolist())
        top20.append(len(true_top20 & pred_top20) / 20.0)
        top50.append(len(true_top50 & pred_top50) / 50.0)
        top1_ranks.append(float(pred_rank_pos[true_order[0]]))
        top5_best_ranks.append(float(np.min(pred_rank_pos[true_order[:5]])))
    return {
        "profile_pearson_mean": float(np.nanmean(pearsons)),
        "profile_spearman_mean": float(np.nanmean(spearmans)),
        "top20_recovery": float(np.nanmean(top20)),
        "top50_recovery": float(np.nanmean(top50)),
        "true_top1_rank_median": float(np.nanmedian(top1_ranks)),
        "true_top5_best_rank_median": float(np.nanmedian(top5_best_ranks)),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    latent_criterion: nn.Module,
    svd_components_t: torch.Tensor,
    scaler_mean_t: torch.Tensor,
    scaler_scale_t: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray]]:
    model.eval()
    loss_rows: list[dict[str, float]] = []
    pred_rows: list[np.ndarray] = []
    true_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            _, parts, pred_profile = compute_loss_parts(
                model,
                batch,
                device,
                latent_criterion,
                svd_components_t,
                scaler_mean_t,
                scaler_scale_t,
                args,
            )
            loss_rows.append(parts)
            true_profile = batch[4]
            pred_rows.extend(pred_profile.detach().cpu().numpy())
            true_rows.extend(true_profile.numpy())
    metrics = mean_loss_parts(loss_rows, "val")
    metrics.update({f"val_{k}": v for k, v in profile_metrics_from_arrays(pred_rows, true_rows).items()})
    return metrics, pred_rows, true_rows


def selection_value(metrics: dict[str, float], selection_metric: str) -> tuple[float, bool]:
    if selection_metric == "val_total_loss":
        return -float(metrics["val_total_loss"]), True
    if selection_metric == "val_profile_pearson":
        return float(metrics["val_profile_pearson_mean"]), True
    if selection_metric == "val_top20_recovery":
        return float(metrics["val_top20_recovery"]), True
    if selection_metric == "combined":
        return float(metrics["val_profile_pearson_mean"] + metrics["val_top20_recovery"]), True
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def apply_loss_preset(args: argparse.Namespace) -> None:
    if args.loss_preset is None:
        return
    presets = {
        "A": (0.2, 0.1),
        "B": (0.2, 0.2),
        "C": (0.5, 0.1),
        "D": (0.5, 0.2),
    }
    args.profile_loss_weight, args.ranking_loss_weight = presets[args.loss_preset]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-tsv", required=True)
    ap.add_argument("--per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    ap.add_argument("--per-residue-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    ap.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--latent-dim", type=int, default=50)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--latent-loss-weight", type=float, default=1.0)
    ap.add_argument("--profile-loss-weight", type=float, default=0.2)
    ap.add_argument("--ranking-loss-weight", type=float, default=0.1)
    ap.add_argument("--ranking-margin", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--decoy-n", type=int, default=200)
    ap.add_argument("--loss-type", choices=["smooth_l1", "mse"], default="smooth_l1")
    ap.add_argument("--selection-metric", choices=["val_total_loss", "val_profile_pearson", "val_top20_recovery", "combined"], default="combined")
    ap.add_argument("--loss-preset", choices=["A", "B", "C", "D"], default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--quick-test", action="store_true")
    ap.add_argument("--torch-num-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260617)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    apply_loss_preset(args)
    if args.quick_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)

    out = Path(args.output_dir)
    out = ROOT / out if not out.is_absolute() else out
    out.mkdir(parents=True, exist_ok=True)

    split_tsv = resolve_path(args.split_tsv, [Path(args.split_tsv).name])
    h5_path = resolve_path(args.per_residue_h5, ["rnacompete_rbd_per_residue_esmc.h5"])
    manifest_path = resolve_path(args.per_residue_manifest, ["rnacompete_rbd_per_residue_manifest.tsv"], required=False)
    motif_path = resolve_path(args.motif_npz, ["motif_profiles.npz"])

    x_map, _ = load_h5_features(h5_path, manifest_path)
    motif_ids, y, kmers, id2y = load_motif(motif_path)
    motif_id_set = set(motif_ids)
    train_ids = ids_from_split(split_tsv, "train", motif_id_set, x_map)
    val_ids = ids_from_split(split_tsv, "val", motif_id_set, x_map)
    test_ids = ids_from_split(split_tsv, "test", motif_id_set, x_map)
    train_ids, train_dropped = filter_finite_ids(train_ids, y, id2y)
    val_ids, val_dropped = filter_finite_ids(val_ids, y, id2y)
    test_ids, test_dropped = filter_finite_ids(test_ids, y, id2y)
    if args.quick_test:
        train_ids = train_ids[: min(16, len(train_ids))]
        val_ids = val_ids[: min(8, len(val_ids))]
        test_ids = test_ids[: min(8, len(test_ids))]
    if not train_ids or not val_ids:
        raise RuntimeError(f"Need non-empty train and val sets, got train={len(train_ids)} val={len(val_ids)}")

    y_train = y[[id2y[pid] for pid in train_ids]]
    scaler = StandardScaler(with_mean=True, with_std=True)
    y_train_scaled = scaler.fit_transform(y_train).astype(np.float32)
    latent_dim = max(2, min(args.latent_dim, len(train_ids) - 1, y.shape[1] - 1))
    svd = TruncatedSVD(n_components=latent_dim, random_state=args.seed)
    svd.fit(y_train_scaled)

    all_used_ids = train_ids + val_ids + test_ids
    latent_by_id, profile_by_id = build_latent_and_profile_maps(all_used_ids, y, id2y, scaler, svd)
    topk_by_id, decoy_by_id = build_topk_decoys(all_used_ids, profile_by_id, args.top_k, args.decoy_n, args.seed)

    train_loader = DataLoader(
        MotifProfileDataset(train_ids, x_map, latent_by_id, profile_by_id, topk_by_id, decoy_by_id, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_profile_batch,
    )
    val_loader = DataLoader(
        MotifProfileDataset(val_ids, x_map, latent_by_id, profile_by_id, topk_by_id, decoy_by_id, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_profile_batch,
    )
    test_loader = DataLoader(
        MotifProfileDataset(test_ids, x_map, latent_by_id, profile_by_id, topk_by_id, decoy_by_id, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_profile_batch,
    )

    model = PerResidueCnn(
        next(iter(x_map.values())).shape[1],
        args.hidden_dim,
        latent_dim,
        args.kernel_size,
        args.num_blocks,
        args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    latent_criterion: nn.Module = nn.SmoothL1Loss() if args.loss_type == "smooth_l1" else nn.MSELoss()
    svd_components_t = torch.tensor(svd.components_.astype(np.float32), device=device)
    scaler_mean_t = torch.tensor(scaler.mean_.astype(np.float32), device=device)
    scaler_scale_t = torch.tensor(scaler.scale_.astype(np.float32), device=device)

    log(
        "train=%d val=%d test=%d latent_dim=%d seed=%d loss=%s weights=(latent %.3g, profile %.3g, ranking %.3g)"
        % (
            len(train_ids),
            len(val_ids),
            len(test_ids),
            latent_dim,
            args.seed,
            args.loss_type,
            args.latent_loss_weight,
            args.profile_loss_weight,
            args.ranking_loss_weight,
        )
    )
    if train_dropped or val_dropped or test_dropped:
        log(f"dropped_nonfinite train={len(train_dropped)} val={len(val_dropped)} test={len(test_dropped)}")

    best_value = -float("inf")
    best_epoch = 0
    best_state = None
    best_metrics: dict[str, float] = {}
    wait = 0
    train_log_rows = []
    val_metric_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_parts = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            total_loss, parts, _ = compute_loss_parts(
                model,
                batch,
                device,
                latent_criterion,
                svd_components_t,
                scaler_mean_t,
                scaler_scale_t,
                args,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            train_parts.append(parts)

        train_metrics = mean_loss_parts(train_parts, "train")
        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
            latent_criterion,
            svd_components_t,
            scaler_mean_t,
            scaler_scale_t,
            args,
        )
        current_value, _ = selection_value(val_metrics, args.selection_metric)
        row = {"epoch": epoch, **train_metrics, **val_metrics, "selection_value": current_value}
        train_log_rows.append(row)
        val_metric_rows.append({"epoch": epoch, **val_metrics, "selection_value": current_value})
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = val_metrics.copy()
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            log(
                "epoch=%d train_total=%.5g val_total=%.5g val_pearson=%.4f val_top20=%.4f best_epoch=%d"
                % (
                    epoch,
                    train_metrics["train_total_loss"],
                    val_metrics["val_total_loss"],
                    val_metrics["val_profile_pearson_mean"],
                    val_metrics["val_top20_recovery"],
                    best_epoch,
                )
            )
        if wait >= args.patience:
            log(f"early_stop epoch={epoch} best_epoch={best_epoch} best_selection_value={best_value:.6g}")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint state was captured")
    model.load_state_dict(best_state)
    test_metrics, _, _ = evaluate(
        model,
        test_loader,
        device,
        latent_criterion,
        svd_components_t,
        scaler_mean_t,
        scaler_scale_t,
        args,
    )

    pd.DataFrame(train_log_rows).to_csv(out / "training_log.tsv", sep="\t", index=False)
    pd.DataFrame(val_metric_rows).to_csv(out / "validation_metrics.tsv", sep="\t", index=False)
    with gzip.open(out / "split_ids_used.tsv.gz", "wt") as handle:
        handle.write("split\tprotein_id\n")
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            for pid in ids:
                handle.write(f"{split_name}\t{pid}\n")

    model_config = {
        "input_dim": int(next(iter(x_map.values())).shape[1]),
        "hidden_dim": args.hidden_dim,
        "latent_dim": latent_dim,
        "kernel_size": args.kernel_size,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
    }
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": model_config,
        "training_args": vars(args),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "svd_components": svd.components_.astype(np.float32),
        "kmer_list": kmers,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "selection_metric": args.selection_metric,
        "selection_value": best_value,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        # Legacy compatibility for earlier prediction scripts.
        "preprocess": {
            "svd_components": svd.components_.astype(np.float32),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
        },
        "kmers": kmers,
        "args": vars(args),
    }
    ckpt_path = out / "best_model.pt"
    legacy_ckpt_path = out / "best_cnn_fixed_split_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, legacy_ckpt_path)
    summary = {
        "checkpoint": str(ckpt_path),
        "legacy_checkpoint": str(legacy_ckpt_path),
        "seed": args.seed,
        "split_tsv": str(split_tsv),
        "train_n": len(train_ids),
        "val_n": len(val_ids),
        "test_n": len(test_ids),
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "selection_value": best_value,
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "loss_weights": {
            "latent": args.latent_loss_weight,
            "profile": args.profile_loss_weight,
            "ranking": args.ranking_loss_weight,
        },
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"checkpoint={ckpt_path}")
    log(
        "best_epoch=%d val_pearson=%.4f val_top20=%.4f test_pearson=%.4f test_top20=%.4f"
        % (
            best_epoch,
            best_metrics.get("val_profile_pearson_mean", float("nan")),
            best_metrics.get("val_top20_recovery", float("nan")),
            test_metrics.get("val_profile_pearson_mean", float("nan")),
            test_metrics.get("val_top20_recovery", float("nan")),
        )
    )


if __name__ == "__main__":
    main()
