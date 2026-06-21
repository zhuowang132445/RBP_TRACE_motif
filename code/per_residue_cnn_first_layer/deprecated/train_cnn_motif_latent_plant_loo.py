#!/usr/bin/env python3
"""Train exploratory per-residue RBD CNN motif-latent model and compare with RBPTrace plant LOO."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3] if SCRIPT_PATH.parent.name == "deprecated" else SCRIPT_PATH.parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

try:
    from rbp_trace_core.model import RBPTraceFirstLayer
except Exception:
    RBPTraceFirstLayer = None


def log(msg: str) -> None:
    print(f"[per-residue-cnn-loo] {msg}", flush=True)


def setup_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    torch.set_num_interop_threads(max(1, int(n)))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_text: str | None, patterns: list[str], required: bool = True) -> Path | None:
    if path_text:
        p = Path(path_text)
        for c in [p, ROOT / p]:
            if c.exists():
                return c
        matches = list(ROOT.rglob(p.name))
        if matches:
            return matches[0]
    for pat in patterns:
        matches = list(ROOT.rglob(pat))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Could not resolve {path_text or patterns}")
    return None


def safe_key(pid: str) -> str:
    return pid.replace("/", "__slash__")


def first_key(npz, keys: list[str]) -> str:
    for k in keys:
        if k in npz.files:
            return k
    raise KeyError(f"None of {keys} in {npz.files}")


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def metric_profile(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        out["pearson_corr"] = np.nan
        out["spearman_corr"] = np.nan
    else:
        try:
            out["pearson_corr"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            out["pearson_corr"] = np.nan
        try:
            out["spearman_corr"] = float(spearmanr(y_true, y_pred, nan_policy="omit")[0])
        except Exception:
            out["spearman_corr"] = np.nan
    for k in [10, 20, 50]:
        true_top = set(np.argsort(-y_true)[:k])
        pred_top = set(np.argsort(-y_pred)[:k])
        val = len(true_top & pred_top) / float(k)
        out[f"top{k}_overlap"] = float(val)
    return out


class RbdEmbeddingDataset(Dataset):
    def __init__(self, ids: list[str], x_map: dict[str, np.ndarray], y: np.ndarray, id_to_y: dict[str, int], max_len: int | None = None):
        self.ids = ids
        self.x_map = x_map
        self.y = y
        self.id_to_y = id_to_y
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        x = self.x_map[pid]
        if self.max_len is not None and x.shape[0] > self.max_len:
            x = x[: self.max_len]
        return pid, torch.from_numpy(x.astype(np.float32)), torch.from_numpy(self.y[self.id_to_y[pid]].astype(np.float32))


def collate_batch(batch):
    ids, xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    dim = int(xs[0].shape[1])
    padded = torch.zeros((len(xs), max_len, dim), dtype=torch.float32)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)
    for i, x in enumerate(xs):
        l = x.shape[0]
        padded[i, :l] = x
        mask[i, :l] = True
    return list(ids), padded, mask, torch.stack(ys)


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class PerResidueCnn(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, kernel_size: int, num_blocks: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.input = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden_dim, kernel_size, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        mask_f = mask.unsqueeze(1).to(h.dtype)
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        mean_pool = (h * mask_f).sum(dim=2) / denom
        h_masked = h.masked_fill(~mask.unsqueeze(1), -1e9)
        max_pool = h_masked.max(dim=2).values
        return self.head(torch.cat([mean_pool, max_pool], dim=1))


def load_h5_features(path: Path, manifest_path: Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    x_map: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    manifest_ok = None
    if manifest_path and manifest_path.exists():
        m = pd.read_csv(manifest_path, sep="\t")
        if "status" in m.columns and "protein_id" in m.columns:
            manifest_ok = set(m.loc[m["status"].astype(str) == "ok", "protein_id"].astype(str))
    with h5py.File(path, "r") as h5:
        if "metadata" in h5 and "protein_ids" in h5["metadata"]:
            ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["metadata/protein_ids"][()]]
        else:
            ids = list(h5["embeddings"].keys())
        for pid in ids:
            if manifest_ok is not None and pid not in manifest_ok:
                continue
            key = safe_key(pid)
            if key not in h5["embeddings"]:
                continue
            arr = np.asarray(h5["embeddings"][key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                x_map[pid] = arr
                lengths[pid] = int(arr.shape[0])
    return x_map, lengths


def load_tables(args) -> dict[str, Any]:
    motif_path = resolve_path(args.motif_npz, ["motif_profiles.npz"])
    motif = np.load(motif_path, allow_pickle=True)
    motif_ids = np.asarray(motif[first_key(motif, ["profile_ids", "protein_ids", "ids", "names"])]).astype(str)
    y = np.asarray(motif[first_key(motif, ["zscores", "scores", "Y", "profiles"])], dtype=np.float32)

    plant = pd.read_csv(resolve_path(args.plant_label_tsv, ["plant_nonplant_label_check.tsv"]), sep="\t")
    plant = plant[[c for c in ["protein_id", "species", "is_plant"] if c in plant.columns]].copy()
    plant["is_plant"] = bool_series(plant["is_plant"])

    domain_path = resolve_path(args.domain_annotation_tsv, ["domain_annotation_check.tsv"], required=False)
    domain = pd.read_csv(domain_path, sep="\t") if domain_path else pd.DataFrame(columns=["protein_id"])
    keep_cols = [c for c in ["protein_id", "domain_family", "domain_architecture", "domain_count"] if c in domain.columns]
    domain = domain[keep_cols].drop_duplicates("protein_id") if keep_cols else pd.DataFrame(columns=["protein_id"])

    meta = pd.DataFrame({"protein_id": motif_ids})
    meta = meta.merge(plant.drop_duplicates("protein_id"), on="protein_id", how="left")
    meta = meta.merge(domain, on="protein_id", how="left")
    meta["species"] = meta["species"].fillna("Unknown")
    meta["is_plant"] = bool_series(meta["is_plant"].fillna(False))
    meta["domain_family"] = meta["domain_family"].fillna("Unknown").astype(str)
    meta["domain_architecture"] = meta["domain_architecture"].fillna("Unknown").astype(str)
    return {"motif_ids": motif_ids, "motif_scores": y, "id_to_y": {pid: i for i, pid in enumerate(motif_ids)}, "meta": meta}


def split_train_val(available: list[str], meta: pd.DataFrame, seed: int, val_fraction: float) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    meta_i = meta.set_index("protein_id")
    plants = [pid for pid in available if bool(meta_i.loc[pid, "is_plant"])]
    nonplants = [pid for pid in available if not bool(meta_i.loc[pid, "is_plant"])]
    rng.shuffle(plants)
    rng.shuffle(nonplants)
    n_val = max(4, int(round(len(available) * val_fraction)))
    n_val_plant = min(max(1, int(round(len(plants) * val_fraction))), max(0, len(plants) - 1)) if len(plants) > 1 else 0
    val = plants[:n_val_plant]
    remain_slots = max(0, n_val - len(val))
    val += nonplants[:remain_slots]
    val = [pid for pid in val if pid in available]
    if len(val) == 0:
        val = available[:1]
    train = [pid for pid in available if pid not in set(val)]
    return train, val


def fit_fold_svd(y: np.ndarray, ids: list[str], id_to_y: dict[str, int], latent_dim: int) -> tuple[TruncatedSVD, int]:
    idx = [id_to_y[pid] for pid in ids]
    dim = max(2, min(latent_dim, len(idx) - 1, y.shape[1] - 1))
    svd = TruncatedSVD(n_components=dim, random_state=0)
    svd.fit(y[idx])
    return svd, dim


def transform_targets(svd: TruncatedSVD, y: np.ndarray, ids: list[str], id_to_y: dict[str, int]) -> np.ndarray:
    return svd.transform(y[[id_to_y[pid] for pid in ids]]).astype(np.float32)


def eval_model_on_ids(model, loader, device, svd, latent_mean, latent_std, y_true_all, id_to_y) -> tuple[dict[str, np.ndarray], float, float]:
    model.eval()
    pred_latents: dict[str, np.ndarray] = {}
    losses = []
    mse = nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for ids, x, mask, target in loader:
            x = x.to(device)
            mask = mask.to(device)
            target = target.to(device)
            pred = model(x, mask)
            losses.append(float(mse(pred, target).detach().cpu()))
            pred_np = pred.detach().cpu().numpy() * latent_std + latent_mean
            for pid, row in zip(ids, pred_np):
                pred_latents[pid] = row
    pearsons = []
    for pid, lat in pred_latents.items():
        pred_y = svd.inverse_transform(lat.reshape(1, -1))[0]
        true_y = y_true_all[id_to_y[pid]]
        pearsons.append(metric_profile(true_y, pred_y)["pearson_corr"])
    return pred_latents, float(np.nanmean(losses)) if losses else np.nan, float(np.nanmean(pearsons)) if pearsons else np.nan


def train_one_fold(test_id: str, fold_ids: dict[str, list[str]], x_map, y, id_to_y, lengths, meta, args, device) -> dict[str, Any]:
    seed_all(args.seed + abs(hash(test_id)) % 100000)
    available = fold_ids["available"]
    train_ids, val_ids = split_train_val(available, meta, args.seed + len(test_id), args.val_fraction)
    svd, latent_dim = fit_fold_svd(y, available, id_to_y, args.latent_dim)
    latent_available = {pid: None for pid in available}
    train_lat = transform_targets(svd, y, train_ids, id_to_y)
    val_lat = transform_targets(svd, y, val_ids, id_to_y)
    latent_mean = train_lat.mean(axis=0, keepdims=True).astype(np.float32)
    latent_std = train_lat.std(axis=0, keepdims=True).astype(np.float32)
    latent_std[latent_std < 1e-6] = 1.0
    train_lat_z = (train_lat - latent_mean) / latent_std
    val_lat_z = (val_lat - latent_mean) / latent_std

    # Temporary target arrays keep the same row order as ids for Dataset lookup.
    z_y = np.zeros((len(id_to_y), latent_dim), dtype=np.float32)
    for pid, lat in zip(train_ids, train_lat_z):
        z_y[id_to_y[pid]] = lat
    for pid, lat in zip(val_ids, val_lat_z):
        z_y[id_to_y[pid]] = lat
    test_lat = transform_targets(svd, y, [test_id], id_to_y)
    z_y[id_to_y[test_id]] = (test_lat[0] - latent_mean[0]) / latent_std[0]

    train_ds = RbdEmbeddingDataset(train_ids, x_map, z_y, id_to_y, args.max_len)
    val_ds = RbdEmbeddingDataset(val_ids, x_map, z_y, id_to_y, args.max_len)
    test_ds = RbdEmbeddingDataset([test_id], x_map, z_y, id_to_y, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)

    input_dim = next(iter(x_map.values())).shape[1]
    model = PerResidueCnn(input_dim, args.hidden_dim, latent_dim, args.kernel_size, args.num_blocks, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    best_state = None
    best_val = -np.inf
    best_epoch = 0
    patience_left = args.patience
    curve = []
    status = "ok"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for _, x, mask, target in train_loader:
            x = x.to(device)
            mask = mask.to(device)
            target = target.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = mse(pred, target)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        _, val_loss, val_metric = eval_model_on_ids(model, val_loader, device, svd, latent_mean, latent_std, y, id_to_y)
        curve.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)) if train_losses else np.nan, "val_loss": val_loss, "val_metric": val_metric})
        score = val_metric if np.isfinite(val_metric) else -val_loss
        if score > best_val:
            best_val = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        status = "failed_no_best_state"
    pred_latents, _, _ = eval_model_on_ids(model, test_loader, device, svd, latent_mean, latent_std, y, id_to_y)
    pred_y = svd.inverse_transform(pred_latents[test_id].reshape(1, -1))[0]
    metrics = metric_profile(y[id_to_y[test_id]], pred_y)
    if args.save_checkpoints and status == "ok":
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "test_id": test_id, "latent_dim": latent_dim}, ckpt_dir / f"cnn_fold_{test_id}.pt")
    return {
        "test_id": test_id,
        "metrics": metrics,
        "status": status,
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "rbd_length": int(lengths[test_id]),
        "curve": curve,
    }


def load_baseline(args, ids: list[str]) -> pd.DataFrame:
    path = resolve_path(args.baseline_exp3_tsv, ["exp3_plant_leave_one_out_metrics.tsv"], required=False)
    if path is not None:
        b = pd.read_csv(path, sep="\t")
        if "protein_id" in b.columns:
            return b[b["protein_id"].astype(str).isin(ids)].copy()
    if RBPTraceFirstLayer is None:
        raise RuntimeError("Cannot recompute baseline: RBPTraceFirstLayer import failed and baseline TSV missing")
    emb_path = resolve_path(args.pooled_embedding_npz, ["rnacompete_domain_merged_esmc_embeddings.npz"])
    motif_path = resolve_path(args.motif_npz, ["motif_profiles.npz"])
    emb = np.load(emb_path, allow_pickle=True)
    x_ids = np.asarray(emb[first_key(emb, ["protein_ids", "ids", "names"])]).astype(str)
    x = np.asarray(emb[first_key(emb, ["embeddings", "X", "embedding"])], dtype=np.float32)
    motif = np.load(motif_path, allow_pickle=True)
    y_ids = np.asarray(motif[first_key(motif, ["profile_ids", "protein_ids", "ids", "names"])]).astype(str)
    y = np.asarray(motif[first_key(motif, ["zscores", "scores", "Y", "profiles"])], dtype=np.float32)
    y_idx = {pid: i for i, pid in enumerate(y_ids)}
    common = [pid for pid in x_ids if pid in y_idx]
    x_idx = {pid: i for i, pid in enumerate(x_ids)}
    rows = []
    for test_id in ids:
        train_ids = [pid for pid in common if pid != test_id]
        model = RBPTraceFirstLayer(num_eigenvector=122, threshold=0.01, std=0.2)
        model.fit(x[[x_idx[p] for p in train_ids]], y[[y_idx[p] for p in train_ids]])
        pred, dist, _ = model.predict_protein(x[[x_idx[test_id]]])
        m = metric_profile(y[y_idx[test_id]], pred[0])
        rows.append({"protein_id": test_id, "nearest_train_distance": float(dist[0]), **m})
    return pd.DataFrame(rows)


def add_bins(df: pd.DataFrame, col: str, out_col: str, labels: list[str]) -> pd.DataFrame:
    vals = pd.to_numeric(df[col], errors="coerce")
    finite = vals[np.isfinite(vals)]
    if len(finite) < 3:
        df[out_col] = "unknown"
        return df
    q1, q2 = np.nanquantile(finite, [1 / 3, 2 / 3])
    df[out_col] = np.where(vals <= q1, labels[0], np.where(vals <= q2, labels[1], labels[2]))
    df.loc[~np.isfinite(vals), out_col] = "unknown"
    return df


def paired_stats(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    b = pd.to_numeric(df[f"baseline_{metric}"], errors="coerce")
    c = pd.to_numeric(df[f"cnn_{metric}"], errors="coerce")
    mask = np.isfinite(b) & np.isfinite(c)
    delta = c[mask] - b[mask]
    pval = np.nan
    if wilcoxon is not None and len(delta) >= 2 and np.nanstd(delta) > 0:
        try:
            pval = float(wilcoxon(c[mask], b[mask]).pvalue)
        except Exception:
            pval = np.nan
    return {
        "metric": metric,
        "n_valid": int(mask.sum()),
        "baseline_mean": float(b[mask].mean()) if mask.any() else np.nan,
        "cnn_mean": float(c[mask].mean()) if mask.any() else np.nan,
        "baseline_median": float(b[mask].median()) if mask.any() else np.nan,
        "cnn_median": float(c[mask].median()) if mask.any() else np.nan,
        "mean_delta": float(delta.mean()) if len(delta) else np.nan,
        "median_delta": float(delta.median()) if len(delta) else np.nan,
        "positive_delta_fraction": float((delta > 0).mean()) if len(delta) else np.nan,
        "wilcoxon_p": pval,
    }


def first_present(row: dict[str, Any], keys: list[str], default=np.nan):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def normalize_overlap(value, k: int):
    try:
        v = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(v):
        return np.nan
    return v / float(k) if v > 1.0 else v


def summarize_and_report(metrics: pd.DataFrame, curves: dict[str, list[dict[str, float]]], args, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [paired_stats(metrics, "pearson"), paired_stats(metrics, "spearman"), paired_stats(metrics, "top20_overlap")]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "cnn_vs_rbp_trace_summary.tsv", sep="\t", index=False)
    pearson_stats = summary_rows[0]
    top20_stats = summary_rows[2]
    use_cnn = (
        pearson_stats["mean_delta"] >= 0.05
        and pearson_stats["median_delta"] >= 0.05
        and pearson_stats["positive_delta_fraction"] >= 0.60
        and (np.isfinite(pearson_stats["wilcoxon_p"]) and pearson_stats["wilcoxon_p"] < 0.05)
        and top20_stats["mean_delta"] >= -1e-9
    )
    if use_cnn:
        status = "use_cnn_as_final_model"
        reason = "CNN meets mean/median Pearson, positive-delta, Wilcoxon, and top20 criteria."
    elif pearson_stats["mean_delta"] > 0 and pearson_stats["positive_delta_fraction"] >= 0.5:
        status = "inconclusive"
        reason = "CNN has some positive signal but does not meet stable replacement criteria."
    else:
        status = "keep_rbp_trace_as_final_model"
        reason = "CNN does not provide stable improvement over original RBPTrace baseline."
    stats = {
        "paired_metric_stats": summary_rows,
        "cnn_model_status": status,
        "reason": reason,
    }
    with (out_dir / "cnn_vs_rbp_trace_stats.json").open("w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    judgment = {
        "cnn_model_status": status,
        "reason": reason,
        "mean_delta_pearson": pearson_stats["mean_delta"],
        "median_delta_pearson": pearson_stats["median_delta"],
        "positive_delta_fraction": pearson_stats["positive_delta_fraction"],
        "wilcoxon_p_pearson": pearson_stats["wilcoxon_p"],
        "baseline_mean_pearson": pearson_stats["baseline_mean"],
        "cnn_mean_pearson": pearson_stats["cnn_mean"],
        "baseline_median_pearson": pearson_stats["baseline_median"],
        "cnn_median_pearson": pearson_stats["cnn_median"],
        "recommendation": "Use CNN only if status is use_cnn_as_final_model; otherwise keep RBPTraceFirstLayer + confidence.",
    }
    with (out_dir / "final_cnn_model_judgment.json").open("w") as handle:
        json.dump(judgment, handle, indent=2, sort_keys=True)

    strat_rows = []
    for group_col in ["domain_family", "domain_architecture", "rbd_length_bin", "nearest_distance_bin", "nearest_train_same_architecture"]:
        if group_col not in metrics.columns:
            continue
        for group, g in metrics.groupby(group_col, dropna=False):
            strat_rows.append({
                "stratification": group_col,
                "group": str(group),
                "n": int(len(g)),
                "baseline_mean_pearson": float(pd.to_numeric(g["baseline_pearson"], errors="coerce").mean()),
                "cnn_mean_pearson": float(pd.to_numeric(g["cnn_pearson"], errors="coerce").mean()),
                "mean_delta_pearson": float(pd.to_numeric(g["delta_pearson"], errors="coerce").mean()),
                "baseline_mean_top20_overlap": float(pd.to_numeric(g["baseline_top20_overlap"], errors="coerce").mean()),
                "cnn_mean_top20_overlap": float(pd.to_numeric(g["cnn_top20_overlap"], errors="coerce").mean()),
            })
    pd.DataFrame(strat_rows).to_csv(out_dir / "stratified_cnn_vs_baseline_summary.tsv", sep="\t", index=False)

    def boxplot(metric: str, path: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(5, 4))
        data = [pd.to_numeric(metrics[f"baseline_{metric}"], errors="coerce").dropna(), pd.to_numeric(metrics[f"cnn_{metric}"], errors="coerce").dropna()]
        ax.boxplot(data, tick_labels=["RBPTrace", "CNN"], showfliers=False)
        ax.set_ylabel(ylabel)
        fig.tight_layout(); fig.savefig(fig_dir / path, dpi=200); plt.close(fig)
    boxplot("pearson", "cnn_vs_baseline_pearson_boxplot.png", "Pearson")
    boxplot("spearman", "cnn_vs_baseline_spearman_boxplot.png", "Spearman")
    boxplot("top20_overlap", "cnn_vs_baseline_top20_overlap_boxplot.png", "Top20 overlap")

    fig, ax = plt.subplots(figsize=(max(7, len(metrics) * 0.18), 4))
    ax.bar(metrics["protein_id"], metrics["delta_pearson"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("CNN - baseline Pearson")
    ax.set_xticklabels(metrics["protein_id"], rotation=90, fontsize=6)
    fig.tight_layout(); fig.savefig(fig_dir / "cnn_delta_pearson_per_rbp.png", dpi=200); plt.close(fig)

    for col, path in [("domain_architecture", "cnn_delta_by_domain_architecture.png"), ("nearest_distance_bin", "cnn_delta_by_nearest_distance_bin.png")]:
        if col in metrics.columns:
            g = metrics.groupby(col)["delta_pearson"].mean().sort_values()
            if len(g) > 10:
                top = set(metrics[col].value_counts().head(10).index)
                tmp = metrics.assign(_group=metrics[col].map(lambda v: v if v in top else "Other"))
                g = tmp.groupby("_group")["delta_pearson"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(max(6, len(g) * 0.7), 4))
            ax.bar(g.index.astype(str), g.values)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_ylabel("Mean delta Pearson")
            ax.set_xticklabels(g.index.astype(str), rotation=45, ha="right")
            fig.tight_layout(); fig.savefig(fig_dir / path, dpi=200); plt.close(fig)

    if curves:
        first_id = next(iter(curves))
        cdf = pd.DataFrame(curves[first_id])
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(cdf["epoch"], cdf["train_loss"], label="train loss")
        ax1.plot(cdf["epoch"], cdf["val_loss"], label="val loss")
        ax2 = ax1.twinx()
        ax2.plot(cdf["epoch"], cdf["val_metric"], color="black", linestyle="--", label="val Pearson")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE"); ax2.set_ylabel("Val Pearson")
        ax1.legend(loc="upper left", frameon=False); ax2.legend(loc="upper right", frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "cnn_training_curve_example.png", dpi=200); plt.close(fig)

    command = f"""nice -n 10 python code/per_residue_cnn_first_layer/train_cnn_motif_latent_plant_loo.py \\
  --per-residue-h5 {args.per_residue_h5} \\
  --motif-npz {args.motif_npz} \\
  --plant-label-tsv {args.plant_label_tsv} \\
  --domain-annotation-tsv {args.domain_annotation_tsv} \\
  --baseline-exp3-tsv {args.baseline_exp3_tsv} \\
  --output-dir {args.output_dir} \\
  --loo-scope {args.loo_scope} \\
  --device {args.device} \\
  --gpu-memory-fraction {args.gpu_memory_fraction} \\
  --batch-size {args.batch_size} \\
  --epochs {args.epochs} \\
  --patience {args.patience} \\
  --latent-dim {args.latent_dim} \\
  --hidden-dim {args.hidden_dim} \\
  --torch-num-threads {args.torch_num_threads}"""
    report = f"""# Per-Residue RBD CNN First-Layer Exploratory Validation

## Run Command

```bash
{command}
```

## Purpose

This exploratory model tests whether per-residue RBD ESMC embeddings plus a lightweight 1D-CNN can outperform the current pooled RBD embedding + original `RBPTraceFirstLayer` in plant RBP leave-one-out validation. Pooled embeddings do not retain residue position, while a CNN can learn local patterns along RBD residue positions.

## Data Overview

- Compared plant LOO folds: {len(metrics)}
- Motif profile dimension: 16,384 7-mer scores
- Per-residue embedding dimension: {int(metrics.get("embedding_dim", pd.Series([np.nan])).dropna().iloc[0]) if "embedding_dim" in metrics and metrics["embedding_dim"].notna().any() else "NA"}
- RBD length median: {float(metrics["rbd_length"].median()) if len(metrics) else "NA"}

## Model

- Baseline: original `RBPTraceFirstLayer` with pooled RBD ESMC embedding.
- CNN: per-residue RBD ESMC embedding, 1D convolution along residue positions, masked mean/max pooling, and an MLP motif-latent head.
- Motif profiles are first projected to a fold-specific TruncatedSVD latent space fitted only on the fold training set. The CNN predicts motif latent coordinates, then inverse-transforms to 16,384-dimensional motif profiles for evaluation.

## Plant LOO Results

- Baseline mean Pearson: {pearson_stats["baseline_mean"]}
- CNN mean Pearson: {pearson_stats["cnn_mean"]}
- Mean delta Pearson: {pearson_stats["mean_delta"]}
- Median delta Pearson: {pearson_stats["median_delta"]}
- Positive delta fraction: {pearson_stats["positive_delta_fraction"]}
- Wilcoxon p-value Pearson: {pearson_stats["wilcoxon_p"]}

See `cnn_vs_rbp_trace_summary.tsv` and `stratified_cnn_vs_baseline_summary.tsv` for Spearman, top20 overlap, and stratified results.

## Final Judgment

`cnn_model_status = {status}`

Reason: {reason}

## Recommendation

If CNN does not satisfy the replacement criteria, keep the original `RBPTraceFirstLayer` with architecture/distance confidence stratification. CNN can remain an exploratory supplemental model or future ensemble candidate.
"""
    (out_dir / "PER_RESIDUE_CNN_FIRST_LAYER_REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plant LOO validation for per-residue RBD CNN motif-latent model.")
    p.add_argument("--per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--per-residue-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--plant-label-tsv", default="results/species_transfer_analysis/plant_nonplant_label_check.tsv")
    p.add_argument("--domain-annotation-tsv", default="results/embedding_domain_audit/domain_annotation_check.tsv")
    p.add_argument("--baseline-exp3-tsv", default="results/species_transfer_analysis/exp3_plant_leave_one_out_metrics.tsv")
    p.add_argument("--plant-loo-domain-metrics-tsv", default="results/embedding_domain_audit/exp3_plant_loo_domain_neighbor_metrics.tsv")
    p.add_argument("--pooled-embedding-npz", default="data/embeddings/rnacompete_domain_merged_esmc_embeddings.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer")
    p.add_argument("--loo-scope", choices=["plant", "all"], default="plant", help="Run LOO on plant RBPs only or all aligned RBPs.")
    p.add_argument("--test-ids-tsv", default=None, help="Optional TSV with a protein_id column. If provided, LOO is run only for these IDs.")
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--gpu-memory-fraction", type=float, default=0.25, help="Max fraction of visible GPU memory this process may reserve when using CUDA. Set <=0 to disable.")
    p.add_argument("--fallback-cpu-on-oom", action="store_true", help="If a CUDA OOM occurs in a fold, retry that fold on CPU instead of aborting.")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--seed", type=int, default=20260615)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick_test:
        args.output_dir = args.output_dir if args.output_dir.endswith("quick_test") else str(Path(args.output_dir) / "quick_test")
        args.epochs = min(args.epochs, 5)
        args.patience = min(args.patience, 3)
        args.latent_dim = min(args.latent_dim, 10)
        args.hidden_dim = min(args.hidden_dim, 32)
        args.max_folds = args.max_folds or 3
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    args.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_fraction and args.gpu_memory_fraction > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(float(args.gpu_memory_fraction), device=0)
            log(f"Set CUDA per-process memory fraction to {args.gpu_memory_fraction}")
        except Exception as exc:
            log(f"Could not set CUDA memory fraction: {type(exc).__name__}: {exc}")
    h5_path = resolve_path(args.per_residue_h5, ["rnacompete_rbd_per_residue_esmc.h5"], required=False)
    if h5_path is None:
        raise SystemExit("Missing per-residue H5. Run extract_per_residue_rbd_esmc_embeddings.py first; CNN input must be L x D per-residue embeddings, not pooled 1152D vectors.")
    manifest_path = resolve_path(args.per_residue_manifest, ["rnacompete_rbd_per_residue_manifest.tsv"], required=False)
    x_map, lengths = load_h5_features(h5_path, manifest_path)
    log(f"Loaded per-residue embeddings: {len(x_map)} proteins from {h5_path}")
    tables = load_tables(args)
    y = tables["motif_scores"]
    id_to_y = tables["id_to_y"]
    meta = tables["meta"]
    common = [pid for pid in tables["motif_ids"] if pid in x_map]
    plant_ids = [pid for pid in common if bool(meta.set_index("protein_id").loc[pid, "is_plant"])]
    nonplant_ids = [pid for pid in common if pid not in set(plant_ids)]
    if args.quick_test:
        plant_ids = plant_ids[: min(3, len(plant_ids))]
        keep_nonplant = nonplant_ids[:30]
        keep_plants_for_train = [pid for pid in common if pid in set(plant_ids)] + [pid for pid in common if bool(meta.set_index("protein_id").loc[pid, "is_plant"])]
        allowed = set(keep_nonplant + keep_plants_for_train)
        common = [pid for pid in common if pid in allowed]
        nonplant_ids = [pid for pid in common if pid not in set(plant_ids)]
    if args.test_ids_tsv:
        test_df = pd.read_csv(resolve_path(args.test_ids_tsv, [Path(args.test_ids_tsv).name]), sep="\t")
        if "protein_id" not in test_df.columns:
            raise SystemExit("--test-ids-tsv must contain a protein_id column")
        requested_ids = test_df["protein_id"].astype(str).tolist()
        common_set = set(common)
        test_ids = [pid for pid in requested_ids if pid in common_set]
    else:
        test_ids = common if args.loo_scope == "all" else plant_ids
    if args.max_folds:
        test_ids = test_ids[: args.max_folds]
    if not test_ids:
        raise SystemExit("No requested LOO IDs overlap per-residue H5 and motif profiles.")
    baseline = load_baseline(args, test_ids)
    domain_loo_path = resolve_path(args.plant_loo_domain_metrics_tsv, ["exp3_plant_loo_domain_neighbor_metrics.tsv"], required=False)
    domain_loo = pd.read_csv(domain_loo_path, sep="\t") if domain_loo_path else pd.DataFrame(columns=["protein_id"])
    meta_i = meta.set_index("protein_id")
    rows = []
    curves: dict[str, list[dict[str, float]]] = {}
    for fold_idx, test_id in enumerate(test_ids, start=1):
        available = [pid for pid in common if pid != test_id]
        log(f"fold {fold_idx}/{len(test_ids)} test={test_id} train_available={len(available)}")
        try:
            res = train_one_fold(test_id, {"available": available}, x_map, y, id_to_y, lengths, meta, args, device)
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower() and args.fallback_cpu_on_oom:
                log(f"CUDA OOM on fold {test_id}; clearing cache and retrying on CPU")
                torch.cuda.empty_cache()
                gc.collect()
                res = train_one_fold(test_id, {"available": available}, x_map, y, id_to_y, lengths, meta, args, torch.device("cpu"))
                res["status"] = res.get("status", "ok") + ";cuda_oom_retried_cpu"
            else:
                raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        curves[test_id] = res["curve"]
        b = baseline[baseline["protein_id"].astype(str) == test_id]
        b_row = b.iloc[0].to_dict() if len(b) else {}
        d = domain_loo[domain_loo["protein_id"].astype(str) == test_id]
        d_row = d.iloc[0].to_dict() if len(d) else {}
        m = res["metrics"]
        row = {
            "protein_id": test_id,
            "species": meta_i.loc[test_id, "species"] if test_id in meta_i.index else "Unknown",
            "domain_family": meta_i.loc[test_id, "domain_family"] if test_id in meta_i.index else "Unknown",
            "domain_architecture": meta_i.loc[test_id, "domain_architecture"] if test_id in meta_i.index else "Unknown",
            "baseline_pearson": first_present(b_row, ["pearson_corr", "pearson"]),
            "cnn_pearson": m.get("pearson_corr", np.nan),
            "baseline_spearman": first_present(b_row, ["spearman_corr", "spearman"]),
            "cnn_spearman": m.get("spearman_corr", np.nan),
            "baseline_top10_overlap": normalize_overlap(first_present(b_row, ["top10_overlap"]), 10),
            "cnn_top10_overlap": m.get("top10_overlap", np.nan),
            "baseline_top20_overlap": normalize_overlap(first_present(b_row, ["top20_overlap"]), 20),
            "cnn_top20_overlap": m.get("top20_overlap", np.nan),
            "baseline_top50_overlap": normalize_overlap(first_present(b_row, ["top50_overlap"]), 50),
            "cnn_top50_overlap": m.get("top50_overlap", np.nan),
            "cnn_prediction_status": res["status"],
            "cnn_best_epoch": res["best_epoch"],
            "cnn_val_metric": res["best_val"],
            "rbd_length": res["rbd_length"],
            "embedding_dim": int(next(iter(x_map.values())).shape[1]),
            "nearest_train_distance": first_present(d_row, ["nearest_train_distance"], first_present(b_row, ["nearest_train_distance", "min_neighbor_cosine_distance"])),
            "nearest_train_same_family": d_row.get("nearest_train_same_family", np.nan),
            "nearest_train_same_architecture": d_row.get("nearest_train_same_architecture", np.nan),
        }
        for metric in ["pearson", "spearman", "top10_overlap", "top20_overlap", "top50_overlap"]:
            row[f"delta_{metric}"] = row.get(f"cnn_{metric}", np.nan) - row.get(f"baseline_{metric}", np.nan)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "plant_loo_cnn_vs_rbp_trace_metrics.tsv", sep="\t", index=False)
    metrics = pd.DataFrame(rows)
    metrics = add_bins(metrics, "rbd_length", "rbd_length_bin", ["short", "medium", "long"])
    metrics = add_bins(metrics, "nearest_train_distance", "nearest_distance_bin", ["close", "medium", "distant"])
    metrics.to_csv(out_dir / "plant_loo_cnn_vs_rbp_trace_metrics.tsv", sep="\t", index=False)
    summarize_and_report(metrics, curves, args, out_dir)
    log("Metrics: " + str(out_dir / "plant_loo_cnn_vs_rbp_trace_metrics.tsv"))
    log("Report: " + str(out_dir / "PER_RESIDUE_CNN_FIRST_LAYER_REPORT.md"))
    log("Judgment: " + str(out_dir / "final_cnn_model_judgment.json"))


if __name__ == "__main__":
    main()
