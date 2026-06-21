#!/usr/bin/env python3
"""Optimization grid from the C latent-CNN motif baseline.

Runs label-free ablations over:
  A: model_type
  B: protein_input (B0 rbd_only currently; B2/B3 are skipped if inputs unavailable)
  C: postprocess

Query predictions are evaluation-only and never used for checkpoint selection.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
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
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))

from cnn_model_utils import PerResidueCnn, first_key, load_h5_features, resolve_path, setup_threads  # noqa: E402


def log(msg: str) -> None:
    print(f"[optimization-grid] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_motif(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    z = np.load(path, allow_pickle=True)
    ids = np.asarray(z[first_key(z, ["profile_ids", "protein_ids", "ids", "names"])]).astype(str)
    profiles = np.asarray(z[first_key(z, ["zscores", "scores", "Y", "profiles"])], dtype=np.float32)
    kmers = np.asarray(z["kmers"]).astype(str)
    return ids, profiles, kmers, {pid: i for i, pid in enumerate(ids)}


def short_id(pid: str) -> str:
    return pid.split("|", 1)[0]


def latent_to_profile_torch(pred_latent, svd_components, scaler_mean, scaler_scale):
    return (pred_latent @ svd_components) * scaler_scale + scaler_mean


class ProteinProfileDataset(Dataset):
    def __init__(self, ids, x_map, profile_by_id, max_len=None):
        self.ids = list(ids)
        self.x_map = x_map
        self.profile_by_id = profile_by_id
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        x = self.x_map[pid]
        if self.max_len is not None and x.shape[0] > self.max_len:
            x = x[: self.max_len]
        return pid, torch.from_numpy(x.astype(np.float32)), torch.from_numpy(self.profile_by_id[pid].astype(np.float32))


def collate_protein_batch(batch):
    ids, xs, profiles = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    dim = xs[0].shape[1]
    padded = torch.zeros((len(xs), max_len, dim), dtype=torch.float32)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True
    return list(ids), padded, mask, torch.stack(profiles)


class PairwiseKmerModel(nn.Module):
    def __init__(self, input_dim, protein_dim, kmer_dim, hidden_dim, dropout, kmer_encoder="embedding", n_kmers=16384):
        super().__init__()
        self.protein_encoder = PerResidueCnn(input_dim, hidden_dim, protein_dim, kernel_size=5, num_blocks=2, dropout=dropout)
        self.kmer_encoder_type = kmer_encoder
        self.n_kmers = n_kmers
        self.protein_dim = protein_dim
        if kmer_encoder == "embedding":
            self.kmer_embedding = nn.Embedding(n_kmers, protein_dim)
        elif kmer_encoder == "onehot_cnn":
            self.kmer_embedding = nn.Sequential(
                nn.Conv1d(4, kmer_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Linear(kmer_dim, protein_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(kmer_encoder)
        self.mlp = nn.Sequential(
            nn.Linear(protein_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_protein(self, x, mask):
        return self.protein_encoder(x, mask)

    def encode_kmers(self, kmer_idx, kmer_onehot_table=None):
        if self.kmer_encoder_type == "embedding":
            return self.kmer_embedding(kmer_idx)
        flat = kmer_idx.reshape(-1)
        onehot = kmer_onehot_table[flat].transpose(1, 2)
        enc = self.kmer_embedding(onehot)
        return enc.reshape(*kmer_idx.shape, -1)

    def score_with_repr(self, protein_repr, kmer_idx, kmer_onehot_table=None):
        kmer_repr = self.encode_kmers(kmer_idx, kmer_onehot_table)
        p = protein_repr.unsqueeze(1).expand(-1, kmer_idx.shape[1], -1)
        feat = torch.cat([p, kmer_repr, p * kmer_repr], dim=-1)
        return self.mlp(feat).squeeze(-1)


def make_kmer_onehot(kmers: np.ndarray) -> np.ndarray:
    code = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    arr = np.zeros((len(kmers), 7, 4), dtype=np.float32)
    for i, k in enumerate(kmers):
        for j, ch in enumerate(k):
            arr[i, j, code[ch]] = 1.0
    return arr


def split_ids_from_tsv(split_tsv, x_map, id2y):
    df = pd.read_csv(split_tsv, sep="\t")
    out = {}
    for split in ["train", "val", "test"]:
        ids = df.loc[df["split"].astype(str) == split, "protein_id"].astype(str).tolist()
        out[split] = [pid for pid in ids if pid in x_map and pid in id2y]
    return out


def build_profile_maps(ids, profiles, id2y):
    out = {}
    dropped = []
    for pid in ids:
        row = profiles[id2y[pid]].astype(np.float32)
        if np.isfinite(row).all():
            out[pid] = row
        else:
            dropped.append(pid)
    return out, dropped


def kmer_index(kmers):
    return {k: i for i, k in enumerate(kmers)}


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def motif_masks(kmers):
    cuucu_seeds = ["CUUCU", "UCUUC", "CUUCUC", "UCUUCU", "CUUCUU"]
    ucucuc_seeds = ["UCUCUC", "CUCUCU", "UCUCU", "CUCUC"]
    cuucu = []
    ucucuc = []
    urich = []
    for k in kmers:
        urich.append(k.count("U") >= 5 or "UUUUU" in k)
        cu_hit = False
        uc_hit = False
        for seed in cuucu_seeds:
            L = len(seed)
            for s in range(0, 7 - L + 1):
                if hamming(k[s : s + L], seed) <= 1:
                    cu_hit = True
                    break
            if cu_hit:
                break
        for seed in ucucuc_seeds:
            L = len(seed)
            for s in range(0, 7 - L + 1):
                if hamming(k[s : s + L], seed) <= 1:
                    uc_hit = True
                    break
            if uc_hit:
                break
        cuucu.append(cu_hit)
        ucucuc.append(uc_hit)
    return {
        "CUUCU_like": np.asarray(cuucu, dtype=bool),
        "UCUCUC_like": np.asarray(ucucuc, dtype=bool),
        "U_rich": np.asarray(urich, dtype=bool),
    }


def neighbor_indices(kmers, distance):
    idx = kmer_index(kmers)
    alphabet = "ACGU"
    neigh = []
    for k in kmers:
        vals = set()
        if distance >= 1:
            for i in range(7):
                for a in alphabet:
                    if a != k[i]:
                        vals.add(idx[k[:i] + a + k[i + 1 :]])
        if distance >= 2:
            for i, j in itertools.combinations(range(7), 2):
                for a in alphabet:
                    if a == k[i]:
                        continue
                    for b in alphabet:
                        if b == k[j]:
                            continue
                        s = list(k)
                        s[i] = a
                        s[j] = b
                        vals.add(idx["".join(s)])
        neigh.append(np.asarray(sorted(vals), dtype=np.int64))
    return neigh


def apply_postprocess(scores, postprocess, h1=None, h2=None):
    scores = np.asarray(scores, dtype=np.float32)
    if postprocess == "C0":
        return scores
    if postprocess == "C1":
        std = float(scores.std())
        return (scores - scores.mean()) / (std if std > 1e-8 else 1.0)
    if postprocess == "C2":
        out = np.empty_like(scores)
        for i, ns in enumerate(h1):
            out[i] = 0.7 * scores[i] + 0.3 * float(scores[ns].mean())
        return out
    if postprocess == "C3":
        out = np.empty_like(scores)
        for i in range(len(scores)):
            out[i] = scores[i] + 0.1 * float(scores[h1[i]].mean()) + 0.1 * float(scores[h2[i]].mean())
        return out
    raise ValueError(postprocess)


def profile_metrics(pred_profiles, true_profiles):
    pearsons, spearmans, top20, top50, top1_rank, top5_best = [], [], [], [], [], []
    for pred, true in zip(pred_profiles, true_profiles):
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        pc = pred - pred.mean()
        tc = true - true.mean()
        den = np.sqrt(np.sum(pc * pc) * np.sum(tc * tc))
        pearsons.append(np.sum(pc * tc) / den if den > 0 else np.nan)
        pr = rankdata(pred)
        tr = rankdata(true)
        prc = pr - pr.mean()
        trc = tr - tr.mean()
        rden = np.sqrt(np.sum(prc * prc) * np.sum(trc * trc))
        spearmans.append(np.sum(prc * trc) / rden if rden > 0 else np.nan)
        to = np.argsort(-true)
        po = np.argsort(-pred)
        inv = np.empty_like(po)
        inv[po] = np.arange(1, len(po) + 1)
        top20.append(len(set(to[:20]) & set(po[:20])) / 20.0)
        top50.append(len(set(to[:50]) & set(po[:50])) / 50.0)
        top1_rank.append(float(inv[to[0]]))
        top5_best.append(float(np.min(inv[to[:5]])))
    return {
        "val_profile_pearson_mean": float(np.nanmean(pearsons)),
        "val_profile_spearman_mean": float(np.nanmean(spearmans)),
        "val_top20_recovery": float(np.nanmean(top20)),
        "val_top50_recovery": float(np.nanmean(top50)),
        "val_true_top1_rank_median": float(np.nanmedian(top1_rank)),
        "val_true_top5_best_rank_median": float(np.nanmedian(top5_best)),
    }


def combined_score(metrics):
    return (
        float(metrics["val_profile_pearson_mean"])
        * float(metrics["val_top20_recovery"])
        * float(metrics["val_top50_recovery"])
    )


def query_rows(config, query_profiles, kmers, masks):
    rows = []
    for qid, scores in query_profiles.items():
        order = np.argsort(-scores)
        inv = np.empty_like(order)
        inv[order] = np.arange(1, len(order) + 1)
        row = {
            **config,
            "query_id": qid,
            "short_id": short_id(qid),
            "top1_kmer": kmers[order[0]],
            "top5_kmers": ",".join(kmers[order[:5]]),
            "top10_kmers": ",".join(kmers[order[:10]]),
            "top20_kmers": ",".join(kmers[order[:20]]),
        }
        for name, mask in masks.items():
            idx = np.where(mask)[0]
            best_local = idx[np.argmin(inv[idx])]
            prefix = name.replace("_like", "").replace("_rich", "rich")
            row[f"best_{name}_rank"] = int(inv[best_local])
            row[f"best_{name}_kmer"] = kmers[best_local]
            row[f"best_{name}_score"] = float(scores[best_local])
            row[f"contains_{name}_top20"] = bool(np.any(mask[order[:20]]))
        rows.append(row)
    return rows


def read_matrix_gz(path):
    out = {}
    with gzip.open(path, "rt") as handle:
        kmers = np.asarray(handle.readline().rstrip("\n").split("\t")[1:]).astype(str)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            out[parts[0]] = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
    return kmers, out


def predict_latent_checkpoint(ckpt_path, ids, x_map, batch_size, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = PerResidueCnn(cfg["input_dim"], cfg["hidden_dim"], cfg["latent_dim"], cfg["kernel_size"], cfg["num_blocks"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    svd = torch.tensor(np.asarray(ckpt["svd_components"], dtype=np.float32), device=device)
    mean = torch.tensor(np.asarray(ckpt["scaler_mean"], dtype=np.float32), device=device)
    scale = torch.tensor(np.asarray(ckpt["scaler_scale"], dtype=np.float32), device=device)
    dummy_profiles = {pid: np.zeros((cfg["latent_dim"],), dtype=np.float32) for pid in ids}
    loader = DataLoader(ProteinProfileDataset(ids, x_map, dummy_profiles), batch_size=batch_size, shuffle=False, collate_fn=collate_protein_batch)
    out = {}
    with torch.no_grad():
        for batch_ids, x, mask, _ in loader:
            pred_lat = model(x.to(device), mask.to(device))
            prof = latent_to_profile_torch(pred_lat, svd, mean, scale).cpu().numpy()
            for pid, row in zip(batch_ids, prof):
                out[pid] = row.astype(np.float32)
    return out


def sample_regression_indices(profile, rng, n, mode):
    n_kmers = len(profile)
    if mode == "A1":
        return rng.integers(0, n_kmers, size=n, endpoint=False)
    order = np.argsort(-profile)
    pos = order[: min(64, n // 4)]
    mid_start = max(0, n_kmers // 2 - 500)
    mid_pool = order[mid_start : min(n_kmers, mid_start + 1000)]
    low_pool = order[-1000:]
    random_pool = np.arange(n_kmers)
    chunks = [pos]
    for pool in [mid_pool, low_pool, random_pool]:
        need = max(0, n - sum(len(c) for c in chunks))
        take = min(max(1, n // 4), need)
        if take:
            chunks.append(rng.choice(pool, size=take, replace=len(pool) < take))
    cur = np.concatenate(chunks)
    if len(cur) < n:
        cur = np.concatenate([cur, rng.choice(random_pool, size=n - len(cur), replace=False)])
    return cur[:n]


def train_pairwise(config_id, model_type, train_ids, val_ids, x_map, profile_by_id, kmers, args, device):
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir) / config_id
    out_dir.mkdir(parents=True, exist_ok=True)
    kmer_onehot = torch.tensor(make_kmer_onehot(kmers), device=device) if model_type == "A3" else None
    model = PairwiseKmerModel(
        input_dim=next(iter(x_map.values())).shape[1],
        protein_dim=args.protein_dim,
        kmer_dim=args.kmer_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        kmer_encoder="onehot_cnn" if model_type == "A3" else "embedding",
        n_kmers=len(kmers),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(ProteinProfileDataset(train_ids, x_map, profile_by_id), batch_size=args.batch_size, shuffle=True, collate_fn=collate_protein_batch)
    val_loader = DataLoader(ProteinProfileDataset(val_ids, x_map, profile_by_id), batch_size=args.batch_size, shuffle=False, collate_fn=collate_protein_batch)
    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    best_metrics = None
    wait = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_ids, x, mask, true_profile in train_loader:
            x = x.to(device)
            mask = mask.to(device)
            true_profile = true_profile.to(device)
            idx_np = np.stack([sample_regression_indices(profile_by_id[pid], rng, args.kmer_samples, model_type) for pid in batch_ids])
            idx = torch.tensor(idx_np, device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            protein_repr = model.encode_protein(x, mask)
            pred = model.score_with_repr(protein_repr, idx, kmer_onehot)
            target = true_profile.gather(1, idx)
            loss = loss_fn(pred, target)
            rank_loss_value = torch.tensor(0.0, device=device)
            if model_type in {"A2", "A3"}:
                pos_np = []
                neg_np = []
                for pid in batch_ids:
                    order = np.argsort(-profile_by_id[pid])
                    pos_np.append(order[: args.rank_pos_n])
                    neg_np.append(rng.choice(order[-1000:], size=args.rank_neg_n, replace=args.rank_neg_n > 1000))
                pos = torch.tensor(np.stack(pos_np), device=device, dtype=torch.long)
                neg = torch.tensor(np.stack(neg_np), device=device, dtype=torch.long)
                pos_scores = model.score_with_repr(protein_repr, pos, kmer_onehot).unsqueeze(2)
                neg_scores = model.score_with_repr(protein_repr, neg, kmer_onehot).unsqueeze(1)
                rank_loss_value = torch.relu(args.ranking_margin - pos_scores + neg_scores).mean()
                loss = loss + args.pairwise_ranking_weight * rank_loss_value
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict_pairwise_profiles(model, val_loader, len(kmers), args.profile_chunk, device, kmer_onehot)
        val_true = [profile_by_id[pid] for pid in val_ids]
        metrics = profile_metrics([val_pred[pid] for pid in val_ids], val_true)
        score = combined_score(metrics)
        log_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics, "combined_val_score": score})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = metrics.copy()
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0:
            log(f"{config_id} epoch={epoch} train_loss={np.mean(losses):.4g} val_pearson={metrics['val_profile_pearson_mean']:.4f} val_top20={metrics['val_top20_recovery']:.4f} best_epoch={best_epoch}")
        if wait >= args.patience:
            break
    model.load_state_dict(best_state)
    ckpt = {
        "model_type": model_type,
        "model_config": {
            "input_dim": next(iter(x_map.values())).shape[1],
            "protein_dim": args.protein_dim,
            "kmer_dim": args.kmer_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "kmer_encoder": "onehot_cnn" if model_type == "A3" else "embedding",
            "n_kmers": len(kmers),
        },
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "combined_val_score": best_score,
        "training_args": vars(args),
        "kmer_list": kmers,
    }
    ckpt_path = out_dir / "best_model.pt"
    torch.save(ckpt, ckpt_path)
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.tsv", sep="\t", index=False)
    return ckpt_path, best_epoch, best_metrics, best_score


def load_pairwise_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = PairwiseKmerModel(
        cfg["input_dim"],
        cfg["protein_dim"],
        cfg["kmer_dim"],
        cfg["hidden_dim"],
        cfg["dropout"],
        cfg["kmer_encoder"],
        cfg["n_kmers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    kmer_onehot = torch.tensor(make_kmer_onehot(np.asarray(ckpt["kmer_list"]).astype(str)), device=device) if cfg["kmer_encoder"] == "onehot_cnn" else None
    return model, kmer_onehot


def predict_pairwise_profiles(model, loader, n_kmers, chunk, device, kmer_onehot=None):
    out = {}
    all_idx = torch.arange(n_kmers, device=device, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for ids, x, mask, _ in loader:
            protein_repr = model.encode_protein(x.to(device), mask.to(device))
            parts = []
            for start in range(0, n_kmers, chunk):
                idx = all_idx[start : start + chunk].unsqueeze(0).expand(len(ids), -1)
                parts.append(model.score_with_repr(protein_repr, idx, kmer_onehot).cpu().numpy())
            mat = np.concatenate(parts, axis=1).astype(np.float32)
            for pid, row in zip(ids, mat):
                out[pid] = row
    return out


def evaluate_config_profiles(config, val_profiles_pred, val_ids, profile_by_id, query_profiles_raw, kmers, masks, h1, h2):
    val_pred = []
    query_pred = {}
    for pid in val_ids:
        val_pred.append(apply_postprocess(val_profiles_pred[pid], config["postprocess"], h1, h2))
    metrics = profile_metrics(val_pred, [profile_by_id[pid] for pid in val_ids])
    metrics["combined_val_score"] = combined_score(metrics)
    for qid, scores in query_profiles_raw.items():
        query_pred[qid] = apply_postprocess(scores, config["postprocess"], h1, h2)
    qrows = query_rows(config, query_pred, kmers, masks)
    return metrics, qrows


def baseline_delta_rows(validation_df, query_df):
    base_val = validation_df.loc[validation_df["config_id"] == "C_baseline"].iloc[0]
    base_q = query_df.loc[query_df["config_id"] == "C_baseline"]
    base_by_short = {r["short_id"]: r for _, r in base_q.iterrows()}
    rows = []
    for _, row in validation_df.iterrows():
        cid = row["config_id"]
        q = query_df.loc[query_df["config_id"] == cid]
        q_by_short = {r["short_id"]: r for _, r in q.iterrows()}
        w2_delta = base_by_short["w2"]["best_CUUCU_like_rank"] - q_by_short["w2"]["best_CUUCU_like_rank"]
        at_cu_delta = base_by_short["AtPTBP3"]["best_CUUCU_like_rank"] - q_by_short["AtPTBP3"]["best_CUUCU_like_rank"]
        at_uc_delta = base_by_short["AtPTBP3"]["best_UCUCUC_like_rank"] - q_by_short["AtPTBP3"]["best_UCUCUC_like_rank"]
        val_better = (
            row["combined_val_score"] >= base_val["combined_val_score"]
            and row["val_profile_pearson_mean"] >= base_val["val_profile_pearson_mean"] - 0.01
            and (row["val_top20_recovery"] > base_val["val_top20_recovery"] or row["val_top50_recovery"] > base_val["val_top50_recovery"])
        )
        query_better = (w2_delta > 0) or (at_cu_delta > 0) or (at_uc_delta > 0)
        rows.append({
            "config_id": cid,
            "delta_val_profile_pearson": row["val_profile_pearson_mean"] - base_val["val_profile_pearson_mean"],
            "delta_val_top20_recovery": row["val_top20_recovery"] - base_val["val_top20_recovery"],
            "delta_val_top50_recovery": row["val_top50_recovery"] - base_val["val_top50_recovery"],
            "delta_w2_CUUCU_like_rank": w2_delta,
            "delta_AtPTBP3_CUUCU_like_rank": at_cu_delta,
            "delta_AtPTBP3_UCUCUC_like_rank": at_uc_delta,
            "better_than_C_baseline_validation": bool(val_better),
            "better_than_C_baseline_query": bool(query_better),
            "status": "query_overfit_or_unstable" if query_better and not val_better else "candidate" if val_better and query_better else "not_better",
        })
    return pd.DataFrame(rows)


def direction_summary(validation_df, delta_df):
    rows = []
    merged = validation_df.merge(delta_df, on="config_id")
    for direction, col in [("model_type", "model_type"), ("protein_input", "protein_input"), ("postprocess", "postprocess")]:
        for cand, g in merged.groupby(col):
            qimp = g[["delta_w2_CUUCU_like_rank", "delta_AtPTBP3_CUUCU_like_rank", "delta_AtPTBP3_UCUCUC_like_rank"]].mean().mean()
            rows.append({
                "direction": direction,
                "candidate": cand,
                "mean_combined_val_score": float(g["combined_val_score"].mean()),
                "mean_query_rank_improvement": float(qimp),
                "is_better_than_C_baseline": bool((g["better_than_C_baseline_validation"] & g["better_than_C_baseline_query"]).any()),
                "notes": "",
            })
    return pd.DataFrame(rows)


def final_recommendation(validation_df, delta_df):
    base = validation_df.loc[validation_df["config_id"] == "C_baseline"].iloc[0]
    good = delta_df.loc[delta_df["better_than_C_baseline_validation"] & delta_df["better_than_C_baseline_query"]]
    lines = []
    lines.append(f"C_baseline combined_val_score={base['combined_val_score']:.6g}.")
    if len(good):
        best = validation_df[validation_df["config_id"].isin(good["config_id"])].sort_values("combined_val_score", ascending=False).iloc[0]
        lines.append(f"At least one config improves validation and query ranks. Best candidate by validation: {best['config_id']}.")
    else:
        lines.append("No config simultaneously improves validation and target query motif-family ranks under the strict criteria.")
        lines.append("Keep C_baseline as the final motif preference baseline unless later multi-seed reruns show a stable improvement.")
    unstable = delta_df.loc[delta_df["status"] == "query_overfit_or_unstable", "config_id"].tolist()
    if unstable:
        lines.append("Query-only improvements with weaker validation were marked query_overfit_or_unstable: " + ",".join(unstable[:20]))
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-tsv", default="results/per_residue_cnn_first_layer/query_centered_w1_w6_atptbp3_top20_20260616/fully_random_split_seed20260617/fully_random_split_seed20260617.tsv")
    p.add_argument("--per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--per-residue-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--baseline-checkpoint", default="results/per_residue_cnn_first_layer/profile_ranking_strong_C_fully_random_seed20260617/best_model.pt")
    p.add_argument("--query-h5", action="append", required=True)
    p.add_argument("--query-manifest", action="append", default=[])
    p.add_argument("--query-label", action="append", default=[])
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/optimization_grid_from_C")
    p.add_argument("--model-types", default="A0,A1,A2,A3")
    p.add_argument("--protein-inputs", default="B0")
    p.add_argument("--postprocesses", default="C0,C1,C2")
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    p.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--protein-dim", type=int, default=128)
    p.add_argument("--kmer-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--kmer-samples", type=int, default=768)
    p.add_argument("--profile-chunk", type=int, default=2048)
    p.add_argument("--pairwise-ranking-weight", type=float, default=0.1)
    p.add_argument("--ranking-margin", type=float, default=0.2)
    p.add_argument("--rank-pos-n", type=int, default=50)
    p.add_argument("--rank-neg-n", type=int, default=200)
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--torch-num-threads", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.kmer_samples = min(args.kmer_samples, 128)
        args.profile_chunk = min(args.profile_chunk, 2048)
    seed_all(args.seed)
    setup_threads(args.torch_num_threads)
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
    args.output_dir = str(out)

    x_map, _ = load_h5_features(resolve_path(args.per_residue_h5, ["rnacompete_rbd_per_residue_esmc.h5"]), resolve_path(args.per_residue_manifest, ["rnacompete_rbd_per_residue_manifest.tsv"], required=False))
    motif_ids, profiles, kmers, id2y = load_motif(resolve_path(args.motif_npz, ["motif_profiles.npz"]))
    split_ids = split_ids_from_tsv(resolve_path(args.split_tsv, [Path(args.split_tsv).name]), x_map, id2y)
    all_ids = split_ids["train"] + split_ids["val"] + split_ids["test"]
    profile_by_id, dropped = build_profile_maps(all_ids, profiles, id2y)
    for split in split_ids:
        split_ids[split] = [pid for pid in split_ids[split] if pid in profile_by_id]
        if args.quick_test:
            split_ids[split] = split_ids[split][: min(len(split_ids[split]), 12 if split == "train" else 6)]
    log(f"ids train={len(split_ids['train'])} val={len(split_ids['val'])} test={len(split_ids['test'])} dropped={len(dropped)}")

    manifests = list(args.query_manifest)
    while len(manifests) < len(args.query_h5):
        manifests.append(None)
    query_x = {}
    for h5_text, man_text in zip(args.query_h5, manifests):
        qx, _ = load_h5_features(resolve_path(h5_text, [Path(h5_text).name]), resolve_path(man_text, [Path(man_text).name], required=False) if man_text else None)
        query_x.update(qx)

    masks = motif_masks(kmers)
    h1 = neighbor_indices(kmers, 1)
    h2 = None
    if "C3" in args.postprocesses.split(","):
        h2 = neighbor_indices(kmers, 2)

    grid_rows, val_rows, query_all = [], [], []
    model_types = args.model_types.split(",")
    protein_inputs = args.protein_inputs.split(",")
    postprocesses = args.postprocesses.split(",")
    ckpt_by_model = {}
    val_raw_by_model = {}
    query_raw_by_model = {}
    best_epoch_by_model = {}
    metrics_by_model = {}

    for model_type in model_types:
        for protein_input in protein_inputs:
            if protein_input != "B0":
                grid_rows.append({"config_id": f"{model_type}_{protein_input}_SKIPPED", "model_type": model_type, "protein_input": protein_input, "postprocess": "", "seed": args.seed, "train_split": len(split_ids["train"]), "val_split": len(split_ids["val"]), "test_split": len(split_ids["test"]), "notes": "skipped: full-length/multi-RBD segment embeddings unavailable in current bundle"})
                continue
            model_key = f"{model_type}_{protein_input}"
            if model_type == "A0":
                ckpt = resolve_path(args.baseline_checkpoint, [Path(args.baseline_checkpoint).name])
                ckpt_by_model[model_key] = str(ckpt)
                val_raw_by_model[model_key] = predict_latent_checkpoint(ckpt, split_ids["val"], x_map, args.batch_size, device)
                query_raw_by_model[model_key] = predict_latent_checkpoint(ckpt, list(query_x), query_x, 1, device)
                best_epoch_by_model[model_key] = torch.load(ckpt, map_location="cpu", weights_only=False).get("best_epoch")
            else:
                cid = f"{model_type}_{protein_input}_train"
                ckpt, best_epoch, best_metrics, score = train_pairwise(cid, model_type, split_ids["train"], split_ids["val"], x_map, profile_by_id, kmers, args, device)
                ckpt_by_model[model_key] = str(ckpt)
                best_epoch_by_model[model_key] = best_epoch
                model, kmer_onehot = load_pairwise_checkpoint(ckpt, device)
                val_loader = DataLoader(ProteinProfileDataset(split_ids["val"], x_map, profile_by_id), batch_size=args.batch_size, shuffle=False, collate_fn=collate_protein_batch)
                val_raw_by_model[model_key] = predict_pairwise_profiles(model, val_loader, len(kmers), args.profile_chunk, device, kmer_onehot)
                q_profile_dummy = {pid: np.zeros(len(kmers), dtype=np.float32) for pid in query_x}
                q_loader = DataLoader(ProteinProfileDataset(list(query_x), query_x, q_profile_dummy), batch_size=1, shuffle=False, collate_fn=collate_protein_batch)
                query_raw_by_model[model_key] = predict_pairwise_profiles(model, q_loader, len(kmers), args.profile_chunk, device, kmer_onehot)
            for post in postprocesses:
                cid = "C_baseline" if model_type == "A0" and protein_input == "B0" and post == "C0" else f"{model_type}_{protein_input}_{post}"
                cfg = {"config_id": cid, "model_type": model_type, "protein_input": protein_input, "postprocess": post, "seed": args.seed}
                grid_rows.append({**cfg, "train_split": len(split_ids["train"]), "val_split": len(split_ids["val"]), "test_split": len(split_ids["test"]), "notes": ""})
                metrics, qrows = evaluate_config_profiles(cfg, val_raw_by_model[model_key], split_ids["val"], profile_by_id, query_raw_by_model[model_key], kmers, masks, h1, h2)
                val_rows.append({**cfg, **metrics, "best_epoch": best_epoch_by_model.get(model_key), "checkpoint_path": ckpt_by_model[model_key]})
                query_all.extend(qrows)

    grid = pd.DataFrame(grid_rows)
    val_df = pd.DataFrame(val_rows)
    query_df = pd.DataFrame(query_all)
    delta_df = baseline_delta_rows(val_df, query_df)
    direction_df = direction_summary(val_df, delta_df)
    grid.to_csv(out / "experiment_grid.tsv", sep="\t", index=False)
    val_df.to_csv(out / "validation_results_all_configs.tsv", sep="\t", index=False)
    query_df.to_csv(out / "query_results_all_configs.tsv", sep="\t", index=False)
    delta_df.to_csv(out / "baseline_vs_optimized_summary.tsv", sep="\t", index=False)
    direction_df.to_csv(out / "direction_best_summary.tsv", sep="\t", index=False)
    (out / "final_recommendation.txt").write_text(final_recommendation(val_df, delta_df))
    log(f"wrote results to {out}")


if __name__ == "__main__":
    main()
