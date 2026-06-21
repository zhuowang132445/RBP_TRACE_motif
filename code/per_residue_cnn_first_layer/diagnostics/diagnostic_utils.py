#!/usr/bin/env python3
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ID_KEYS = ["protein_ids", "profile_ids", "ids", "names"]
MOTIF_SCORE_KEYS = ["zscores", "scores", "Y", "profiles"]


def first_key(npz: np.lib.npyio.NpzFile, keys: list[str]) -> str:
    for key in keys:
        if key in npz.files:
            return key
    raise KeyError(f"None of {keys} found in {npz.files}")


def safe_id(key: str) -> str:
    return str(key).replace("__slash__", "/")


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


def standardize_rows(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    std = np.std(y, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return y / std


def load_motif_npz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    ids = np.asarray(z[first_key(z, ID_KEYS)]).astype(str).tolist()
    y = np.asarray(z[first_key(z, MOTIF_SCORE_KEYS)], dtype=np.float32)
    kmers = np.asarray(z["kmers"]).astype(str)
    return ids, y, kmers


def load_h5_per_residue(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for key in h5["embeddings"].keys():
            arr = np.asarray(h5["embeddings"][key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                out[safe_id(key)] = arr
    return out


def pool_embedding(x_map: dict[str, np.ndarray], mode: str) -> tuple[list[str], np.ndarray]:
    ids = sorted(x_map)
    rows = []
    for pid in ids:
        x = x_map[pid]
        if mode == "mean":
            row = x.mean(axis=0)
        elif mode == "max":
            row = x.max(axis=0)
        elif mode == "mean_max":
            row = np.concatenate([x.mean(axis=0), x.max(axis=0)])
        else:
            raise ValueError(f"unknown pool mode: {mode}")
        rows.append(row.astype(np.float32))
    return ids, np.vstack(rows).astype(np.float32)


def align(ids: list[str], x: np.ndarray, motif_ids: list[str], y: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx = {pid: i for i, pid in enumerate(motif_ids)}
    keep_ids, keep_x, keep_y = [], [], []
    for i, pid in enumerate(ids):
        if pid in idx and np.isfinite(x[i]).all() and np.isfinite(y[idx[pid]]).all():
            keep_ids.append(pid)
            keep_x.append(i)
            keep_y.append(idx[pid])
    return keep_ids, x[np.asarray(keep_x)], y[np.asarray(keep_y)]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata_simple(a), rankdata_simple(b))


def top_overlap(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    pred_top = set(np.argsort(-pred)[:k].tolist())
    true_top = set(np.argsort(-true)[:k].tolist())
    return len(pred_top & true_top) / float(k)


def ndcg_at_k(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    pred_order = np.argsort(-pred)[:k]
    ideal_order = np.argsort(-true)[:k]
    rel = true.astype(np.float64) - float(np.nanmin(true))
    denom = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[pred_order] / denom))
    idcg = float(np.sum(rel[ideal_order] / denom))
    return dcg / idcg if idcg > 0 else float("nan")


def true_top1_rank(pred: np.ndarray, true: np.ndarray) -> int:
    order = np.argsort(-pred)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return int(rank[int(np.argmax(true))])


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


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
    if seed_like(kmer, ["UGUGUG", "GUGUGU", "UGUGU", "GUGUG"]):
        return "UGUGUG-like"
    if (kmer.count("G") + kmer.count("A")) >= 5 or seed_like(kmer, ["GAGGA", "GGAGG", "GAAGA", "AGGAG", "GGAUG"]):
        return "GA-rich"
    return "mixed/other"


FAMILIES = ["U-rich", "CUUCU-like", "UGUGUG-like", "GA-rich", "mixed/other"]


def family_scores_from_profile(profile: np.ndarray, kmers: np.ndarray, top_k: int = 50) -> dict[str, float]:
    order = np.argsort(-profile)[:top_k]
    fams = [kmer_family(str(kmers[i])) for i in order]
    return {fam: fams.count(fam) / float(top_k) for fam in FAMILIES}


def assign_profile_family(profile: np.ndarray, kmers: np.ndarray, top_k: int = 50, min_frac: float = 0.25) -> str:
    scores = family_scores_from_profile(profile, kmers, top_k)
    best = max(scores, key=scores.get)
    return best if scores[best] >= min_frac else "mixed/other"


def summarize_profile(query_id: str, profile: np.ndarray, kmers: np.ndarray, variant: str) -> dict[str, Any]:
    order = np.argsort(-profile)
    row: dict[str, Any] = {
        "query_id": query_id,
        "variant": variant,
        "top1_kmer": str(kmers[order[0]]),
        "top5_kmers": ",".join(kmers[order[:5]].astype(str)),
        "top10_kmers": ",".join(kmers[order[:10]].astype(str)),
        "assigned_family_top50": assign_profile_family(profile, kmers, 50),
    }
    row.update({f"family_score_top50_{k}": v for k, v in family_scores_from_profile(profile, kmers, 50).items()})
    for fam in ["U-rich", "CUUCU-like", "UGUGUG-like", "GA-rich"]:
        idx = np.asarray([i for i, k in enumerate(kmers.astype(str)) if kmer_family(k) == fam], dtype=int)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        best_idx = idx[np.argmin(ranks[idx])]
        row[f"best_{fam}_rank"] = int(ranks[best_idx])
        row[f"best_{fam}_kmer"] = str(kmers[best_idx])
        row[f"best_{fam}_score"] = float(profile[best_idx])
    return row


def weighted_knn_predict(query_x: np.ndarray, train_x: np.ndarray, y_train: np.ndarray, k: int, std: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = cdist(query_x, train_x, "cosine")
    pred = []
    neigh = []
    weights_all = []
    for row in dist:
        idx = np.argsort(row)[:k]
        w = np.exp(-(row[idx] ** 2) / (std**2))
        if float(w.sum()) == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        pred.append(np.sum(w[:, None] * y_train[idx], axis=0))
        neigh.append(idx)
        weights_all.append(w)
    return np.asarray(pred, dtype=np.float32), np.asarray(neigh), np.asarray(weights_all, dtype=np.float32)


def write_tsv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with gzip.open(path, "wt") as handle:
        pd.DataFrame(rows).to_csv(handle, sep="\t", index=False)
