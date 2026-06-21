#!/usr/bin/env python3
"""Shared lightweight CNN utilities for final per-residue RBD query prediction."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]

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
