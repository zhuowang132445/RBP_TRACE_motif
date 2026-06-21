#!/usr/bin/env python3
"""Build a query-centered RNAcompete neighbor pool from per-residue RBD embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def safe_key(pid: str) -> str:
    return pid.replace("/", "__slash__")


def load_h5_mean_vectors(path: Path, manifest: Path | None = None) -> tuple[list[str], np.ndarray, dict[str, int]]:
    allowed = None
    if manifest and manifest.exists():
        m = pd.read_csv(manifest, sep="\t")
        if "status" in m.columns and "protein_id" in m.columns:
            allowed = set(m.loc[m["status"].astype(str) == "ok", "protein_id"].astype(str))
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    lengths: dict[str, int] = {}
    with h5py.File(path, "r") as h5:
        if "metadata" in h5 and "protein_ids" in h5["metadata"]:
            raw_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["metadata/protein_ids"][()]]
        else:
            raw_ids = list(h5["embeddings"].keys())
        for pid in raw_ids:
            if allowed is not None and pid not in allowed:
                continue
            key = safe_key(pid)
            if key not in h5["embeddings"]:
                continue
            arr = np.asarray(h5["embeddings"][key], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            ids.append(pid)
            vecs.append(arr.mean(axis=0))
            lengths[pid] = int(arr.shape[0])
    return ids, normalize(np.vstack(vecs)), lengths


def short_id(pid: str) -> str:
    if "|original=" in pid:
        return pid.split("|original=", 1)[1].split("|", 1)[0]
    return pid.split("|", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    ap.add_argument("--train-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    ap.add_argument("--query-h5", action="append", required=True)
    ap.add_argument("--query-manifest", action="append", default=[])
    ap.add_argument("--metadata-tsv", default="results/embedding_domain_audit/domain_annotation_check.tsv")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ids, train_x, train_len = load_h5_mean_vectors(Path(args.train_h5), Path(args.train_manifest))

    q_ids_all: list[str] = []
    q_x_all: list[np.ndarray] = []
    q_len_all: dict[str, int] = {}
    manifests = list(args.query_manifest)
    while len(manifests) < len(args.query_h5):
        manifests.append(None)
    for h5_text, manifest_text in zip(args.query_h5, manifests):
        q_ids, q_x, q_len = load_h5_mean_vectors(Path(h5_text), Path(manifest_text) if manifest_text else None)
        q_ids_all.extend(q_ids)
        q_x_all.append(q_x)
        q_len_all.update(q_len)
    query_x = np.vstack(q_x_all)
    dist = pairwise_distances(query_x, train_x, metric="cosine")

    meta = pd.read_csv(args.metadata_tsv, sep="\t") if Path(args.metadata_tsv).exists() else pd.DataFrame(columns=["protein_id"])
    if "protein_id" in meta.columns:
        meta["protein_id"] = meta["protein_id"].astype(str)
        meta = meta.drop_duplicates("protein_id").set_index("protein_id")

    rows = []
    selected: set[str] = set()
    for qi, qid in enumerate(q_ids_all):
        order = np.argsort(dist[qi])[: args.top_k]
        for rank, ti in enumerate(order, start=1):
            pid = train_ids[ti]
            selected.add(pid)
            info = meta.loc[pid].to_dict() if pid in meta.index else {}
            rows.append(
                {
                    "query_id": qid,
                    "query_short_id": short_id(qid),
                    "query_rbd_length": q_len_all.get(qid, np.nan),
                    "neighbor_rank": rank,
                    "protein_id": pid,
                    "cosine_distance": float(dist[qi, ti]),
                    "train_rbd_length": train_len.get(pid, np.nan),
                    "domain_family": info.get("domain_family", np.nan),
                    "domain_architecture": info.get("domain_architecture", np.nan),
                    "species": info.get("species", np.nan),
                    "is_plant": info.get("is_plant", np.nan),
                }
            )
    neighbor_df = pd.DataFrame(rows)
    neighbor_df.to_csv(out / f"query_top{args.top_k}_neighbor_table.tsv", sep="\t", index=False)

    pool = neighbor_df.sort_values(["protein_id", "cosine_distance"]).drop_duplicates("protein_id")
    pool = pool.sort_values(["cosine_distance", "protein_id"]).reset_index(drop=True)
    pool.insert(0, "pool_rank", np.arange(1, len(pool) + 1))
    pool.to_csv(out / f"query_like_pool_top{args.top_k}.tsv", sep="\t", index=False)
    pool[["protein_id"]].to_csv(out / f"query_like_pool_top{args.top_k}_ids.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {
                "top_k": args.top_k,
                "n_queries": len(q_ids_all),
                "n_train_proteins": len(train_ids),
                "n_neighbor_rows": len(neighbor_df),
                "n_unique_query_like_proteins": len(pool),
                "min_distance": float(neighbor_df["cosine_distance"].min()),
                "median_distance": float(neighbor_df["cosine_distance"].median()),
                "max_selected_distance": float(neighbor_df["cosine_distance"].max()),
            }
        ]
    )
    summary.to_csv(out / "query_like_pool_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
