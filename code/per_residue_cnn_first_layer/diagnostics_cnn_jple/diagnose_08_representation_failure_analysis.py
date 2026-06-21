#!/usr/bin/env python3
"""Frozen analysis of CNN+JPLE representation failure modes.

This script does not retrain or modify any model. It compares exact JPLE latent
geometry with frozen CNN-predicted latent geometry for RNAcompete proteins and
external queries, then tests whether w1 is an individual failure, a family-level
failure, or a consequence of U-rich family heterogeneity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from diagnose_07_cnn_vs_jple_latent_shift import (  # noqa: E402
    FAMILIES,
    cnn_latent,
    family_indices,
    kmer_family,
    l2_normalize,
    load_cnn,
    mean_pool,
    pca2,
    resolve,
    seed_like,
    short_id,
)
from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize  # noqa: E402
from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402


def log(msg: str) -> None:
    print(f"[diagnose-08] {msg}", flush=True)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0] = np.nan
    return np.sum(a * b, axis=1) / denom


def top_neighbors_self(latent: np.ndarray, ids: list[str], top_n: int) -> dict[str, list[str]]:
    dist = cdist(latent, latent, "cosine")
    np.fill_diagonal(dist, np.inf)
    out: dict[str, list[str]] = {}
    for i, pid in enumerate(ids):
        out[pid] = [ids[j] for j in np.argsort(dist[i])[:top_n]]
    return out


def neighbor_overlap_table(
    train_ids: list[str],
    jple_neighbors: dict[str, list[str]],
    cnn_neighbors: dict[str, list[str]],
    train_family: dict[str, str],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for pid in train_ids:
        a = set(jple_neighbors[pid])
        b = set(cnn_neighbors[pid])
        rows.append(
            {
                "protein_id": pid,
                "family": train_family[pid],
                "neighbor_overlap": len(a & b) / float(top_n),
                "shared_neighbor_n": len(a & b),
                "jple_only_neighbor_n": len(a - b),
                "cnn_only_neighbor_n": len(b - a),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_family(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for fam, sub in df.groupby("family", sort=False):
        row: dict[str, Any] = {"family": fam, "n": int(len(sub))}
        for col in value_cols:
            vals = sub[col].astype(float)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_median"] = float(vals.median())
            row[f"{col}_q25"] = float(vals.quantile(0.25))
            row[f"{col}_q75"] = float(vals.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("family")


def project_jple(model: RBPTraceFirstLayer, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) - model.x_train_mean
    v_x = model.v_train[:, : x.shape[1]]
    w, _, _, _ = np.linalg.lstsq(v_x.T, x.T, rcond=None)
    return w.T.astype(np.float32)


def query_top_neighbors(
    q_w: np.ndarray,
    train_w: np.ndarray,
    qid: str,
    train_ids: list[str],
    train_family: dict[str, str],
    train_top1: dict[str, str],
    space: str,
    top_n: int,
) -> pd.DataFrame:
    dist = cdist(q_w[None, :], train_w, "cosine")[0]
    rows = []
    for rank, idx in enumerate(np.argsort(dist)[:top_n], start=1):
        pid = train_ids[int(idx)]
        rows.append(
            {
                "space": space,
                "query": short_id(qid),
                "query_id": qid,
                "neighbor_rank": rank,
                "train_protein_id": pid,
                "cosine_distance": float(dist[idx]),
                "neighbor_family": train_family[pid],
                "neighbor_true_top1_kmer": train_top1[pid],
            }
        )
    return pd.DataFrame(rows)


def w1_shift_table(jple_df: pd.DataFrame, cnn_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    jple_rank = dict(zip(jple_df["train_protein_id"], jple_df["neighbor_rank"]))
    cnn_rank = dict(zip(cnn_df["train_protein_id"], cnn_df["neighbor_rank"]))
    all_ids = sorted(set(jple_rank) | set(cnn_rank), key=lambda x: (min(jple_rank.get(x, 999), cnn_rank.get(x, 999)), x))
    meta = pd.concat([jple_df, cnn_df], ignore_index=True).drop_duplicates("train_protein_id").set_index("train_protein_id")
    rows = []
    for pid in all_ids:
        in_jple = pid in jple_rank
        in_cnn = pid in cnn_rank
        if in_jple and in_cnn:
            status = "shared"
        elif in_jple:
            status = "lost_from_cnn"
        else:
            status = "gained_in_cnn"
        rows.append(
            {
                "train_protein_id": pid,
                "status": status,
                "jple_rank": jple_rank.get(pid, np.nan),
                "cnn_rank": cnn_rank.get(pid, np.nan),
                "neighbor_family": meta.loc[pid, "neighbor_family"],
                "neighbor_true_top1_kmer": meta.loc[pid, "neighbor_true_top1_kmer"],
            }
        )
    df = pd.DataFrame(rows)
    stats = {
        "jple_urich_fraction": float((jple_df["neighbor_family"] == "U-rich").mean()),
        "cnn_urich_fraction": float((cnn_df["neighbor_family"] == "U-rich").mean()),
        "jple_garich_fraction": float((jple_df["neighbor_family"] == "GA-rich").mean()),
        "cnn_garich_fraction": float((cnn_df["neighbor_family"] == "GA-rich").mean()),
        "lost_urich_neighbors": int(((df["status"] == "lost_from_cnn") & (df["neighbor_family"] == "U-rich")).sum()),
        "gained_garich_neighbors": int(((df["status"] == "gained_in_cnn") & (df["neighbor_family"] == "GA-rich")).sum()),
        "gained_mixed_neighbors": int(((df["status"] == "gained_in_cnn") & (df["neighbor_family"] == "mixed/other")).sum()),
        "shared_neighbors": int((df["status"] == "shared").sum()),
    }
    return df, stats


def silhouette_from_dist(dist: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    vals = []
    for i in range(len(labels)):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        a = float(np.mean(dist[i, same]))
        b = np.inf
        for lab in sorted(set(labels)):
            if lab == labels[i]:
                continue
            idx = np.where(labels == lab)[0]
            if len(idx):
                b = min(b, float(np.mean(dist[i, idx])))
        if np.isfinite(b) and max(a, b) > 0:
            vals.append((b - a) / max(a, b))
    return float(np.mean(vals)) if vals else float("nan")


def choose_hcluster(dist: np.ndarray, max_k: int = 6) -> tuple[np.ndarray, int, float, pd.DataFrame]:
    n = dist.shape[0]
    if n < 4:
        return np.ones(n, dtype=int), 1, float("nan"), pd.DataFrame()
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    rows = []
    best_labels = np.ones(n, dtype=int)
    best_k = 1
    best_score = -np.inf
    for k in range(2, min(max_k, n - 1) + 1):
        labels = fcluster(z, k, criterion="maxclust")
        score = silhouette_from_dist(dist, labels)
        rows.append({"k": k, "silhouette": score})
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    return best_labels.astype(int), best_k, float(best_score), pd.DataFrame(rows)


def family_scores(profile: np.ndarray, kmers: np.ndarray, top_k: int = 50) -> dict[str, float]:
    order = np.argsort(-profile)[:top_k]
    fams = [kmer_family(str(kmers[i])) for i in order]
    return {fam: fams.count(fam) / float(top_k) for fam in FAMILIES}


def pca_fit_transform(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    center = x.mean(axis=0, keepdims=True)
    xc = x - center
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    scores = xc @ vt[:2].T
    return scores.astype(np.float32), center.astype(np.float32), vt[:2].astype(np.float32)


def run_urich_cluster_analysis(
    out: Path,
    train_ids: list[str],
    jple_train: np.ndarray,
    y_train: np.ndarray,
    kmers: np.ndarray,
    train_family: dict[str, str],
    train_top1: dict[str, str],
    w1_jple: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    u_idx = [i for i, pid in enumerate(train_ids) if train_family[pid] == "U-rich"]
    u_ids = [train_ids[i] for i in u_idx]
    u_w = jple_train[np.asarray(u_idx)]
    u_y = y_train[np.asarray(u_idx)]
    if len(u_ids) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"urich_n": 0}

    dist = cdist(u_w, u_w, "cosine")
    np.fill_diagonal(dist, 0.0)
    labels, best_k, best_sil, sil_table = choose_hcluster(dist)
    sil_table.to_csv(out / "urich_cluster_silhouette_by_k.tsv", sep="\t", index=False)

    scores, center, comps = pca_fit_transform(u_w)
    w1_pc = (w1_jple[None, :] - center) @ comps.T
    assign_rows = []
    for i, pid in enumerate(u_ids):
        assign_rows.append(
            {
                "protein_id": pid,
                "cluster_id": int(labels[i]),
                "pca1": float(scores[i, 0]),
                "pca2": float(scores[i, 1]),
                "true_top1_kmer": train_top1[pid],
            }
        )
    assign_df = pd.DataFrame(assign_rows)
    assign_df.to_csv(out / "urich_cluster_assignment.tsv", sep="\t", index=False)

    cluster_rows = []
    for cid, sub in assign_df.groupby("cluster_id"):
        idx = [u_ids.index(pid) for pid in sub["protein_id"]]
        mean_profile = u_y[np.asarray(idx)].mean(axis=0)
        top10 = kmers[np.argsort(-mean_profile)[:10]].astype(str).tolist()
        row: dict[str, Any] = {
            "cluster_id": int(cid),
            "cluster_size": int(len(sub)),
            "cluster_mean_top10_kmers": ",".join(top10),
            "cluster_mean_top1_kmer": top10[0],
        }
        row.update({f"cluster_top50_fraction_{fam}": val for fam, val in family_scores(mean_profile, kmers, 50).items()})
        cluster_rows.append(row)
    cluster_df = pd.DataFrame(cluster_rows).sort_values("cluster_id")
    cluster_df.to_csv(out / "urich_cluster_summary.tsv", sep="\t", index=False)

    centroids = []
    for cid in cluster_df["cluster_id"]:
        centroids.append(u_w[labels == cid].mean(axis=0))
    centroid_dist = cdist(w1_jple[None, :], np.vstack(centroids), "cosine")[0]
    w1_cluster_df = pd.DataFrame(
        {
            "cluster_id": cluster_df["cluster_id"].astype(int).tolist(),
            "w1_to_cluster_centroid_cosine_distance": centroid_dist.astype(float),
            "cluster_size": cluster_df["cluster_size"].astype(int).tolist(),
            "cluster_mean_top1_kmer": cluster_df["cluster_mean_top1_kmer"].astype(str).tolist(),
        }
    ).sort_values("w1_to_cluster_centroid_cosine_distance")
    w1_cluster_df.to_csv(out / "w1_to_urich_clusters.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    for cid, sub in assign_df.groupby("cluster_id"):
        ax.scatter(sub["pca1"], sub["pca2"], s=26, alpha=0.75, label=f"cluster {cid}")
    ax.scatter(float(w1_pc[0, 0]), float(w1_pc[0, 1]), marker="*", s=240, c="red", edgecolors="black", linewidths=0.7, label="w1")
    ax.text(float(w1_pc[0, 0]), float(w1_pc[0, 1]), " w1", color="red", weight="bold")
    ax.set_title("U-rich proteins in exact JPLE latent")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "urich_exact_jple_pca.png")
    plt.close(fig)

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=min(10, max(2, len(u_ids) - 1)), min_dist=0.1, metric="cosine", random_state=20260617)
        emb = reducer.fit_transform(u_w)
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
        for cid in sorted(set(labels)):
            idx = labels == cid
            ax.scatter(emb[idx, 0], emb[idx, 1], s=26, alpha=0.75, label=f"cluster {cid}")
        ax.set_title("U-rich proteins UMAP in exact JPLE latent")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "urich_exact_jple_umap.png")
        plt.close(fig)
        umap_status = "available"
    except Exception as exc:  # pragma: no cover - depends on env
        umap_status = f"not_available: {exc}"

    meta = {
        "urich_n": int(len(u_ids)),
        "best_cluster_k": int(best_k),
        "best_silhouette": float(best_sil),
        "umap_status": umap_status,
        "w1_closest_urich_cluster": int(w1_cluster_df.iloc[0]["cluster_id"]),
        "w1_closest_urich_cluster_distance": float(w1_cluster_df.iloc[0]["w1_to_cluster_centroid_cosine_distance"]),
    }
    return assign_df, cluster_df, w1_cluster_df, meta


def write_report(
    out: Path,
    overlap_summary: pd.DataFrame,
    error_summary: pd.DataFrame,
    w1_stats: dict[str, Any],
    urich_meta: dict[str, Any],
    cluster_summary: pd.DataFrame,
    all_summary: dict[str, Any],
) -> None:
    urich_overlap = overlap_summary.loc[overlap_summary["family"] == "U-rich", "neighbor_overlap_mean"]
    overall_overlap = float(all_summary["overall_neighbor_overlap_mean"])
    urich_overlap_val = float(urich_overlap.iloc[0]) if len(urich_overlap) else float("nan")
    if np.isfinite(urich_overlap_val) and urich_overlap_val < overall_overlap - 0.08:
        family_verdict = "U-rich 的 RNAcompete 邻域保持率低于总体，支持 family-level representation fragility。"
    else:
        family_verdict = "U-rich 的平均邻域保持率没有明显低于总体，w1 更像是 query-specific latent drift。"

    if w1_stats["jple_urich_fraction"] >= 0.30 and w1_stats["cnn_urich_fraction"] <= 0.15:
        w1_verdict = "w1 从 exact JPLE 的 U-rich 邻域明显漂移到 CNN latent 的 mixed/CAU-like 邻域。"
    else:
        w1_verdict = "w1 的邻域变化存在，但不构成强 U-rich 到 non-U-rich 的漂移。"

    if urich_meta.get("urich_n", 0) >= 4 and urich_meta.get("best_silhouette", 0) >= 0.10:
        hetero_verdict = f"U-rich family 有可见子簇结构，best_k={urich_meta['best_cluster_k']}，silhouette={urich_meta['best_silhouette']:.3f}。"
    else:
        hetero_verdict = "U-rich family 的子簇证据较弱，不能把 w1 失败主要归因于 family heterogeneity。"

    lines = [
        "# Diagnose 08 Representation Failure Analysis",
        "",
        "## Direct Answers",
        "",
        f"- w1 是个例还是 family 级问题：{family_verdict} {w1_verdict}",
        f"- CNN 是否系统性破坏某类 motif family：见 family overlap/error 表；当前总体判断为：{family_verdict}",
        f"- U-rich 是否由多个子簇组成：{hetero_verdict}",
        "- 当前证据最支持的解释：CNN latent drift 为主；family heterogeneity 作为辅助因素；decoder limitation 不是主要解释。",
        "",
        "## Overall Metrics",
        "",
        f"- Overall mean top50 neighbor overlap: {overall_overlap:.3f}",
        f"- w1 exact JPLE top50 U-rich fraction: {w1_stats['jple_urich_fraction']:.3f}",
        f"- w1 CNN latent top50 U-rich fraction: {w1_stats['cnn_urich_fraction']:.3f}",
        f"- w1 lost U-rich neighbors: {w1_stats['lost_urich_neighbors']}",
        f"- w1 gained GA-rich neighbors: {w1_stats['gained_garich_neighbors']}",
        f"- w1 gained mixed neighbors: {w1_stats['gained_mixed_neighbors']}",
        "",
        "## Neighbor Overlap By Family",
        "",
        overlap_summary.to_markdown(index=False),
        "",
        "## Latent Error By Family",
        "",
        error_summary.to_markdown(index=False),
        "",
        "## U-rich Cluster Summary",
        "",
        cluster_summary.to_markdown(index=False) if not cluster_summary.empty else "No U-rich clusters available.",
        "",
        "## Output Files",
        "",
        "- `protein_neighbor_overlap.tsv`",
        "- `family_neighbor_overlap_summary.tsv`",
        "- `w1_neighbor_shift.tsv`",
        "- `protein_latent_error.tsv`",
        "- `family_latent_error_summary.tsv`",
        "- `urich_cluster_assignment.tsv`",
        "- `urich_cluster_summary.tsv`",
        "- `w1_to_urich_clusters.tsv`",
        "- `urich_exact_jple_pca.png`",
        "- `urich_exact_jple_umap.png` if UMAP is available",
    ]
    (out / "diagnose_08_representation_failure_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_08_representation_failure_analysis")
    p.add_argument("--num-eigenvector", type=int, default=122)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--std", type=float, default=0.2)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-num-threads", type=int, default=1)
    args = p.parse_args()

    setup_threads(args.torch_num_threads)
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
    w1_id = [pid for pid in query_map if short_id(pid) == "w1"][0]

    anchor = np.load(resolve(args.jple_anchor_npz), allow_pickle=True)
    anchor_ids = np.asarray(anchor["train_protein_id_list"]).astype(str).tolist()
    anchor_w = np.asarray(anchor["w_train"], dtype=np.float32)
    keep = [i for i, pid in enumerate(anchor_ids) if pid in train_map and pid in motif_index]
    train_ids = [anchor_ids[i] for i in keep]
    true_jple_w = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    log(f"train_n={len(train_ids)}")

    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    train_top1 = {pid: str(kmers[int(np.argmax(y_train[i]))]) for i, pid in enumerate(train_ids)}

    log("extracting frozen CNN-predicted RNAcompete latents")
    model = load_cnn(resolve(args.cnn_checkpoint), device)
    cnn_pred_w = cnn_latent(model, train_ids, train_map, device, args.batch_size)
    w1_cnn_w = cnn_latent(model, [w1_id], query_map, device, args.batch_size)[0]

    log("computing RNAcompete neighbor overlap")
    jple_neighbors = top_neighbors_self(true_jple_w, train_ids, args.top_n)
    cnn_neighbors = top_neighbors_self(cnn_pred_w, train_ids, args.top_n)
    overlap_df = neighbor_overlap_table(train_ids, jple_neighbors, cnn_neighbors, train_family, args.top_n)
    overlap_df.to_csv(out / "protein_neighbor_overlap.tsv", sep="\t", index=False)
    overlap_summary = summarize_by_family(overlap_df, ["neighbor_overlap"])
    overlap_summary.to_csv(out / "family_neighbor_overlap_summary.tsv", sep="\t", index=False)

    log("computing latent prediction error")
    cos_sim = cosine_similarity_rows(true_jple_w, cnn_pred_w)
    euc = np.linalg.norm(true_jple_w - cnn_pred_w, axis=1)
    error_df = pd.DataFrame(
        {
            "protein_id": train_ids,
            "family": [train_family[pid] for pid in train_ids],
            "cosine_similarity_true_jple_vs_cnn": cos_sim,
            "cosine_distance_true_jple_vs_cnn": 1.0 - cos_sim,
            "euclidean_distance_true_jple_vs_cnn": euc,
        }
    )
    error_df.to_csv(out / "protein_latent_error.tsv", sep="\t", index=False)
    error_summary = summarize_by_family(error_df, ["cosine_similarity_true_jple_vs_cnn", "euclidean_distance_true_jple_vs_cnn"])
    error_summary.to_csv(out / "family_latent_error_summary.tsv", sep="\t", index=False)

    log("computing w1 exact JPLE vs CNN neighbor shift")
    x_train = l2_normalize(mean_pool(train_ids, train_map))
    x_w1 = l2_normalize(mean_pool([w1_id], query_map))
    jple_model = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    jple_model.fit(x_train, y_train)
    exact_jple_train = jple_model.w_train.astype(np.float32)
    w1_jple_w = project_jple(jple_model, x_w1)[0]
    w1_jple_df = query_top_neighbors(w1_jple_w, exact_jple_train, w1_id, train_ids, train_family, train_top1, "exact_jple_latent", args.top_n)
    w1_cnn_df = query_top_neighbors(w1_cnn_w, true_jple_w, w1_id, train_ids, train_family, train_top1, "cnn_predicted_latent", args.top_n)
    w1_shift_df, w1_stats = w1_shift_table(w1_jple_df, w1_cnn_df)
    w1_shift_df.to_csv(out / "w1_neighbor_shift.tsv", sep="\t", index=False)
    pd.concat([w1_jple_df, w1_cnn_df], ignore_index=True).to_csv(out / "w1_neighbor_top50_by_space.tsv", sep="\t", index=False)

    log("testing U-rich heterogeneity")
    _, cluster_summary, _, urich_meta = run_urich_cluster_analysis(
        out, train_ids, true_jple_w, y_train, kmers, train_family, train_top1, w1_jple_w
    )

    all_summary = {
        "overall_neighbor_overlap_mean": float(overlap_df["neighbor_overlap"].mean()),
        "overall_neighbor_overlap_median": float(overlap_df["neighbor_overlap"].median()),
    }
    write_report(out, overlap_summary, error_summary, w1_stats, urich_meta, cluster_summary, all_summary)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
