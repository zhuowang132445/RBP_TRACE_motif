#!/usr/bin/env python3
"""Audit which U-rich neighbors w1 loses from exact JPLE to CNN latent.

Frozen analysis only: no retraining, no parameter changes, no new model.
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
    kmer_family,
    l2_normalize,
    load_cnn,
    mean_pool,
    pca2,
    resolve,
    short_id,
)
from diagnose_08_representation_failure_analysis import project_jple  # noqa: E402
from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize  # noqa: E402
from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402


def log(msg: str) -> None:
    print(f"[diagnose-10] {msg}", flush=True)


def entropy(counts: list[int]) -> tuple[float, float]:
    total = float(sum(counts))
    if total <= 0:
        return float("nan"), float("nan")
    p = np.asarray([c / total for c in counts if c > 0], dtype=np.float64)
    h = float(-np.sum(p * np.log(p)))
    return h, float(h / np.log(len(FAMILIES)))


def top_kmer_profile(profile: np.ndarray, kmers: np.ndarray, k: int = 10) -> list[str]:
    return kmers[np.argsort(-profile)[:k]].astype(str).tolist()


def motif_composition(profile: np.ndarray, kmers: np.ndarray, top_k: int = 50) -> dict[str, Any]:
    top = top_kmer_profile(profile, kmers, top_k)
    fams = [kmer_family(k) for k in top]
    counts = {fam: fams.count(fam) for fam in FAMILIES}
    h, hn = entropy([counts[fam] for fam in FAMILIES])
    row: dict[str, Any] = {
        "top_motif": top[0],
        "top10_motifs": ",".join(top[:10]),
        "motif_entropy": h,
        "motif_entropy_normalized": hn,
    }
    for fam in FAMILIES:
        row[f"motif_top50_fraction_{fam}"] = counts[fam] / float(top_k)
    return row


def motif_region(row: pd.Series) -> str:
    top = str(row["top_motif"])
    top10 = str(row.get("top10_motifs", ""))
    if row.get("motif_family") == "U-rich":
        return "U-rich"
    if row.get("motif_family") == "GA-rich":
        return "GA-rich"
    if row.get("motif_family") == "CUUCU-like":
        return "CUUCU-like"
    cau_hits = sum(seed in top10 for seed in ["CAU", "ACAU", "CAUA", "AUAGU", "CAUAG"])
    if cau_hits >= 2 or any(seed in top for seed in ["CAU", "AUA", "UAC"]):
        return "CAU/AU-rich mixed"
    au_count = top.count("A") + top.count("U")
    if au_count >= 5:
        return "AU-rich mixed"
    return "random/mixed"


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["protein_id", "domain_family", "domain_architecture"])
    df = pd.read_csv(path, sep="\t")
    keep = [c for c in ["protein_id", "domain_family", "domain_architecture", "sequence_length"] if c in df.columns]
    return df[keep].drop_duplicates("protein_id")


def rank_all(query_w: np.ndarray, train_w: np.ndarray, train_ids: list[str]) -> pd.DataFrame:
    dist = cdist(query_w[None, :], train_w, "cosine")[0]
    order = np.argsort(dist)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return pd.DataFrame({"protein_id": train_ids, "rank": rank.astype(int), "distance": dist.astype(float)})


def local_clusters(latent: np.ndarray, ids: list[str], max_k: int = 4) -> pd.DataFrame:
    if len(ids) < 4:
        return pd.DataFrame({"protein_id": ids, "local_latent_cluster": 1})
    dist = cdist(latent, latent, "cosine")
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(z, min(max_k, len(ids) - 1), criterion="maxclust")
    return pd.DataFrame({"protein_id": ids, "local_latent_cluster": labels.astype(int)})


def build_neighbor_tables(
    exact_rank: pd.DataFrame,
    cnn_rank: pd.DataFrame,
    train_family: dict[str, str],
    y_by_id: dict[str, np.ndarray],
    kmers: np.ndarray,
    manifest: pd.DataFrame,
    local_cluster: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact_top = set(exact_rank.nsmallest(50, "rank")["protein_id"])
    cnn_top = set(cnn_rank.nsmallest(50, "rank")["protein_id"])
    rank_df = exact_rank.rename(columns={"rank": "rank_exact", "distance": "distance_exact_jple"}).merge(
        cnn_rank.rename(columns={"rank": "rank_cnn", "distance": "distance_cnn"}), on="protein_id", how="outer"
    )
    rank_df["motif_family"] = rank_df["protein_id"].map(train_family)
    rows = []
    for pid in rank_df["protein_id"]:
        in_exact = pid in exact_top
        in_cnn = pid in cnn_top
        if in_exact and in_cnn:
            status = "shared"
        elif in_exact:
            status = "lost"
        elif in_cnn:
            status = "gained"
        else:
            status = "outside"
        rows.append(status)
    rank_df["status"] = rows
    for pid in rank_df["protein_id"]:
        pass
    comp_rows = []
    for pid in rank_df["protein_id"]:
        comp = motif_composition(y_by_id[pid], kmers, 50)
        comp["protein_id"] = pid
        comp_rows.append(comp)
    rank_df = rank_df.merge(pd.DataFrame(comp_rows), on="protein_id", how="left")
    if not manifest.empty:
        rank_df = rank_df.merge(manifest, on="protein_id", how="left")
    rank_df = rank_df.merge(local_cluster, on="protein_id", how="left")
    rank_df["motif_region"] = rank_df.apply(motif_region, axis=1)
    lost = rank_df[rank_df["status"] == "lost"].copy().sort_values("rank_exact")
    gained = rank_df[rank_df["status"] == "gained"].copy().sort_values("rank_cnn")
    return rank_df, lost, gained


def shift_direction_table(
    w1_exact: np.ndarray,
    w1_cnn: np.ndarray,
    train_w: np.ndarray,
    train_ids: list[str],
    train_family: dict[str, str],
    train_top1: dict[str, str],
) -> pd.DataFrame:
    delta = np.asarray(w1_cnn - w1_exact, dtype=np.float64)
    norm = np.linalg.norm(delta)
    if norm == 0:
        norm = 1.0
    unit = delta / norm
    rel = np.asarray(train_w - w1_exact[None, :], dtype=np.float64)
    rel_norm = np.linalg.norm(rel, axis=1)
    rel_norm[rel_norm == 0] = np.nan
    projection = rel @ unit
    cos = projection / rel_norm
    df = pd.DataFrame(
        {
            "protein_id": train_ids,
            "projection_score": projection,
            "direction_cosine": cos,
            "distance_from_w1_exact": rel_norm,
            "motif_family": [train_family[pid] for pid in train_ids],
            "top_motif": [train_top1[pid] for pid in train_ids],
        }
    )
    return df.sort_values(["projection_score", "direction_cosine"], ascending=False)


def plot_local_manifold(
    path: Path,
    union_ids: list[str],
    train_w: np.ndarray,
    train_id_to_idx: dict[str, int],
    status_by_id: dict[str, str],
    w1_exact: np.ndarray,
    w1_cnn: np.ndarray,
) -> None:
    x = np.vstack([train_w[[train_id_to_idx[pid] for pid in union_ids]], w1_exact[None, :], w1_cnn[None, :]])
    xy = pca2(x)
    n = len(union_ids)
    colors = {"shared": "#555555", "lost": "#2ca02c", "gained": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    for status in ["shared", "lost", "gained"]:
        idx = [i for i, pid in enumerate(union_ids) if status_by_id[pid] == status]
        if idx:
            ax.scatter(xy[idx, 0], xy[idx, 1], s=30, alpha=0.75, label=status, c=colors[status])
    ax.scatter(xy[n, 0], xy[n, 1], marker="*", s=260, c="blue", edgecolors="black", linewidths=0.7, label="w1 exact")
    ax.scatter(xy[n + 1, 0], xy[n + 1, 1], marker="X", s=140, c="red", edgecolors="black", linewidths=0.7, label="w1 CNN")
    ax.plot([xy[n, 0], xy[n + 1, 0]], [xy[n, 1], xy[n + 1, 1]], c="red", lw=1.5, alpha=0.8)
    ax.set_title("w1 local manifold: exact top50 vs CNN top50")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def try_plot_umap(
    path: Path,
    union_ids: list[str],
    train_w: np.ndarray,
    train_id_to_idx: dict[str, int],
    status_by_id: dict[str, str],
    w1_exact: np.ndarray,
    w1_cnn: np.ndarray,
) -> str:
    try:
        import umap  # type: ignore
    except Exception as exc:
        return f"not_available: {exc}"
    x = np.vstack([train_w[[train_id_to_idx[pid] for pid in union_ids]], w1_exact[None, :], w1_cnn[None, :]])
    xy = umap.UMAP(n_neighbors=min(15, len(union_ids) - 1), min_dist=0.1, metric="cosine", random_state=20260617).fit_transform(x)
    n = len(union_ids)
    colors = {"shared": "#555555", "lost": "#2ca02c", "gained": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    for status in ["shared", "lost", "gained"]:
        idx = [i for i, pid in enumerate(union_ids) if status_by_id[pid] == status]
        if idx:
            ax.scatter(xy[idx, 0], xy[idx, 1], s=30, alpha=0.75, label=status, c=colors[status])
    ax.scatter(xy[n, 0], xy[n, 1], marker="*", s=260, c="blue", edgecolors="black", linewidths=0.7, label="w1 exact")
    ax.scatter(xy[n + 1, 0], xy[n + 1, 1], marker="X", s=140, c="red", edgecolors="black", linewidths=0.7, label="w1 CNN")
    ax.set_title("w1 local manifold UMAP")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return "available"


def count_by_family(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    vc = df["motif_family"].value_counts().to_dict()
    for fam in FAMILIES:
        rows.append({"category": prefix, "motif_family": fam, "count": int(vc.get(fam, 0))})
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    lost: pd.DataFrame,
    gained: pd.DataFrame,
    lost_u: pd.DataFrame,
    gained_mixed: pd.DataFrame,
    rescue: pd.DataFrame,
    direction: pd.DataFrame,
    umap_status: str,
) -> None:
    lost_counts = lost["motif_family"].value_counts().to_dict()
    gained_counts = gained["motif_family"].value_counts().to_dict()
    gained_regions = gained_mixed["motif_region"].value_counts().to_dict() if not gained_mixed.empty else {}
    lost_u_domains = lost_u["domain_architecture"].value_counts(dropna=False).to_dict() if "domain_architecture" in lost_u else {}
    rescue_text = "存在关键可救邻居" if len(rescue) else "没有满足 exact rank < 10 且 CNN rank > 100 的严格可救邻居"
    if len(rescue):
        primary = "CNN encoder latent drift"
    elif lost_counts.get("U-rich", 0) >= 10 and gained_counts.get("mixed/other", 0) >= 10:
        primary = "CNN encoder latent drift + motif ambiguity"
    else:
        primary = "training support deficiency / motif ambiguity"

    lines = [
        "# Diagnose 10 Neighbor Loss Audit",
        "",
        "## Direct Answers",
        "",
        f"1. w1 lost U-rich neighbors: {lost_counts.get('U-rich', 0)}. Full list is in `w1_lost_neighbors.tsv`.",
        f"2. CNN gained mixed neighbors: {gained_counts.get('mixed/other', 0)}. Full list is in `w1_gained_neighbors.tsv`.",
        f"3. CNN pushes w1 toward motif regions: {gained_regions}.",
        f"4. Lost U-rich domain architectures: {lost_u_domains}.",
        f"5. Candidate rescue neighbors: {len(rescue)}; {rescue_text}.",
        f"6. Current strongest explanation: {primary}.",
        "",
        "## Lost Neighbor Counts",
        "",
        count_by_family(lost, "lost").to_markdown(index=False),
        "",
        "## Gained Neighbor Counts",
        "",
        count_by_family(gained, "gained").to_markdown(index=False),
        "",
        "## Lost U-rich Neighbors",
        "",
        lost_u[["protein_id", "rank_exact", "rank_cnn", "distance_exact_jple", "distance_cnn", "top_motif", "domain_family", "domain_architecture", "local_latent_cluster"]].to_markdown(index=False) if not lost_u.empty else "None.",
        "",
        "## Gained Mixed Neighbors",
        "",
        gained_mixed[["protein_id", "rank_exact", "rank_cnn", "distance_exact_jple", "distance_cnn", "top_motif", "motif_region", "domain_family", "domain_architecture", "local_latent_cluster"]].head(30).to_markdown(index=False) if not gained_mixed.empty else "None.",
        "",
        "## Shift Direction Top20",
        "",
        direction.head(20).to_markdown(index=False),
        "",
        "## Candidate Rescue Neighbors",
        "",
        rescue.to_markdown(index=False) if len(rescue) else "None under strict rule exact_rank < 10 and cnn_rank > 100.",
        "",
        f"UMAP status: {umap_status}",
        "",
        "## Output Files",
        "",
        "- `w1_lost_neighbors.tsv`",
        "- `w1_gained_neighbors.tsv`",
        "- `neighbor_loss_gain_counts.tsv`",
        "- `lost_u_rich_neighbor_summary.tsv`",
        "- `gained_mixed_neighbor_summary.tsv`",
        "- `latent_shift_direction.tsv`",
        "- `candidate_rescue_neighbors.tsv`",
        "- `w1_local_manifold_pca.png`",
        "- `w1_local_manifold_umap.png` if UMAP is available",
    ]
    (out / "diagnose_10_neighbor_loss_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--train-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_10_neighbor_loss_audit")
    p.add_argument("--num-eigenvector", type=int, default=122)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--std", type=float, default=0.2)
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
    query_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    w1_id = [pid for pid in query_map if short_id(pid) == "w1"][0]
    manifest = load_manifest(resolve(args.train_manifest))

    anchor = np.load(resolve(args.jple_anchor_npz), allow_pickle=True)
    anchor_ids = np.asarray(anchor["train_protein_id_list"]).astype(str).tolist()
    anchor_w = np.asarray(anchor["w_train"], dtype=np.float32)
    keep = [i for i, pid in enumerate(anchor_ids) if pid in train_map and pid in motif_index]
    train_ids = [anchor_ids[i] for i in keep]
    true_jple_w = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    y_by_id = {pid: y_train[i] for i, pid in enumerate(train_ids)}
    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    train_top1 = {pid: top_kmer_profile(y_train[i], kmers, 1)[0] for i, pid in enumerate(train_ids)}
    train_id_to_idx = {pid: i for i, pid in enumerate(train_ids)}
    log(f"train_n={len(train_ids)}")

    log("computing w1 exact JPLE and CNN latents")
    x_train = l2_normalize(mean_pool(train_ids, train_map))
    x_w1 = l2_normalize(mean_pool([w1_id], query_map))
    jple = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    jple.fit(x_train, y_train)
    exact_train = jple.w_train.astype(np.float32)
    w1_exact = project_jple(jple, x_w1)[0]

    # Align recomputed exact JPLE basis to saved anchor basis so delta lives in the CNN target basis.
    signs = np.sign(np.sum(exact_train * true_jple_w, axis=0))
    signs[signs == 0] = 1.0
    exact_train_aligned = exact_train * signs[None, :]
    w1_exact_aligned = w1_exact * signs

    model = load_cnn(resolve(args.cnn_checkpoint), device)
    w1_cnn = cnn_latent(model, [w1_id], query_map, device, args.batch_size)[0]

    log("ranking exact and CNN neighbors")
    exact_rank = rank_all(w1_exact_aligned, exact_train_aligned, train_ids)
    cnn_rank = rank_all(w1_cnn, true_jple_w, train_ids)
    exact_top = set(exact_rank.nsmallest(50, "rank")["protein_id"])
    cnn_top = set(cnn_rank.nsmallest(50, "rank")["protein_id"])
    union_ids = sorted(exact_top | cnn_top)
    local_cluster = local_clusters(true_jple_w[[train_id_to_idx[pid] for pid in union_ids]], union_ids, 4)
    all_rank, lost, gained = build_neighbor_tables(exact_rank, cnn_rank, train_family, y_by_id, kmers, manifest, local_cluster)

    lost.to_csv(out / "w1_lost_neighbors.tsv", sep="\t", index=False)
    gained.to_csv(out / "w1_gained_neighbors.tsv", sep="\t", index=False)
    pd.concat([count_by_family(lost, "lost"), count_by_family(gained, "gained")], ignore_index=True).to_csv(
        out / "neighbor_loss_gain_counts.tsv", sep="\t", index=False
    )

    lost_u = lost[lost["motif_family"] == "U-rich"].copy()
    gained_mixed = gained[gained["motif_family"] == "mixed/other"].copy()
    lost_u.to_csv(out / "lost_u_rich_neighbor_summary.tsv", sep="\t", index=False)
    gained_mixed.to_csv(out / "gained_mixed_neighbor_summary.tsv", sep="\t", index=False)

    log("computing shift direction")
    direction = shift_direction_table(w1_exact_aligned, w1_cnn, true_jple_w, train_ids, train_family, train_top1)
    direction.to_csv(out / "latent_shift_direction.tsv", sep="\t", index=False)

    rescue = lost_u[(lost_u["rank_exact"] < 10) & (lost_u["rank_cnn"] > 100)].copy()
    rescue.to_csv(out / "candidate_rescue_neighbors.tsv", sep="\t", index=False)

    log("plotting local manifold")
    status_by_id = {pid: ("shared" if pid in exact_top and pid in cnn_top else "lost" if pid in exact_top else "gained") for pid in union_ids}
    plot_local_manifold(out / "w1_local_manifold_pca.png", union_ids, true_jple_w, train_id_to_idx, status_by_id, w1_exact_aligned, w1_cnn)
    umap_status = try_plot_umap(out / "w1_local_manifold_umap.png", union_ids, true_jple_w, train_id_to_idx, status_by_id, w1_exact_aligned, w1_cnn)

    write_report(out, lost, gained, lost_u, gained_mixed, rescue, direction, umap_status)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
