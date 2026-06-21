#!/usr/bin/env python3
"""Diagnose CNN latent vs exact JPLE latent shift for CNN+JPLE queries.

No retraining is performed. The script compares:
1. frozen CNN output latent from per-residue RBD embeddings;
2. exact JPLE latent projection from mean-pooled per-residue RBD embeddings.
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
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from cnn_model_utils import PerResidueCnn, collate_batch, load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import align, assign_profile_family, load_motif_npz, row_l2_normalize, standardize_rows  # noqa: E402
from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402


FAMILIES = ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich", "mixed/other"]


def log(msg: str) -> None:
    print(f"[diagnose-07] {msg}", flush=True)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    q = ROOT / p
    if q.exists():
        return q
    raise FileNotFoundError(path)


def short_id(pid: str) -> str:
    pid = str(pid)
    if pid.startswith("AtPTBP3"):
        return "AtPTBP3"
    if "|original=" in pid:
        return pid.split("|original=", 1)[1].split("|", 1)[0]
    return pid.split("|", 1)[0]


def seed_like(kmer: str, seeds: list[str], max_hamming: int = 1) -> bool:
    for seed in seeds:
        n = len(seed)
        if n > len(kmer):
            continue
        for start in range(len(kmer) - n + 1):
            if sum(a != b for a, b in zip(kmer[start : start + n], seed)) <= max_hamming:
                return True
    return False


def kmer_family(kmer: str) -> str:
    kmer = str(kmer)
    if kmer.count("U") >= 5 or "UUUUU" in kmer:
        return "U-rich"
    if seed_like(kmer, ["CUUCU", "UCUUC", "CUUCUC", "UCUUCU", "CUUCUU"]):
        return "CUUCU-like"
    if seed_like(kmer, ["UCUCUC", "CUCUCU", "UCUCU", "CUCUC"]):
        return "UCUCUC-like"
    if seed_like(kmer, ["UGUGUG", "GUGUGU", "UGUGU", "GUGUG"]):
        return "UGUGUG-like"
    if (kmer.count("G") + kmer.count("A")) >= 5 or seed_like(kmer, ["GAGGA", "GGAGG", "GAAGA", "AGGAG", "GGAUG"]):
        return "GA-rich"
    return "mixed/other"


def family_indices(kmers: np.ndarray) -> dict[str, np.ndarray]:
    out = {fam: [] for fam in FAMILIES}
    for i, kmer in enumerate(kmers.astype(str)):
        out[kmer_family(kmer)].append(i)
    return {fam: np.asarray(idx, dtype=np.int64) for fam, idx in out.items()}


def best_rank(profile: np.ndarray, idx: np.ndarray, kmers: np.ndarray) -> tuple[int, str, float]:
    order = np.argsort(-profile)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    best = idx[np.argmin(ranks[idx])]
    return int(ranks[best]), str(kmers[best]), float(profile[best])


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return (x / denom).astype(np.float32)


def mean_pool(ids: list[str], x_map: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([x_map[pid].mean(axis=0) for pid in ids]).astype(np.float32)


def load_cnn(checkpoint: Path, device: torch.device) -> PerResidueCnn:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    conf = ckpt["model_config"]
    model = PerResidueCnn(
        int(conf["input_dim"]),
        int(conf["hidden_dim"]),
        int(conf["latent_dim"]),
        int(conf["kernel_size"]),
        int(conf["num_blocks"]),
        float(conf["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, ids: list[str], x_map: dict[str, np.ndarray], latent_dim: int):
        self.ids = ids
        self.x_map = x_map
        self.y = np.zeros((len(ids), latent_dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        return pid, torch.from_numpy(self.x_map[pid].astype(np.float32)), torch.from_numpy(self.y[idx])


def cnn_latent(model: PerResidueCnn, ids: list[str], x_map: dict[str, np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    latent_dim = int(model.head[-1].out_features)
    loader = torch.utils.data.DataLoader(
        LatentDataset(ids, x_map, latent_dim),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    rows = []
    with torch.no_grad():
        for _, x, mask, _ in loader:
            rows.append(model(x.to(device), mask.to(device)).cpu().numpy().astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def jple_project(model: RBPTraceFirstLayer, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) - model.x_train_mean
    v_x = model.v_train[:, : x.shape[1]]
    w, _, _, _ = np.linalg.lstsq(v_x.T, x.T, rcond=None)
    return w.T.astype(np.float32)


def decode(
    q_w: np.ndarray,
    t_w: np.ndarray,
    y_train: np.ndarray,
    query_ids: list[str],
    train_ids: list[str],
    space: str,
    threshold: float,
    std: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    dist = cdist(q_w, t_w, "cosine")
    sim = np.exp(-(dist**2) / (std**2))
    preds = []
    rows: list[dict[str, Any]] = []
    for qi, qid in enumerate(query_ids):
        idx = np.argwhere(sim[qi] >= threshold).flatten()
        if len(idx) == 0:
            idx = np.asarray([int(np.argmax(sim[qi]))])
        idx = idx[np.argsort(-sim[qi, idx])]
        w = sim[qi, idx]
        w = w / w.sum()
        preds.append(np.sum(w[:, None] * y_train[idx], axis=0))
        for rank, (ti, ww) in enumerate(zip(idx, w), start=1):
            rows.append(
                {
                    "space": space,
                    "query": short_id(qid),
                    "query_id": qid,
                    "decoder_neighbor_rank": rank,
                    "train_protein_id": train_ids[int(ti)],
                    "cosine_distance": float(dist[qi, ti]),
                    "decoder_weight": float(ww),
                }
            )
    return standardize_rows(np.asarray(preds, dtype=np.float32)), pd.DataFrame(rows)


def nearest_table(
    q_w: np.ndarray,
    t_w: np.ndarray,
    query_ids: list[str],
    train_ids: list[str],
    train_family: dict[str, str],
    train_top1: dict[str, str],
    space: str,
    top_n: int,
) -> pd.DataFrame:
    dist = cdist(q_w, t_w, "cosine")
    rows = []
    for qi, qid in enumerate(query_ids):
        for rank, ti in enumerate(np.argsort(dist[qi])[:top_n], start=1):
            pid = train_ids[int(ti)]
            rows.append(
                {
                    "space": space,
                    "query": short_id(qid),
                    "query_id": qid,
                    "neighbor_rank": rank,
                    "train_protein_id": pid,
                    "cosine_distance": float(dist[qi, ti]),
                    "neighbor_family": train_family[pid],
                    "neighbor_true_top1_kmer": train_top1[pid],
                }
            )
    return pd.DataFrame(rows)


def family_distribution(neigh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (space, query, qid), sub in neigh.groupby(["space", "query", "query_id"], sort=False):
        counts = sub["neighbor_family"].value_counts().to_dict()
        row: dict[str, Any] = {"space": space, "query": query, "query_id": qid, "neighbor_n": int(len(sub))}
        for fam in FAMILIES:
            row[f"top50_count_{fam}"] = int(counts.get(fam, 0))
            row[f"top50_fraction_{fam}"] = float(counts.get(fam, 0) / len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_shift(dist_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summary = [], []
    for (query, qid), sub in dist_df.groupby(["query", "query_id"], sort=False):
        by = {r["space"]: r for _, r in sub.iterrows()}
        srow: dict[str, Any] = {"query": query, "query_id": qid}
        for fam in FAMILIES:
            cnn = float(by["cnn_latent"][f"top50_fraction_{fam}"])
            jple = float(by["jple_latent"][f"top50_fraction_{fam}"])
            rows.append(
                {
                    "query": query,
                    "query_id": qid,
                    "family": fam,
                    "cnn_top50_fraction": cnn,
                    "jple_top50_fraction": jple,
                    "jple_minus_cnn_fraction": jple - cnn,
                }
            )
            srow[f"cnn_top50_fraction_{fam}"] = cnn
            srow[f"jple_top50_fraction_{fam}"] = jple
            srow[f"shift_jple_minus_cnn_{fam}"] = jple - cnn
        summary.append(srow)
    return pd.DataFrame(rows), pd.DataFrame(summary)


def rank_shift(query_ids: list[str], profiles: dict[str, np.ndarray], kmers: np.ndarray, fam_idx: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    u7 = int(np.where(kmers.astype(str) == "UUUUUUU")[0][0])
    long_rows = []
    for qi, qid in enumerate(query_ids):
        for space, mat in profiles.items():
            row_profile = mat[qi]
            order = np.argsort(-row_profile)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            row: dict[str, Any] = {
                "query": short_id(qid),
                "query_id": qid,
                "space": space,
                "top1_kmer": str(kmers[order[0]]),
                "top5_kmers": ",".join(kmers[order[:5]].astype(str)),
                "top20_kmers": ",".join(kmers[order[:20]].astype(str)),
                "UUUUUUU_rank": int(ranks[u7]),
                "UUUUUUU_score": float(row_profile[u7]),
            }
            for fam in ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich"]:
                br, bk, bs = best_rank(row_profile, fam_idx[fam], kmers)
                row[f"best_{fam}_rank"] = br
                row[f"best_{fam}_kmer"] = bk
                row[f"best_{fam}_score"] = bs
            long_rows.append(row)
    long_df = pd.DataFrame(long_rows)
    rows = []
    metrics = ["UUUUUUU_rank", "best_U-rich_rank", "best_CUUCU-like_rank", "best_UCUCUC-like_rank", "best_UGUGUG-like_rank", "best_GA-rich_rank"]
    for (query, qid), sub in long_df.groupby(["query", "query_id"], sort=False):
        a = sub.set_index("space")
        row: dict[str, Any] = {"query": query, "query_id": qid, "cnn_top1_kmer": a.loc["cnn_latent", "top1_kmer"], "jple_top1_kmer": a.loc["jple_latent", "top1_kmer"]}
        for metric in metrics:
            row[f"cnn_{metric}"] = a.loc["cnn_latent", metric]
            row[f"jple_{metric}"] = a.loc["jple_latent", metric]
            row[f"delta_cnn_minus_jple_{metric}"] = a.loc["cnn_latent", metric] - a.loc["jple_latent", metric]
        rows.append(row)
    return pd.DataFrame(rows), long_df


def pca2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(np.float32)


def plot_pca(path: Path, train_ids: list[str], query_ids: list[str], train_family: dict[str, str], cnn_t: np.ndarray, cnn_q: np.ndarray, jple_t: np.ndarray, jple_q: np.ndarray) -> None:
    colors = {"U-rich": "#2ca02c", "CUUCU-like": "#1f77b4", "UCUCUC-like": "#17becf", "UGUGUG-like": "#9467bd", "GA-rich": "#ff7f0e", "mixed/other": "#7f7f7f"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    for ax, title, train_x, query_x in [(axes[0], "CNN latent", cnn_t, cnn_q), (axes[1], "JPLE latent", jple_t, jple_q)]:
        xy = pca2(np.vstack([train_x, query_x]))
        tr, qu = xy[: len(train_ids)], xy[len(train_ids) :]
        for fam in FAMILIES:
            idx = [i for i, pid in enumerate(train_ids) if train_family[pid] == fam]
            if idx:
                ax.scatter(tr[idx, 0], tr[idx, 1], s=12, alpha=0.55, c=colors[fam], label=fam, linewidths=0)
        for i, qid in enumerate(query_ids):
            sid = short_id(qid)
            if sid == "w1":
                ax.scatter(qu[i, 0], qu[i, 1], marker="*", s=240, c="red", edgecolors="black", linewidths=0.7, zorder=5)
                ax.text(qu[i, 0], qu[i, 1], " w1", fontsize=9, weight="bold", color="red")
            else:
                ax.scatter(qu[i, 0], qu[i, 1], marker="X", s=65, c="black", edgecolors="white", linewidths=0.5, zorder=4)
                ax.text(qu[i, 0], qu[i, 1], f" {sid}", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def write_report(out: Path, args: argparse.Namespace, w1_neighbors: pd.DataFrame, motif_shift: pd.DataFrame, ranks: pd.DataFrame) -> None:
    w1_dist = motif_shift[motif_shift["query"] == "w1"].set_index("family")
    w1_rank = ranks[ranks["query"] == "w1"].iloc[0]
    cnn_u = float(w1_dist.loc["U-rich", "cnn_top50_fraction"])
    jple_u = float(w1_dist.loc["U-rich", "jple_top50_fraction"])
    cnn_ga = float(w1_dist.loc["GA-rich", "cnn_top50_fraction"])
    jple_ga = float(w1_dist.loc["GA-rich", "jple_top50_fraction"])
    cnn_u_rank = int(w1_rank["cnn_best_U-rich_rank"])
    jple_u_rank = int(w1_rank["jple_best_U-rich_rank"])
    verdict = "w1 的 U-rich 信息在 CNN latent 和 exact JPLE latent 两个空间都不强，更像是输入 RBD embedding/训练集邻居结构本身没有把 w1 放进 U-rich 邻域。"
    if jple_u >= 0.25 and cnn_u < 0.15:
        verdict = "exact JPLE latent 比 CNN latent 更接近 U-rich 邻域，主要问题偏向 CNN encoder latent shift。"
    elif cnn_u >= 0.25 and cnn_u_rank > 20:
        verdict = "CNN latent 邻域已有 U-rich 支持但 decoder 后 rank 仍差，主要问题偏向 JPLE neighbor decoder/权重重建。"
    elif cnn_u_rank <= 20 and jple_u_rank > 20:
        verdict = "CNN latent decoder 比 exact JPLE projection 更能恢复 w1 U-rich，问题不支持归因到 CNN encoder 丢失。"
    elif jple_u_rank <= 20 and cnn_u_rank > 20:
        verdict = "exact JPLE projection 比 CNN latent 更能恢复 w1 U-rich，问题偏向 CNN encoder 输出发生 latent shift。"

    lines = [
        "# Diagnose 07 CNN vs JPLE Latent Shift",
        "",
        "## w1 Key Shift",
        "",
        "| metric | CNN latent | JPLE latent |",
        "|---|---:|---:|",
        f"| top50 U-rich neighbor fraction | {cnn_u:.3f} | {jple_u:.3f} |",
        f"| top50 GA-rich neighbor fraction | {cnn_ga:.3f} | {jple_ga:.3f} |",
        f"| best U-rich rank after decoder | {cnn_u_rank} | {jple_u_rank} |",
        f"| exact UUUUUUU rank after decoder | {int(w1_rank['cnn_UUUUUUU_rank'])} | {int(w1_rank['jple_UUUUUUU_rank'])} |",
        f"| top1 kmer after decoder | {w1_rank['cnn_top1_kmer']} | {w1_rank['jple_top1_kmer']} |",
        "",
        "## Interpretation",
        "",
        verdict,
        "",
        "## All Query Rank Shift",
        "",
        ranks[["query", "cnn_top1_kmer", "jple_top1_kmer", "cnn_best_U-rich_rank", "jple_best_U-rich_rank", "cnn_best_CUUCU-like_rank", "jple_best_CUUCU-like_rank", "cnn_best_UGUGUG-like_rank", "jple_best_UGUGUG-like_rank", "cnn_best_GA-rich_rank", "jple_best_GA-rich_rank"]].to_markdown(index=False),
        "",
        "## w1 Top10 Neighbors",
        "",
        w1_neighbors[w1_neighbors["neighbor_rank"] <= 10][["space", "neighbor_rank", "train_protein_id", "cosine_distance", "neighbor_family", "neighbor_true_top1_kmer"]].to_markdown(index=False),
        "",
        "## Output Files",
        "",
        "- `w1_neighbor_shift.tsv`",
        "- `motif_distribution_shift.tsv`",
        "- `rank_shift_table.tsv`",
        "- `pca_projection.png`",
        "- `all_query_shift_summary.tsv`",
        "- `all_query_top50_neighbors.tsv`",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    parser.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    parser.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    parser.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    parser.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    parser.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    parser.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_07_cnn_vs_jple_latent_shift")
    parser.add_argument("--num-eigenvector", type=int, default=122)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--std", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    args = parser.parse_args()

    setup_threads(args.torch_num_threads)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    log("loading embeddings and motifs")
    motif_ids, y_raw, kmers = load_motif_npz(resolve(args.motif_npz))
    y_all = row_l2_normalize(y_raw)
    train_map, _ = load_h5_features(resolve(args.train_per_residue_h5), None)
    rice_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    at_map, _ = load_h5_features(resolve(args.query_atptbp3_per_residue_h5), None)
    query_map = dict(rice_map)
    query_map.update(at_map)

    anchor = np.load(resolve(args.jple_anchor_npz), allow_pickle=True)
    anchor_ids = np.asarray(anchor["train_protein_id_list"]).astype(str).tolist()
    anchor_w = np.asarray(anchor["w_train"], dtype=np.float32)
    motif_index = {pid: i for i, pid in enumerate(motif_ids)}
    keep = [i for i, pid in enumerate(anchor_ids) if pid in train_map and pid in motif_index]
    train_ids = [anchor_ids[i] for i in keep]
    cnn_train_anchor = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    query_ids = sorted(query_map.keys(), key=short_id)
    log(f"train_n={len(train_ids)} query_n={len(query_ids)}")

    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    train_top1 = {pid: str(kmers[int(np.argmax(y_train[i]))]) for i, pid in enumerate(train_ids)}

    log("extracting frozen CNN query latents")
    model = load_cnn(resolve(args.cnn_checkpoint), device)
    cnn_query = cnn_latent(model, query_ids, query_map, device, args.batch_size)

    log("fitting exact JPLE latent projection from mean-pooled per-residue RBD embeddings")
    x_train = l2_normalize(mean_pool(train_ids, train_map))
    x_query = l2_normalize(mean_pool(query_ids, query_map))
    jple = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    jple.fit(x_train, y_train)
    jple_train = jple.w_train.astype(np.float32)
    jple_query = jple_project(jple, x_query)

    log("computing top50 neighbors")
    all_neighbors = pd.concat(
        [
            nearest_table(cnn_query, cnn_train_anchor, query_ids, train_ids, train_family, train_top1, "cnn_latent", args.top_n),
            nearest_table(jple_query, jple_train, query_ids, train_ids, train_family, train_top1, "jple_latent", args.top_n),
        ],
        ignore_index=True,
    )
    all_neighbors.to_csv(out / "all_query_top50_neighbors.tsv", sep="\t", index=False)
    w1_neighbors = all_neighbors[all_neighbors["query"] == "w1"].copy()
    w1_neighbors.to_csv(out / "w1_neighbor_shift.tsv", sep="\t", index=False)

    dist = family_distribution(all_neighbors)
    motif_shift, summary = distribution_shift(dist)
    motif_shift.to_csv(out / "motif_distribution_shift.tsv", sep="\t", index=False)

    log("decoding profiles from both latent spaces")
    cnn_profiles, cnn_decoder = decode(cnn_query, cnn_train_anchor, y_train, query_ids, train_ids, "cnn_latent", args.threshold, args.std)
    jple_profiles, jple_decoder = decode(jple_query, jple_train, y_train, query_ids, train_ids, "jple_latent", args.threshold, args.std)
    pd.concat([cnn_decoder, jple_decoder], ignore_index=True).to_csv(out / "decoder_neighbor_weights.tsv", sep="\t", index=False)
    ranks, rank_long = rank_shift(query_ids, {"cnn_latent": cnn_profiles, "jple_latent": jple_profiles}, kmers, family_indices(kmers))
    ranks.to_csv(out / "rank_shift_table.tsv", sep="\t", index=False)
    rank_long.to_csv(out / "decoded_profile_rank_long.tsv", sep="\t", index=False)

    summary.merge(ranks, on=["query", "query_id"], how="left").to_csv(out / "all_query_shift_summary.tsv", sep="\t", index=False)
    plot_pca(out / "pca_projection.png", train_ids, query_ids, train_family, cnn_train_anchor, cnn_query, jple_train, jple_query)
    write_report(out, args, w1_neighbors, motif_shift, ranks)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
