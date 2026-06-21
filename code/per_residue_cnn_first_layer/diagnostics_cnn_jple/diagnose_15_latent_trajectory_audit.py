#!/usr/bin/env python3
"""Diagnose 15: latent trajectory audit for baseline, B, E, and B+E.

This diagnostic does not train or modify any model. It compares where w1 lands
in the frozen JPLE retrieval space under four existing branches:

baseline current CNN, B cosine-aligned encoder, E residual correction, and
B+E cosine encoder plus residual correction.
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
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from cnn_model_utils import PerResidueCnn, load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize  # noqa: E402
from diagnose_07_cnn_vs_jple_latent_shift import load_cnn, resolve, short_id  # noqa: E402
from diagnose_14_encoder_alignment_benchmark import (  # noqa: E402
    ResidualHead,
    decode_threshold,
    exact_kmer_rank,
    family_indices,
    predict_encoder,
    summarize_queries,
)


def log(msg: str) -> None:
    print(f"[diagnose-15] {msg}", flush=True)


def load_trained_encoder(path: Path, input_dim: int, hidden_dim: int, out_dim: int, kernel_size: int, num_blocks: int, dropout: float, device: torch.device) -> PerResidueCnn:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PerResidueCnn(input_dim, hidden_dim, out_dim, kernel_size, num_blocks, dropout).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def apply_residual_head(path: Path, z: np.ndarray, device: torch.device) -> np.ndarray:
    head = ResidualHead(z.shape[1]).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    head.load_state_dict(state)
    head.eval()
    rows = []
    bs = 256
    with torch.no_grad():
        for start in range(0, len(z), bs):
            x = torch.from_numpy(z[start : start + bs].astype(np.float32)).to(device)
            rows.append(head(x).detach().cpu().numpy().astype(np.float32))
    return np.vstack(rows)


def cosine_distance_to(v: np.ndarray, centroid: np.ndarray) -> float:
    return float(cdist(v[None, :], centroid[None, :], "cosine")[0, 0])


def euclidean_distance_to(v: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.linalg.norm(v - centroid))


def family_comp(top_idx: np.ndarray, train_ids: list[str], train_family: dict[str, str]) -> dict[str, float]:
    families = ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich", "mixed/other"]
    vals = {fam: 0 for fam in families}
    for idx in top_idx:
        vals[train_family[train_ids[int(idx)]]] += 1
    return {fam: vals[fam] / float(len(top_idx)) for fam in families}


def rank_of_train_id(dist_row: np.ndarray, train_ids: list[str], protein_id: str) -> int | None:
    if protein_id not in train_ids:
        return None
    order = np.argsort(dist_row)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return int(ranks[train_ids.index(protein_id)])


def centroid_from_ids(ids: list[str], train_ids: list[str], z_true: np.ndarray) -> np.ndarray:
    idx = [train_ids.index(pid) for pid in ids if pid in train_ids]
    if not idx:
        raise ValueError("No centroid ids found in training ids")
    return z_true[np.asarray(idx, dtype=int)].mean(axis=0)


def plot_pca(
    out_path: Path,
    z_true: np.ndarray,
    train_ids: list[str],
    train_family: dict[str, str],
    w1_points: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
) -> None:
    labels = list(w1_points.keys()) + list(centroids.keys())
    extra = np.vstack([*w1_points.values(), *centroids.values()])
    pca = PCA(n_components=2, random_state=0)
    xy_train = pca.fit_transform(np.vstack([z_true, extra]))[: len(z_true)]
    xy_extra = pca.transform(extra)

    colors = {
        "U-rich": "#2ca02c",
        "CUUCU-like": "#1f77b4",
        "UCUCUC-like": "#17becf",
        "UGUGUG-like": "#9467bd",
        "GA-rich": "#ff7f0e",
        "mixed/other": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    for fam, color in colors.items():
        idx = [i for i, pid in enumerate(train_ids) if train_family[pid] == fam]
        if idx:
            ax.scatter(xy_train[idx, 0], xy_train[idx, 1], s=12, alpha=0.28, c=color, label=fam, linewidths=0)

    n_w1 = len(w1_points)
    xy_w1 = xy_extra[:n_w1]
    xy_cent = xy_extra[n_w1:]
    for i, label in enumerate(w1_points):
        ax.scatter(xy_w1[i, 0], xy_w1[i, 1], s=95, marker="*", c="black", edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(xy_w1[i, 0], xy_w1[i, 1], f" {label}", fontsize=9, weight="bold")
    for i, label in enumerate(centroids):
        ax.scatter(xy_cent[i, 0], xy_cent[i, 1], s=70, marker="X", c="#d62728", edgecolors="white", linewidths=0.8, zorder=4)
        ax.text(xy_cent[i, 0], xy_cent[i, 1], f" {label}", fontsize=9)

    trajectory = ["baseline", "B_cos_0.5", "E_residual", "BplusE"]
    for a, b in zip(trajectory[:-1], trajectory[1:]):
        if a in w1_points and b in w1_points:
            ia, ib = list(w1_points.keys()).index(a), list(w1_points.keys()).index(b)
            ax.annotate("", xy=xy_w1[ib], xytext=xy_w1[ia], arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "black", "alpha": 0.75})

    ax.set_title("w1 latent trajectory in JPLE retrieval space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_umap(
    out_path: Path,
    z_true: np.ndarray,
    train_ids: list[str],
    train_family: dict[str, str],
    w1_points: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
) -> bool:
    try:
        import umap  # type: ignore
    except Exception:
        return False
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="cosine", random_state=0)
    labels = list(w1_points.keys()) + list(centroids.keys())
    extra = np.vstack([*w1_points.values(), *centroids.values()])
    xy = reducer.fit_transform(np.vstack([z_true, extra]))
    xy_train = xy[: len(z_true)]
    xy_extra = xy[len(z_true) :]
    colors = {
        "U-rich": "#2ca02c",
        "CUUCU-like": "#1f77b4",
        "UCUCUC-like": "#17becf",
        "UGUGUG-like": "#9467bd",
        "GA-rich": "#ff7f0e",
        "mixed/other": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    for fam, color in colors.items():
        idx = [i for i, pid in enumerate(train_ids) if train_family[pid] == fam]
        if idx:
            ax.scatter(xy_train[idx, 0], xy_train[idx, 1], s=12, alpha=0.28, c=color, label=fam, linewidths=0)
    n_w1 = len(w1_points)
    xy_w1 = xy_extra[:n_w1]
    xy_cent = xy_extra[n_w1:]
    for i, label in enumerate(list(w1_points.keys())):
        ax.scatter(xy_w1[i, 0], xy_w1[i, 1], s=95, marker="*", c="black", edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(xy_w1[i, 0], xy_w1[i, 1], f" {label}", fontsize=9, weight="bold")
    for i, label in enumerate(list(centroids.keys())):
        ax.scatter(xy_cent[i, 0], xy_cent[i, 1], s=70, marker="X", c="#d62728", edgecolors="white", linewidths=0.8, zorder=4)
        ax.text(xy_cent[i, 0], xy_cent[i, 1], f" {label}", fontsize=9)
    trajectory = ["baseline", "B_cos_0.5", "E_residual", "BplusE"]
    for a, b in zip(trajectory[:-1], trajectory[1:]):
        if a in w1_points and b in w1_points:
            ia, ib = list(w1_points.keys()).index(a), list(w1_points.keys()).index(b)
            ax.annotate("", xy=xy_w1[ib], xytext=xy_w1[ia], arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "black", "alpha": 0.75})
    ax.set_title("w1 latent trajectory UMAP")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--diagnose10-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_10_neighbor_loss_audit")
    p.add_argument("--diagnose14-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_14_encoder_alignment_benchmark")
    p.add_argument("--bpluse-dir", default="results/per_residue_cnn_first_layer/cnn_jple_BplusE_20260617")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_15_latent_trajectory_audit")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--decoder-threshold", type=float, default=0.01)
    p.add_argument("--decoder-std", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.2)
    p.add_argument("--torch-num-threads", type=int, default=1)
    args = p.parse_args()

    setup_threads(args.torch_num_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if args.device == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)

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
    query_ids = sorted(query_map.keys(), key=short_id)
    w1_query_id = next(qid for qid in query_ids if short_id(qid) == "w1")

    anchor = np.load(resolve(args.jple_anchor_npz), allow_pickle=True)
    anchor_ids = np.asarray(anchor["train_protein_id_list"]).astype(str).tolist()
    anchor_w = np.asarray(anchor["w_train"], dtype=np.float32)
    keep = [i for i, pid in enumerate(anchor_ids) if pid in train_map and pid in motif_index]
    train_ids = [anchor_ids[i] for i in keep]
    z_true = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    train_family_list = [train_family[pid] for pid in train_ids]
    fam_idx = family_indices(kmers)

    log("loading existing branch latents")
    baseline_model = load_cnn(resolve(args.cnn_checkpoint), device)
    z_base_train = predict_encoder(baseline_model, train_ids, train_map, device, args.batch_size)
    z_base_query_all = predict_encoder(baseline_model, query_ids, query_map, device, args.batch_size)

    b_encoder = load_trained_encoder(
        resolve(args.diagnose14_dir) / "B_cos_0.5" / "encoder.pt",
        1152,
        args.hidden_dim,
        z_true.shape[1],
        args.kernel_size,
        args.num_blocks,
        args.dropout,
        device,
    )
    z_b_train = predict_encoder(b_encoder, train_ids, train_map, device, args.batch_size)
    z_b_query_all = predict_encoder(b_encoder, query_ids, query_map, device, args.batch_size)

    z_e_train = apply_residual_head(resolve(args.diagnose14_dir) / "E_residual_head" / "residual_head.pt", z_base_train, device)
    z_e_query_all = apply_residual_head(resolve(args.diagnose14_dir) / "E_residual_head" / "residual_head.pt", z_base_query_all, device)

    bpe_encoder = load_trained_encoder(
        resolve(args.bpluse_dir) / "B_encoder_cos0.5" / "encoder.pt",
        1152,
        args.hidden_dim,
        z_true.shape[1],
        args.kernel_size,
        args.num_blocks,
        args.dropout,
        device,
    )
    z_bpe_train_raw = predict_encoder(bpe_encoder, train_ids, train_map, device, args.batch_size)
    z_bpe_query_raw = predict_encoder(bpe_encoder, query_ids, query_map, device, args.batch_size)
    z_bpe_train = apply_residual_head(resolve(args.bpluse_dir) / "E_residual_head_on_B" / "residual_head.pt", z_bpe_train_raw, device)
    z_bpe_query_all = apply_residual_head(resolve(args.bpluse_dir) / "E_residual_head_on_B" / "residual_head.pt", z_bpe_query_raw, device)

    strategies = {
        "baseline": (z_base_train, z_base_query_all),
        "B_cos_0.5": (z_b_train, z_b_query_all),
        "E_residual": (z_e_train, z_e_query_all),
        "BplusE": (z_bpe_train, z_bpe_query_all),
    }

    log("building local centroids")
    diag10 = resolve(args.diagnose10_dir)
    lost = pd.read_csv(diag10 / "w1_lost_neighbors.tsv", sep="\t")
    gained = pd.read_csv(diag10 / "w1_gained_neighbors.tsv", sep="\t")
    lost_u = lost[lost["motif_family"] == "U-rich"]["protein_id"].astype(str).tolist()
    gained_mixed = gained[gained["motif_family"] == "mixed/other"]["protein_id"].astype(str).tolist()
    gained_ga = gained[gained["motif_family"] == "GA-rich"]["protein_id"].astype(str).tolist()
    global_ga = [pid for pid in train_ids if train_family[pid] == "GA-rich"]
    global_mixed = [pid for pid in train_ids if train_family[pid] == "mixed/other"]
    centroids = {
        "lost_U-rich_centroid": centroid_from_ids(lost_u, train_ids, z_true),
        "gained_mixed_centroid": centroid_from_ids(gained_mixed, train_ids, z_true) if gained_mixed else centroid_from_ids(global_mixed, train_ids, z_true),
        "gained_or_global_GA_centroid": centroid_from_ids(gained_ga, train_ids, z_true) if gained_ga else centroid_from_ids(global_ga, train_ids, z_true),
    }

    rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    q_rows: list[pd.DataFrame] = []
    w1_idx = query_ids.index(w1_query_id)
    w1_points: dict[str, np.ndarray] = {}
    for strategy, (z_train_strategy, z_query_all) in strategies.items():
        z_w1 = z_query_all[w1_idx]
        w1_points[strategy] = z_w1
        dist = cdist(z_w1[None, :], z_true, "cosine")[0]
        top50 = np.argsort(dist)[:50]
        comp = family_comp(top50, train_ids, train_family)
        pred_query_profiles = decode_threshold(z_query_all, z_true, y_train, args.decoder_threshold, args.decoder_std, exclude_self=False)
        qdf = summarize_queries(strategy, query_ids, pred_query_profiles, kmers, fam_idx, z_query_all, z_true, train_ids, train_family)
        q_rows.append(qdf)
        w1_q = qdf[qdf["query"] == "w1"].iloc[0].to_dict()
        row: dict[str, Any] = {
            "strategy": strategy,
            "w1_top1": w1_q["top1_motif"],
            "w1_top5": w1_q["top5_motifs"],
            "w1_U-rich_rank": int(w1_q["U-rich_rank"]),
            "w1_UUUUUUU_rank": int(w1_q["UUUUUUU_rank"]),
            "w1_CUUCU_like_rank": int(w1_q["CUUCU-like_rank"]),
            "w1_UGUGUG_like_rank": int(w1_q["UGUGUG-like_rank"]),
            "w1_GA_rich_rank": int(w1_q["GA-rich_rank"]),
            "RNCMPT00434_rank": rank_of_train_id(dist, train_ids, "RNCMPT00434"),
            "top50_U-rich_fraction": comp["U-rich"],
            "top50_mixed_fraction": comp["mixed/other"],
            "top50_GA-rich_fraction": comp["GA-rich"],
        }
        for cname, centroid in centroids.items():
            row[f"cosdist_to_{cname}"] = cosine_distance_to(z_w1, centroid)
            row[f"euclid_to_{cname}"] = euclidean_distance_to(z_w1, centroid)
        rows.append(row)
        for fam, frac in comp.items():
            comp_rows.append({"strategy": strategy, "family": fam, "top50_fraction": frac})

    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(out / "w1_latent_trajectory_summary.tsv", sep="\t", index=False)
    pd.DataFrame(comp_rows).to_csv(out / "w1_top50_family_composition.tsv", sep="\t", index=False)
    pd.concat(q_rows, ignore_index=True).to_csv(out / "query_predictions_by_strategy.tsv", sep="\t", index=False)

    centroid_rows = []
    for name, ids in [
        ("lost_U-rich_centroid", lost_u),
        ("gained_mixed_centroid", gained_mixed),
        ("gained_or_global_GA_centroid", gained_ga if gained_ga else global_ga),
    ]:
        centroid_rows.append({"centroid": name, "n": len(ids), "protein_ids": ",".join(ids)})
    pd.DataFrame(centroid_rows).to_csv(out / "centroid_definitions.tsv", sep="\t", index=False)

    log("plotting trajectory")
    plot_pca(out / "w1_latent_trajectory_pca.png", z_true, train_ids, train_family, w1_points, centroids)
    umap_ok = plot_umap(out / "w1_latent_trajectory_umap.png", z_true, train_ids, train_family, w1_points, centroids)

    b = trajectory.set_index("strategy")
    interpretation = []
    interpretation.append("# Diagnose 15 Latent Trajectory Audit")
    interpretation.append("")
    interpretation.append("This diagnostic compares w1 latent placement under baseline, B, E, and B+E. No model was trained or modified.")
    interpretation.append("")
    interpretation.append("## w1 Summary")
    interpretation.append("")
    interpretation.append(trajectory.to_markdown(index=False))
    interpretation.append("")
    interpretation.append("## Key Interpretation")
    interpretation.append("")
    b_u = int(b.loc["B_cos_0.5", "w1_U-rich_rank"])
    e_u = int(b.loc["E_residual", "w1_U-rich_rank"])
    bpe_u = int(b.loc["BplusE", "w1_U-rich_rank"])
    base_u = int(b.loc["baseline", "w1_U-rich_rank"])
    interpretation.append(f"- Baseline w1 U-rich rank = {base_u}; B_cos_0.5 = {b_u}; E_residual = {e_u}; B+E = {bpe_u}.")
    interpretation.append(f"- B cosine alignment moves w1 toward a usable U-rich solution if its U-rich rank is <=20: `{b_u <= 20}`.")
    interpretation.append(f"- B+E preserves/restores RNCMPT00434 if rank<50: `{int(b.loc['BplusE', 'RNCMPT00434_rank']) < 50}`; observed rank = {int(b.loc['BplusE', 'RNCMPT00434_rank'])}.")
    if bpe_u > max(base_u, b_u, e_u):
        interpretation.append("- B+E pushes the decoded w1 motif away from U-rich despite good RNAcompete-level reconstruction, consistent with residual-head OOD extrapolation.")
    if float(b.loc["B_cos_0.5", "cosdist_to_lost_U-rich_centroid"]) < float(b.loc["baseline", "cosdist_to_lost_U-rich_centroid"]):
        interpretation.append("- B reduces cosine distance from w1 to the lost U-rich centroid.")
    if float(b.loc["BplusE", "cosdist_to_gained_mixed_centroid"]) < float(b.loc["B_cos_0.5", "cosdist_to_gained_mixed_centroid"]):
        interpretation.append("- B+E is closer to the gained mixed centroid than B, supporting the hypothesis that residual correction bends w1 toward the wrong local region.")
    interpretation.append("")
    interpretation.append("## Files")
    interpretation.append("")
    interpretation.append("- `w1_latent_trajectory_summary.tsv`")
    interpretation.append("- `w1_top50_family_composition.tsv`")
    interpretation.append("- `query_predictions_by_strategy.tsv`")
    interpretation.append("- `centroid_definitions.tsv`")
    interpretation.append("- `w1_latent_trajectory_pca.png`")
    if umap_ok:
        interpretation.append("- `w1_latent_trajectory_umap.png`")
    else:
        interpretation.append("- UMAP was unavailable in this environment; PCA was generated.")
    (out / "diagnose_15_latent_trajectory_audit_report.md").write_text("\n".join(interpretation) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
