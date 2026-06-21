#!/usr/bin/env python3
"""Local latent intervention test for w1 CNN+JPLE failure.

No training, no checkpoint edits, no decoder parameter changes. The only
intervention is moving w1's CNN latent toward lost U-rich neighbor centroids.
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

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from diagnose_07_cnn_vs_jple_latent_shift import (  # noqa: E402
    cnn_latent,
    kmer_family,
    load_cnn,
    resolve,
    short_id,
)
from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize, standardize_rows  # noqa: E402


ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
TRACK_KMERS = ["UUUUUUU", "CUUUUUU", "UUUUUUG"]


def log(msg: str) -> None:
    print(f"[diagnose-11] {msg}", flush=True)


def decode_latents(z: np.ndarray, train_w: np.ndarray, y_train: np.ndarray, threshold: float, std: float) -> tuple[np.ndarray, pd.DataFrame]:
    dist = cdist(z, train_w, "cosine")
    sim = np.exp(-(dist**2) / (std**2))
    preds = []
    rows = []
    for qi in range(z.shape[0]):
        idx = np.argwhere(sim[qi] >= threshold).flatten()
        if len(idx) == 0:
            idx = np.asarray([int(np.argmax(sim[qi]))])
        idx = idx[np.argsort(-sim[qi, idx])]
        w = sim[qi, idx]
        w = w / w.sum()
        preds.append(np.sum(w[:, None] * y_train[idx], axis=0))
        for rank, (ti, weight) in enumerate(zip(idx, w), start=1):
            rows.append({"point_idx": qi, "decoder_neighbor_rank": rank, "train_idx": int(ti), "decoder_weight": float(weight), "distance": float(dist[qi, ti])})
    return standardize_rows(np.asarray(preds, dtype=np.float32)), pd.DataFrame(rows)


def family_indices(kmers: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {"U-rich": [], "CUUCU-like": [], "UCUCUC-like": [], "UGUGUG-like": [], "GA-rich": [], "mixed/other": []}
    for i, kmer in enumerate(kmers.astype(str)):
        fam = kmer_family(kmer)
        out.setdefault(fam, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def best_rank(profile: np.ndarray, idx: np.ndarray, kmers: np.ndarray) -> tuple[int, str, float]:
    order = np.argsort(-profile)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    best = idx[np.argmin(ranks[idx])]
    return int(ranks[best]), str(kmers[best]), float(profile[best])


def exact_kmer_rank(profile: np.ndarray, kmer: str, kmers: np.ndarray) -> tuple[int, float]:
    idx = int(np.where(kmers.astype(str) == kmer)[0][0])
    order = np.argsort(-profile)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return int(ranks[idx]), float(profile[idx])


def neighbor_stats(z: np.ndarray, train_w: np.ndarray, train_ids: list[str], train_family: dict[str, str], top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist = cdist(z, train_w, "cosine")
    rows = []
    neigh_rows = []
    for i in range(z.shape[0]):
        order = np.argsort(dist[i])
        top = order[:top_n]
        fams = [train_family[train_ids[int(j)]] for j in top]
        counts = pd.Series(fams).value_counts()
        majority = str(counts.index[0])
        majority_frac = float(counts.iloc[0] / top_n)
        second = str(counts.index[1]) if len(counts) > 1 else ""
        second_frac = float(counts.iloc[1] / top_n) if len(counts) > 1 else 0.0
        row: dict[str, Any] = {
            "point_idx": i,
            "top50_U-rich_fraction": float(fams.count("U-rich") / top_n),
            "top50_GA-rich_fraction": float(fams.count("GA-rich") / top_n),
            "top50_UGUGUG-like_fraction": float(fams.count("UGUGUG-like") / top_n),
            "top50_mixed_fraction": float(fams.count("mixed/other") / top_n),
            "majority_family": majority,
            "majority_fraction": majority_frac,
            "second_majority_family": second,
            "second_majority_fraction": second_frac,
            "support_score": majority_frac - second_frac,
            "top1_neighbor_id": train_ids[int(order[0])],
            "top1_neighbor_family": train_family[train_ids[int(order[0])]],
            "top1_neighbor_distance": float(dist[i, order[0]]),
        }
        rows.append(row)
        for rank, j in enumerate(top, start=1):
            pid = train_ids[int(j)]
            neigh_rows.append({"point_idx": i, "neighbor_rank": rank, "protein_id": pid, "distance": float(dist[i, j]), "motif_family": train_family[pid]})
    return pd.DataFrame(rows), pd.DataFrame(neigh_rows)


def rank_training_neighbors(z: np.ndarray, train_w: np.ndarray, train_ids: list[str], tracked_ids: list[str]) -> pd.DataFrame:
    dist = cdist(z, train_w, "cosine")
    rows = []
    id_to_idx = {pid: i for i, pid in enumerate(train_ids)}
    for pi in range(z.shape[0]):
        order = np.argsort(dist[pi])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for pid in tracked_ids:
            idx = id_to_idx[pid]
            rows.append({"point_idx": pi, "protein_id": pid, "rank": int(ranks[idx]), "distance": float(dist[pi, idx]), "is_top50": bool(ranks[idx] <= 50)})
    return pd.DataFrame(rows)


def make_report(out: Path, rank_curve: pd.DataFrame, neighbor_curve: pd.DataFrame, thresholds: pd.DataFrame, candidate_curve: pd.DataFrame) -> None:
    best = thresholds.sort_values(["centroid_name", "criterion"]).copy()
    lines = ["# Diagnose 11 Local Latent Intervention", ""]
    lines += ["## Recovery Thresholds", "", best.to_markdown(index=False), ""]
    baseline = rank_curve[rank_curve["alpha"] == 0.0].sort_values("centroid_name").head(1).iloc[0]
    lines += [
        "## Baseline CNN Latent",
        "",
        f"- top1 motif: {baseline['top1_motif']}",
        f"- U-rich best rank: {int(baseline['U-rich_best_rank'])}",
        f"- UUUUUUU rank: {int(baseline['UUUUUUU_rank'])}",
        f"- CUUUUUU rank: {int(baseline['CUUUUUU_rank'])}",
        "",
    ]
    for centroid, sub in rank_curve.groupby("centroid_name", sort=False):
        top20 = sub[sub["U-rich_best_rank"] <= 20]
        top10 = sub[sub["U-rich_best_rank"] <= 10]
        top1 = sub[sub["top1_is_U-rich"]]
        nb = neighbor_curve[neighbor_curve["centroid_name"] == centroid]
        lines += [f"## {centroid}", ""]
        lines += [
            f"- min alpha for U-rich top20: {float(top20['alpha'].min()) if len(top20) else 'not_recovered'}",
            f"- min alpha for U-rich top10: {float(top10['alpha'].min()) if len(top10) else 'not_recovered'}",
            f"- min alpha for U-rich top1: {float(top1['alpha'].min()) if len(top1) else 'not_recovered'}",
            f"- U-rich fraction at alpha=0: {float(nb[nb['alpha'] == 0.0]['top50_U-rich_fraction'].iloc[0]):.3f}",
            f"- U-rich fraction at alpha=0.2: {float(nb[nb['alpha'] == 0.2]['top50_U-rich_fraction'].iloc[0]):.3f}",
            f"- U-rich fraction at alpha=1: {float(nb[nb['alpha'] == 1.0]['top50_U-rich_fraction'].iloc[0]):.3f}",
            "",
        ]
    rn = candidate_curve[candidate_curve["protein_id"] == "RNCMPT00434"].sort_values(["centroid_name", "alpha"])
    lines += ["## RNCMPT00434 Recovery", "", rn.to_markdown(index=False), ""]
    early = thresholds[(thresholds["criterion"] == "U-rich_best_rank_le20") & (thresholds["min_alpha"].astype(str) != "not_recovered")]
    early_ok = len(early) and float(pd.to_numeric(early["min_alpha"]).min()) <= 0.2
    if early_ok:
        verdict = "A small latent move is sufficient to recover U-rich ranking, supporting CNN latent placement error."
    else:
        verdict = "Recovery requires a large move or fails, supporting JPLE manifold/training support limitation."
    lines += [
        "## Interpretation",
        "",
        verdict,
        "If future model changes are allowed, the highest-value target is the encoder/retrieval geometry, not decoder parameters.",
        "",
        "## Output Files",
        "",
        "- `urich_centroids.tsv`",
        "- `w1_intervention_rank_curve.tsv`",
        "- `w1_intervention_neighbor_curve.tsv`",
        "- `recovery_threshold.tsv`",
        "- `candidate_neighbor_recovery.tsv`",
    ]
    (out / "diagnose_11_local_latent_intervention_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--lost-urich-tsv", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_10_neighbor_loss_audit/lost_u_rich_neighbor_summary.tsv")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_11_local_latent_intervention")
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

    anchor = np.load(resolve(args.jple_anchor_npz), allow_pickle=True)
    anchor_ids = np.asarray(anchor["train_protein_id_list"]).astype(str).tolist()
    anchor_w = np.asarray(anchor["w_train"], dtype=np.float32)
    keep = [i for i, pid in enumerate(anchor_ids) if pid in train_map and pid in motif_index]
    train_ids = [anchor_ids[i] for i in keep]
    train_w = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    id_to_idx = {pid: i for i, pid in enumerate(train_ids)}

    lost_u = pd.read_csv(resolve(args.lost_urich_tsv), sep="\t").sort_values("rank_exact")
    centroid_specs = {
        "lost_urich_all20": lost_u["protein_id"].astype(str).tolist(),
        "lost_urich_top10": lost_u.head(10)["protein_id"].astype(str).tolist(),
        "lost_urich_top5": lost_u.head(5)["protein_id"].astype(str).tolist(),
    }
    centroid_rows = []
    centroids = {}
    for name, pids in centroid_specs.items():
        z = train_w[[id_to_idx[pid] for pid in pids]].mean(axis=0).astype(np.float32)
        centroids[name] = z
        row: dict[str, Any] = {"centroid_name": name, "n_neighbors": len(pids), "protein_ids": ",".join(pids)}
        row.update({f"z_{i}": float(v) for i, v in enumerate(z)})
        centroid_rows.append(row)
    pd.DataFrame(centroid_rows).to_csv(out / "urich_centroids.tsv", sep="\t", index=False)

    log("computing w1 CNN latent and interpolation trajectories")
    model = load_cnn(resolve(args.cnn_checkpoint), device)
    z_cnn = cnn_latent(model, [w1_id], query_map, device, args.batch_size)[0].astype(np.float32)
    point_rows = []
    point_meta = []
    for cname, z_u in centroids.items():
        for a in ALPHAS:
            z = ((1.0 - a) * z_cnn + a * z_u).astype(np.float32)
            point_meta.append({"point_idx": len(point_rows), "centroid_name": cname, "alpha": a})
            point_rows.append(z)
    z_points = np.vstack(point_rows).astype(np.float32)
    meta = pd.DataFrame(point_meta)

    log("decoding intervention points")
    profiles, _ = decode_latents(z_points, train_w, y_train, args.threshold, args.std)
    fam_idx = family_indices(kmers)
    rank_rows = []
    for i, profile in enumerate(profiles):
        m = meta.iloc[i].to_dict()
        order = np.argsort(-profile)
        top5 = kmers[order[:5]].astype(str).tolist()
        ur_rank, ur_kmer, ur_score = best_rank(profile, fam_idx["U-rich"], kmers)
        row: dict[str, Any] = {
            **m,
            "top1_motif": top5[0],
            "top1_family": kmer_family(top5[0]),
            "top1_is_U-rich": bool(kmer_family(top5[0]) == "U-rich"),
            "top5_motifs": ",".join(top5),
            "U-rich_best_rank": ur_rank,
            "U-rich_best_kmer": ur_kmer,
            "U-rich_best_score": ur_score,
        }
        for kmer in TRACK_KMERS:
            r, s = exact_kmer_rank(profile, kmer, kmers)
            row[f"{kmer}_rank"] = r
            row[f"{kmer}_score"] = s
        rank_rows.append(row)
    rank_curve = pd.DataFrame(rank_rows)
    rank_curve.to_csv(out / "w1_intervention_rank_curve.tsv", sep="\t", index=False)

    log("computing neighbor recovery")
    neigh_stats, _ = neighbor_stats(z_points, train_w, train_ids, train_family, 50)
    neighbor_curve = meta.merge(neigh_stats, on="point_idx", how="left")
    neighbor_curve.to_csv(out / "w1_intervention_neighbor_curve.tsv", sep="\t", index=False)

    thresholds = []
    for cname, sub in rank_curve.groupby("centroid_name", sort=False):
        criteria = {
            "U-rich_best_rank_le20": sub[sub["U-rich_best_rank"] <= 20],
            "U-rich_best_rank_le10": sub[sub["U-rich_best_rank"] <= 10],
            "top1_is_U-rich": sub[sub["top1_is_U-rich"]],
        }
        for crit, hit in criteria.items():
            thresholds.append({"centroid_name": cname, "criterion": crit, "min_alpha": float(hit["alpha"].min()) if len(hit) else "not_recovered"})
    threshold_df = pd.DataFrame(thresholds)
    threshold_df.to_csv(out / "recovery_threshold.tsv", sep="\t", index=False)

    tracked = lost_u["protein_id"].astype(str).tolist()
    if "RNCMPT00434" not in tracked:
        tracked.append("RNCMPT00434")
    cand = rank_training_neighbors(z_points, train_w, train_ids, tracked).merge(meta, on="point_idx", how="left")
    cand = cand.merge(lost_u[["protein_id", "rank_exact", "rank_cnn", "top_motif", "domain_architecture"]], on="protein_id", how="left")
    cand.to_csv(out / "candidate_neighbor_recovery.tsv", sep="\t", index=False)

    make_report(out, rank_curve, neighbor_curve, threshold_df, cand)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
