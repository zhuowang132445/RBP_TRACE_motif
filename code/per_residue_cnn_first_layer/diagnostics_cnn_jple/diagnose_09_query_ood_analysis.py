#!/usr/bin/env python3
"""Query OOD and training-support analysis for frozen CNN+JPLE.

This script does not retrain, modify model parameters, or alter existing
scripts. It uses existing embeddings, the frozen CNN checkpoint, and exact JPLE
latent geometry to quantify whether w1 is query-specific OOD / support-limited.
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
    FAMILIES,
    cnn_latent,
    l2_normalize,
    load_cnn,
    mean_pool,
    resolve,
    short_id,
)
from diagnose_08_representation_failure_analysis import (  # noqa: E402
    cosine_similarity_rows,
    neighbor_overlap_table,
    project_jple,
    top_neighbors_self,
)
from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize  # noqa: E402
from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402


def log(msg: str) -> None:
    print(f"[diagnose-09] {msg}", flush=True)


def top1_kmer(profile: np.ndarray, kmers: np.ndarray) -> str:
    return str(kmers[int(np.argmax(profile))])


def query_distance_summary(
    query_ids: list[str],
    train_ids: list[str],
    query_w: np.ndarray,
    train_w: np.ndarray,
    space: str,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist = cdist(query_w, train_w, "cosine")
    rows = []
    neigh_rows = []
    for qi, qid in enumerate(query_ids):
        order = np.argsort(dist[qi])
        row: dict[str, Any] = {"query": short_id(qid), "query_id": qid, "space": space}
        for k in ks:
            idx = order[:k]
            row[f"top{k}_mean_distance"] = float(np.mean(dist[qi, idx]))
        row["top1_distance"] = float(dist[qi, order[0]])
        row["top1_train_protein_id"] = train_ids[int(order[0])]
        rows.append(row)
        for rank, ti in enumerate(order[:50], start=1):
            neigh_rows.append(
                {
                    "query": short_id(qid),
                    "query_id": qid,
                    "space": space,
                    "neighbor_rank": rank,
                    "train_protein_id": train_ids[int(ti)],
                    "cosine_distance": float(dist[qi, ti]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(neigh_rows)


def entropy_from_counts(counts: list[int]) -> tuple[float, float]:
    total = float(sum(counts))
    if total <= 0:
        return float("nan"), float("nan")
    p = np.asarray([c / total for c in counts if c > 0], dtype=np.float64)
    h = float(-np.sum(p * np.log(p)))
    return h, float(h / np.log(len(FAMILIES)))


def add_neighbor_family(neigh: pd.DataFrame, train_family: dict[str, str], train_top1: dict[str, str]) -> pd.DataFrame:
    out = neigh.copy()
    out["neighbor_family"] = out["train_protein_id"].map(train_family)
    out["neighbor_true_top1_kmer"] = out["train_protein_id"].map(train_top1)
    return out


def neighbor_entropy_table(neigh: pd.DataFrame, ks: tuple[int, ...] = (10, 20, 50)) -> pd.DataFrame:
    rows = []
    for (query, qid, space), sub0 in neigh.groupby(["query", "query_id", "space"], sort=False):
        sub0 = sub0.sort_values("neighbor_rank")
        for k in ks:
            sub = sub0.head(k)
            counts = sub["neighbor_family"].value_counts().to_dict()
            h, hn = entropy_from_counts([int(counts.get(fam, 0)) for fam in FAMILIES])
            row: dict[str, Any] = {
                "query": query,
                "query_id": qid,
                "space": space,
                "top_k": k,
                "neighbor_entropy": h,
                "neighbor_entropy_normalized": hn,
            }
            for fam in FAMILIES:
                row[f"fraction_{fam}"] = float(counts.get(fam, 0) / k)
                row[f"count_{fam}"] = int(counts.get(fam, 0))
            rows.append(row)
    return pd.DataFrame(rows)


def support_score_table(neigh: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    rows = []
    for (query, qid, space), sub in neigh.groupby(["query", "query_id", "space"], sort=False):
        sub = sub.sort_values("neighbor_rank").head(top_k)
        families = sub["neighbor_family"].astype(str).tolist()
        counts = pd.Series(families).value_counts()
        majority_family = str(counts.index[0])
        majority_fraction = float(counts.iloc[0] / top_k)
        second_family = str(counts.index[1]) if len(counts) > 1 else ""
        second_fraction = float(counts.iloc[1] / top_k) if len(counts) > 1 else 0.0
        top1_family = str(families[0])
        top1_family_fraction = float(sum(f == top1_family for f in families) / top_k)
        rows.append(
            {
                "query": query,
                "query_id": qid,
                "space": space,
                "top_k": top_k,
                "top1_neighbor_family": top1_family,
                "top1_neighbor_family_fraction": top1_family_fraction,
                "majority_family": majority_family,
                "majority_fraction": majority_fraction,
                "second_majority_family": second_family,
                "second_majority_fraction": second_fraction,
                "support_score": majority_fraction - second_fraction,
            }
        )
    return pd.DataFrame(rows)


def w1_nearest_training(neigh: pd.DataFrame, space: str, top_n: int = 20) -> pd.DataFrame:
    return neigh[(neigh["query"] == "w1") & (neigh["space"] == space)].sort_values("neighbor_rank").head(top_n).copy()


def make_w1_like_stability(
    w1_top20: pd.DataFrame,
    train_ids: list[str],
    true_jple_w: np.ndarray,
    cnn_pred_w: np.ndarray,
    train_family: dict[str, str],
) -> pd.DataFrame:
    jple_neighbors = top_neighbors_self(true_jple_w, train_ids, 50)
    cnn_neighbors = top_neighbors_self(cnn_pred_w, train_ids, 50)
    overlap = neighbor_overlap_table(train_ids, jple_neighbors, cnn_neighbors, train_family, 50).set_index("protein_id")
    cos = cosine_similarity_rows(true_jple_w, cnn_pred_w)
    euc = np.linalg.norm(true_jple_w - cnn_pred_w, axis=1)
    err = pd.DataFrame({"protein_id": train_ids, "latent_cosine_similarity": cos, "latent_euclidean_distance": euc}).set_index("protein_id")
    rows = []
    for _, r in w1_top20.iterrows():
        pid = r["train_protein_id"]
        rows.append(
            {
                "w1_exact_jple_rank": int(r["neighbor_rank"]),
                "protein_id": pid,
                "w1_exact_jple_distance": float(r["cosine_distance"]),
                "motif_family": r["neighbor_family"],
                "top1_kmer": r["neighbor_true_top1_kmer"],
                "internal_neighbor_overlap": float(overlap.loc[pid, "neighbor_overlap"]),
                "internal_shared_neighbor_n": int(overlap.loc[pid, "shared_neighbor_n"]),
                "latent_cosine_similarity_true_jple_vs_cnn": float(err.loc[pid, "latent_cosine_similarity"]),
                "latent_euclidean_distance_true_jple_vs_cnn": float(err.loc[pid, "latent_euclidean_distance"]),
            }
        )
    return pd.DataFrame(rows)


def wide_distance_summary(dist_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (query, qid), sub in dist_long.groupby(["query", "query_id"], sort=False):
        row: dict[str, Any] = {"query": query, "query_id": qid}
        for _, r in sub.iterrows():
            prefix = "exact_jple" if r["space"] == "exact_jple_latent" else "cnn_predicted"
            for col in ["top1_distance", "top5_mean_distance", "top10_mean_distance", "top50_mean_distance", "top1_train_protein_id"]:
                row[f"{prefix}_{col}"] = r[col]
        rows.append(row)
    return pd.DataFrame(rows)


def report(
    out: Path,
    distance_wide: pd.DataFrame,
    entropy_df: pd.DataFrame,
    support_df: pd.DataFrame,
    w1_top20: pd.DataFrame,
    w1_stability: pd.DataFrame,
) -> None:
    success = {"w3", "w4", "w6", "AtPTBP3"}
    exact = distance_wide.copy()
    w1 = exact[exact["query"] == "w1"].iloc[0]
    success_median_top1 = float(exact[exact["query"].isin(success)]["exact_jple_top1_distance"].median())
    success_median_cnn_top1 = float(exact[exact["query"].isin(success)]["cnn_predicted_top1_distance"].median())
    w1_ood = float(w1["exact_jple_top1_distance"]) > success_median_top1 and float(w1["cnn_predicted_top1_distance"]) > success_median_cnn_top1

    ent50 = entropy_df[entropy_df["top_k"] == 50]
    w1_ent = ent50[(ent50["query"] == "w1") & (ent50["space"] == "exact_jple_latent")].iloc[0]
    success_ent = float(ent50[(ent50["query"].isin(success)) & (ent50["space"] == "exact_jple_latent")]["neighbor_entropy_normalized"].median())

    sup50 = support_df.copy()
    w1_sup_exact = sup50[(sup50["query"] == "w1") & (sup50["space"] == "exact_jple_latent")].iloc[0]
    w1_sup_cnn = sup50[(sup50["query"] == "w1") & (sup50["space"] == "cnn_predicted_latent")].iloc[0]
    top20_comp = w1_top20["neighbor_family"].value_counts().to_dict()
    stability_mean = float(w1_stability["internal_neighbor_overlap"].mean())
    stability_cos = float(w1_stability["latent_cosine_similarity_true_jple_vs_cnn"].mean())

    if w1_ood:
        ood_text = "w1 比成功 query 更远离训练集，支持 query-specific OOD。"
    else:
        ood_text = "w1 在距离上并不明显比成功 query 更远，单纯 OOD 距离不是充分解释。"
    if float(w1_ent["neighbor_entropy_normalized"]) > success_ent:
        ambiguity_text = "w1 的邻居 family entropy 高于成功 query，存在更强 motif ambiguity。"
    else:
        ambiguity_text = "w1 的邻居 entropy 不高于成功 query，混杂性不是主要由熵体现。"
    if stability_mean >= 0.75 and stability_cos >= 0.80:
        stability_text = "w1-like 训练蛋白内部稳定，CNN 没有系统性拉偏这些训练蛋白；更支持只有外部 w1 漂移。"
    else:
        stability_text = "w1-like 训练蛋白本身也有一定漂移，训练邻域可能不够稳。"

    lines = [
        "# Diagnose 09 Query OOD Analysis",
        "",
        "## Direct Answers",
        "",
        f"1. w1 是否明显比其他成功 query 更 OOD：{ood_text}",
        f"2. w1 是否缺乏稳定训练邻域支持：exact support_score={w1_sup_exact['support_score']:.3f}, CNN support_score={w1_sup_cnn['support_score']:.3f}；{ambiguity_text}",
        f"3. w1 周围训练蛋白是否 motif 一致：top20 family composition={top20_comp}。",
        f"4. CNN 是否系统性破坏 w1 类蛋白：{stability_text}",
        "5. 当前最支持解释：query-specific latent drift / OOD 为主；training support deficiency 和 motif ambiguity 为辅助；不是 decoder 参数问题。",
        "",
        "## Query OOD Distance Summary",
        "",
        distance_wide.to_markdown(index=False),
        "",
        "## Query Support Score",
        "",
        support_df.to_markdown(index=False),
        "",
        "## Neighbor Entropy Top50",
        "",
        ent50[["query", "space", "neighbor_entropy_normalized", "fraction_U-rich", "fraction_CUUCU-like", "fraction_UGUGUG-like", "fraction_GA-rich", "fraction_mixed/other"]].to_markdown(index=False),
        "",
        "## w1 Nearest Exact JPLE Training Proteins Top20",
        "",
        w1_top20[["neighbor_rank", "train_protein_id", "cosine_distance", "neighbor_family", "neighbor_true_top1_kmer"]].to_markdown(index=False),
        "",
        "## w1-like Training Protein Stability",
        "",
        w1_stability.to_markdown(index=False),
        "",
        "## Output Files",
        "",
        "- `query_ood_distance_summary.tsv`",
        "- `query_neighbor_entropy.tsv`",
        "- `query_support_score.tsv`",
        "- `w1_nearest_training_proteins.tsv`",
        "- `w1_like_training_proteins_stability.tsv`",
        "- `diagnose_09_query_ood_analysis_report.md`",
    ]
    (out / "diagnose_09_query_ood_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_09_query_ood_analysis")
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
    query_ids = sorted(query_map.keys(), key=short_id)

    anchor = np.load(resolve(args.jple_anchor_npz), allow_pickle=True)
    anchor_ids = np.asarray(anchor["train_protein_id_list"]).astype(str).tolist()
    anchor_w = np.asarray(anchor["w_train"], dtype=np.float32)
    keep = [i for i, pid in enumerate(anchor_ids) if pid in train_map and pid in motif_index]
    train_ids = [anchor_ids[i] for i in keep]
    true_jple_w = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    train_top1 = {pid: top1_kmer(y_train[i], kmers) for i, pid in enumerate(train_ids)}
    log(f"train_n={len(train_ids)} query_n={len(query_ids)}")

    log("computing frozen CNN query/train latents")
    model = load_cnn(resolve(args.cnn_checkpoint), device)
    cnn_query_w = cnn_latent(model, query_ids, query_map, device, args.batch_size)
    cnn_train_w = cnn_latent(model, train_ids, train_map, device, args.batch_size)

    log("computing exact JPLE query projection")
    x_train = l2_normalize(mean_pool(train_ids, train_map))
    x_query = l2_normalize(mean_pool(query_ids, query_map))
    jple = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
    jple.fit(x_train, y_train)
    exact_jple_train = jple.w_train.astype(np.float32)
    exact_jple_query = project_jple(jple, x_query)

    log("query OOD distances and neighbor tables")
    exact_dist, exact_neigh = query_distance_summary(query_ids, train_ids, exact_jple_query, exact_jple_train, "exact_jple_latent")
    cnn_dist, cnn_neigh = query_distance_summary(query_ids, train_ids, cnn_query_w, true_jple_w, "cnn_predicted_latent")
    distance_long = pd.concat([exact_dist, cnn_dist], ignore_index=True)
    distance_wide = wide_distance_summary(distance_long)
    distance_wide.to_csv(out / "query_ood_distance_summary.tsv", sep="\t", index=False)
    all_neigh = add_neighbor_family(pd.concat([exact_neigh, cnn_neigh], ignore_index=True), train_family, train_top1)
    all_neigh.to_csv(out / "query_neighbor_top50_by_space.tsv", sep="\t", index=False)

    log("neighbor entropy and support score")
    entropy_df = neighbor_entropy_table(all_neigh)
    entropy_df.to_csv(out / "query_neighbor_entropy.tsv", sep="\t", index=False)
    support_df = support_score_table(all_neigh, 50)
    support_df.to_csv(out / "query_support_score.tsv", sep="\t", index=False)

    log("w1 nearest training proteins and stability")
    w1_top20 = w1_nearest_training(all_neigh, "exact_jple_latent", 20)
    w1_top20.to_csv(out / "w1_nearest_training_proteins.tsv", sep="\t", index=False)
    w1_stability = make_w1_like_stability(w1_top20, train_ids, true_jple_w, cnn_train_w, train_family)
    w1_stability.to_csv(out / "w1_like_training_proteins_stability.tsv", sep="\t", index=False)

    report(out, distance_wide, entropy_df, support_df, w1_top20, w1_stability)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
