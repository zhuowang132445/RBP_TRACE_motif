#!/usr/bin/env python3
"""Pure CNN+JPLE geometry diagnostics without retraining or motif priors.

Diagnostics:
1. RBD crop/domain input ablation through the frozen CNN+JPLE model.
2. Residue-level soft-alignment neighbor decoder using RNAcompete motif profiles.

This script does not modify training scripts, does not retrain the main model,
does not use C-guided reweighting, and does not use motif-family-aware decoding.
Motif-family ranks are computed only for diagnostics after prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from cnn_model_utils import PerResidueCnn  # noqa: E402
from diagnostic_utils import (  # noqa: E402
    align,
    kmer_family,
    load_h5_per_residue,
    load_motif_npz,
    ndcg_at_k,
    pearson,
    pool_embedding,
    row_l2_normalize,
    spearman,
    standardize_rows,
    top_overlap,
    true_top1_rank,
)


QUERY_EXPECTED = {
    "w1": ["U-rich"],
    "w2": ["CUUCU-like"],
    "w3": ["UGUGUG-like"],
    "w4": ["U-rich"],
    "w5": [],
    "w6": ["U-rich"],
    "AtPTBP3": ["CUUCU-like", "U-rich"],
}


def log(msg: str) -> None:
    print(f"[diagnose-06] {msg}", flush=True)


def short_id(qid: str) -> str:
    qid = str(qid)
    if qid.startswith("AtPTBP3"):
        return "AtPTBP3"
    if "|original=" in qid:
        return qid.split("|original=", 1)[1].split("|", 1)[0]
    return qid.split("|", 1)[0]


def safe_key(pid: str) -> str:
    return pid.replace("/", "__slash__")


def load_h5_query_map(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for key in h5["embeddings"].keys():
            arr = np.asarray(h5["embeddings"][key], dtype=np.float32)
            out[key.replace("__slash__", "/")] = arr
    return out


def collate_single(arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(arr.astype(np.float32))[None, :, :]
    mask = torch.ones((1, arr.shape[0]), dtype=torch.bool)
    return x, mask


def load_frozen_cnn_jple(args: argparse.Namespace, device: torch.device) -> tuple[PerResidueCnn, np.ndarray, np.ndarray, list[str]]:
    ckpt = torch.load(ROOT / args.cnn_jple_checkpoint, map_location="cpu", weights_only=False)
    conf = ckpt["model_config"]
    model = PerResidueCnn(
        conf["input_dim"],
        conf["hidden_dim"],
        conf["latent_dim"],
        conf["kernel_size"],
        conf["num_blocks"],
        conf["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    z = np.load(ROOT / args.cnn_jple_anchor_npz, allow_pickle=True)
    train_w = np.asarray(z["w_train"], dtype=np.float32)
    y_train = np.asarray(z["y_train"], dtype=np.float32)
    train_ids = np.asarray(z["train_protein_id_list"]).astype(str).tolist()
    return model, train_w, y_train, train_ids


def predict_cnn_latent(model: PerResidueCnn, arr: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x, mask = collate_single(arr)
        return model(x.to(device), mask.to(device)).cpu().numpy().astype(np.float32)


def decode_jple_latent(
    query_w: np.ndarray,
    train_w: np.ndarray,
    y_train: np.ndarray,
    threshold: float,
    std: float,
) -> np.ndarray:
    from scipy.spatial.distance import cdist

    dist = cdist(query_w, train_w, "cosine")
    preds = []
    for row in dist:
        sim = np.exp(-(row**2) / (std**2))
        idx = np.argwhere(sim >= threshold).flatten()
        if len(idx) == 0:
            idx = np.asarray([int(np.argmax(sim))])
        idx = idx[np.argsort(-sim[idx])]
        w = sim[idx] / sim[idx].sum()
        preds.append(np.sum(w[:, None] * y_train[idx], axis=0))
    return standardize_rows(np.asarray(preds, dtype=np.float32))


def build_family_indices(kmers: np.ndarray) -> dict[str, np.ndarray]:
    fams = {"U-rich": [], "CUUCU-like": [], "UCUCUC-like": [], "UGUGUG-like": [], "GA-rich": []}
    for i, k in enumerate(kmers.astype(str)):
        fam = kmer_family(k)
        if fam in fams:
            fams[fam].append(i)
        # UCUCUC-like is stricter/additional for AtPTBP3 diagnostics.
        if any(seed in k for seed in ["UCUCUC", "CUCUCU", "UCUCU", "CUCUC"]):
            fams["UCUCUC-like"].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in fams.items()}


def best_rank(profile: np.ndarray, idx: np.ndarray, kmers: np.ndarray) -> tuple[int, str, float]:
    order = np.argsort(-profile)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    best = idx[np.argmin(ranks[idx])]
    return int(ranks[best]), str(kmers[best]), float(profile[best])


def summarize_query_profile(qid: str, variant: str, profile: np.ndarray, kmers: np.ndarray, family_idx: dict[str, np.ndarray]) -> dict[str, Any]:
    order = np.argsort(-profile)
    row: dict[str, Any] = {
        "query_id": qid,
        "query": short_id(qid),
        "variant": variant,
        "top1_kmer": str(kmers[order[0]]),
        "top20_kmers": ",".join(kmers[order[:20]].astype(str)),
        "top50_kmers": ",".join(kmers[order[:50]].astype(str)),
    }
    for fam in ["U-rich", "CUUCU-like", "UCUCUC-like", "UGUGUG-like", "GA-rich"]:
        rank, kmer, score = best_rank(profile, family_idx[fam], kmers)
        row[f"{fam}_rank"] = rank
        row[f"{fam}_best_kmer"] = kmer
        row[f"{fam}_best_score"] = score
    expected = QUERY_EXPECTED.get(row["query"], [])
    expected_ranks = [row[f"{fam}_rank"] for fam in expected if fam in ["U-rich", "CUUCU-like", "UGUGUG-like", "GA-rich"]]
    if row["query"] == "AtPTBP3":
        expected_ranks.append(row["UCUCUC-like_rank"])
    row["expected_family_best_rank"] = min(expected_ranks) if expected_ranks else np.nan
    row["expected_family_hit_top20"] = bool(expected_ranks and min(expected_ranks) <= 20)
    row["expected_family_hit_top50"] = bool(expected_ranks and min(expected_ranks) <= 50)
    return row


def crop_variants(arr: np.ndarray) -> dict[str, np.ndarray]:
    variants = {"original": arr}
    if arr.shape[0] > 10:
        variants["trim_N10"] = arr[10:]
        variants["trim_C10"] = arr[:-10]
    if arr.shape[0] > 20:
        variants["trim_both10"] = arr[10:-10]
    if arr.shape[0] > 40:
        variants["trim_both20"] = arr[20:-20]
    return variants


def load_query_domain_lengths(npz_path: Path) -> dict[str, list[int]]:
    if not npz_path.exists():
        return {}
    z = np.load(npz_path, allow_pickle=True)
    ids = np.asarray(z["protein_ids"]).astype(str)
    starts = np.asarray(z["domain_starts"]).astype(str)
    ends = np.asarray(z["domain_ends"]).astype(str)
    out: dict[str, list[int]] = {}
    for pid, s_text, e_text in zip(ids, starts, ends):
        try:
            s = [int(x) for x in s_text.split(";") if x]
            e = [int(x) for x in e_text.split(";") if x]
            lens = [max(1, b - a + 1) for a, b in zip(s, e)]
            if lens:
                out[pid] = lens
        except Exception:
            continue
    return out


def add_domain_variants(qid: str, arr: np.ndarray, lengths_by_id: dict[str, list[int]], variants: dict[str, np.ndarray]) -> None:
    if qid not in lengths_by_id:
        return
    lengths = lengths_by_id[qid]
    if sum(lengths) > arr.shape[0]:
        return
    start = 0
    for i, length in enumerate(lengths, start=1):
        segment = arr[start : start + length]
        start += length
        if segment.shape[0] > 0:
            variants[f"domain_{i}"] = segment


def run_crop_ablation(args: argparse.Namespace, out: Path, kmers: np.ndarray, family_idx: dict[str, np.ndarray], device: torch.device) -> pd.DataFrame:
    model, train_w, y_train, _ = load_frozen_cnn_jple(args, device)
    q_map = load_h5_query_map(ROOT / args.query_rice_per_residue_h5)
    q_map.update(load_h5_query_map(ROOT / args.query_atptbp3_per_residue_h5))
    domain_lengths = load_query_domain_lengths(ROOT / args.query_rice_domain_npz)
    rows = []
    top_rows = []
    for qid, arr in sorted(q_map.items(), key=lambda x: short_id(x[0])):
        variants = crop_variants(arr)
        add_domain_variants(qid, arr, domain_lengths, variants)
        for name, subarr in variants.items():
            if subarr.shape[0] < 5:
                continue
            q_w = predict_cnn_latent(model, subarr, device)
            pred = decode_jple_latent(q_w, train_w, y_train, args.jple_threshold, args.jple_std)[0]
            row = summarize_query_profile(qid, name, pred, kmers, family_idx)
            row["input_length"] = int(subarr.shape[0])
            rows.append(row)
            order = np.argsort(-pred)[:50]
            for rank, idx in enumerate(order, start=1):
                top_rows.append({"query": short_id(qid), "query_id": qid, "variant": name, "rank": rank, "kmer": str(kmers[idx]), "score": float(pred[idx])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "diagnostic6a_rbd_crop_domain_ablation_summary.tsv", sep="\t", index=False)
    pd.DataFrame(top_rows).to_csv(out / "diagnostic6a_rbd_crop_domain_ablation_top50.tsv", sep="\t", index=False)
    return df


def normalize_residues(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def residue_soft_alignment_similarity(q: np.ndarray, n: np.ndarray, device: torch.device) -> float:
    q_t = torch.from_numpy(normalize_residues(q)).to(device)
    n_t = torch.from_numpy(normalize_residues(n)).to(device)
    sim = q_t @ n_t.T
    s1 = sim.max(dim=1).values.mean()
    s2 = sim.max(dim=0).values.mean()
    return float((0.5 * s1 + 0.5 * s2).detach().cpu())


def compute_similarity_matrix(x_map: dict[str, np.ndarray], ids: list[str], cache_path: Path, device: torch.device, force: bool = False) -> np.ndarray:
    if cache_path.exists() and not force:
        log(f"load cached similarity matrix: {cache_path}")
        return np.load(cache_path)
    n = len(ids)
    sim = np.eye(n, dtype=np.float32)
    for i, pid_i in enumerate(ids):
        if i % 10 == 0:
            log(f"residue similarity row {i + 1}/{n}")
        for j in range(i + 1, n):
            val = residue_soft_alignment_similarity(x_map[pid_i], x_map[ids[j]], device)
            sim[i, j] = val
            sim[j, i] = val
    np.save(cache_path, sim)
    return sim


def predict_from_similarity(sim_query_train: np.ndarray, y_train: np.ndarray, k: int, temperature: float, exclude_self: bool = False) -> np.ndarray:
    preds = []
    for qi, row in enumerate(sim_query_train):
        order = np.argsort(-row)
        if exclude_self:
            order = order[order != qi]
        idx = order[:k]
        logits = row[idx] / max(temperature, 1e-8)
        logits = logits - logits.max()
        w = np.exp(logits)
        w = w / w.sum()
        preds.append(np.sum(w[:, None] * y_train[idx], axis=0))
    return standardize_rows(np.asarray(preds, dtype=np.float32))


def metric_row(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "pearson": pearson(pred, true),
        "spearman": spearman(pred, true),
        "top20_overlap": top_overlap(pred, true, 20),
        "top50_overlap": top_overlap(pred, true, 50),
        "ndcg20": ndcg_at_k(pred, true, 20),
        "ndcg50": ndcg_at_k(pred, true, 50),
        "true_top1_rank": float(true_top1_rank(pred, true)),
    }


def run_residue_neighbor_decoder(args: argparse.Namespace, out: Path, motif_ids: list[str], y_raw: np.ndarray, kmers: np.ndarray, family_idx: dict[str, np.ndarray], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_norm = row_l2_normalize(y_raw)
    train_map_all = load_h5_per_residue(ROOT / args.train_per_residue_h5)
    train_ids0, train_x_dummy = pool_embedding(train_map_all, "mean")
    train_ids, _, y_train = align(train_ids0, train_x_dummy, motif_ids, y_norm)
    train_map = {pid: train_map_all[pid] for pid in train_ids}
    y_raw_by_id = {pid: y_raw[i] for i, pid in enumerate(motif_ids)}
    y_train_raw = np.vstack([y_raw_by_id[pid] for pid in train_ids]).astype(np.float32)
    cache_path = out / "residue_soft_alignment_similarity_matrix.npy"
    sim = compute_similarity_matrix(train_map, train_ids, cache_path, device, args.recompute_similarity)

    q_map = load_h5_query_map(ROOT / args.query_rice_per_residue_h5)
    q_map.update(load_h5_query_map(ROOT / args.query_atptbp3_per_residue_h5))
    q_ids = sorted(q_map, key=short_id)
    sim_q = np.zeros((len(q_ids), len(train_ids)), dtype=np.float32)
    for qi, qid in enumerate(q_ids):
        log(f"query residue similarity {short_id(qid)}")
        for ti, tid in enumerate(train_ids):
            sim_q[qi, ti] = residue_soft_alignment_similarity(q_map[qid], train_map[tid], device)

    metric_rows = []
    query_rows = []
    top_rows = []
    for k in [10, 20, 50]:
        for temp in [0.05, 0.1, 0.2]:
            config = f"residue_softalign_k{k}_t{temp:g}"
            pred_train = predict_from_similarity(sim, y_train, k, temp, exclude_self=True)
            metrics = pd.DataFrame([metric_row(pred_train[i], y_train_raw[i]) for i in range(len(train_ids))])
            mrow: dict[str, Any] = {
                "config_id": config,
                "neighbor_k": k,
                "temperature": temp,
                "pearson_mean": metrics["pearson"].mean(),
                "spearman_mean": metrics["spearman"].mean(),
                "top20_overlap_mean": metrics["top20_overlap"].mean(),
                "top50_overlap_mean": metrics["top50_overlap"].mean(),
                "ndcg20_mean": metrics["ndcg20"].mean(),
                "ndcg50_mean": metrics["ndcg50"].mean(),
                "true_top1_rank_median": metrics["true_top1_rank"].median(),
                "acceptable": bool(metrics["pearson"].mean() >= args.acceptable_pearson and metrics["ndcg20"].mean() >= args.acceptable_ndcg20),
            }
            pred_q = predict_from_similarity(sim_q, y_train, k, temp, exclude_self=False)
            hit_count = 0
            for qi, qid in enumerate(q_ids):
                qrow = summarize_query_profile(qid, config, pred_q[qi], kmers, family_idx)
                qrow.update({"config_id": config, "neighbor_k": k, "temperature": temp})
                query_rows.append(qrow)
                hit_count += int(qrow["expected_family_hit_top20"])
                mrow[f"{qrow['query']}_expected_rank"] = qrow["expected_family_best_rank"]
                mrow[f"{qrow['query']}_hit_top20"] = qrow["expected_family_hit_top20"]
                order = np.argsort(-pred_q[qi])[:50]
                for rank, idx in enumerate(order, start=1):
                    top_rows.append({"config_id": config, "query": qrow["query"], "rank": rank, "kmer": str(kmers[idx]), "score": float(pred_q[qi, idx])})
            mrow["query_expected_hit_count_top20"] = hit_count
            metric_rows.append(mrow)

    metric_df = pd.DataFrame(metric_rows)
    query_df = pd.DataFrame(query_rows)
    metric_df.to_csv(out / "diagnostic6b_residue_neighbor_pseudoquery_metrics.tsv", sep="\t", index=False)
    query_df.to_csv(out / "diagnostic6b_residue_neighbor_query_summary.tsv", sep="\t", index=False)
    pd.DataFrame(top_rows).to_csv(out / "diagnostic6b_residue_neighbor_query_top50.tsv", sep="\t", index=False)
    return metric_df, query_df


def write_report(out: Path, crop_df: pd.DataFrame, residue_metrics: pd.DataFrame, residue_query: pd.DataFrame) -> None:
    lines = []
    lines.append("# Diagnose 06 Pure CNN+JPLE Geometry\n\n")
    lines.append("## RBD Crop / Domain Ablation\n\n")
    if len(crop_df):
        key_cols = ["query", "variant", "top1_kmer", "U-rich_rank", "CUUCU-like_rank", "UCUCUC-like_rank", "UGUGUG-like_rank", "GA-rich_rank", "expected_family_best_rank"]
        lines.append(crop_df[key_cols].to_markdown(index=False))
        lines.append("\n\n")
    lines.append("## Residue-Level Neighbor Decoder\n\n")
    if len(residue_metrics):
        lines.append(residue_metrics[["config_id", "pearson_mean", "ndcg20_mean", "top20_overlap_mean", "query_expected_hit_count_top20", "acceptable"]].to_markdown(index=False))
        lines.append("\n\n")
        ok = residue_metrics[residue_metrics["acceptable"]]
        if len(ok):
            lines.append(f"Acceptable configs: {len(ok)}/{len(residue_metrics)}.\n\n")
            lines.append("Best acceptable configs by query hit count and RNAcompete metrics:\n\n")
            best = ok.sort_values(["query_expected_hit_count_top20", "pearson_mean", "ndcg20_mean"], ascending=False).head(10)
            lines.append(best[["config_id", "pearson_mean", "ndcg20_mean", "query_expected_hit_count_top20", "w1_expected_rank", "w3_expected_rank", "w4_expected_rank", "w6_expected_rank", "AtPTBP3_expected_rank"]].to_markdown(index=False))
            lines.append("\n\n")
        else:
            lines.append("No residue-level neighbor config passed the acceptable RNAcompete threshold.\n\n")
    lines.append("## Success Criteria\n\n")
    lines.append("- RNAcompete acceptable threshold: Pearson >= 0.775 and NDCG@20 >= 0.747.\n")
    lines.append("- Query observations are post-hoc only and are not used to choose model parameters.\n")
    (out / "diagnose_06_pure_cnn_jple_geometry_report.md").write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    parser.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    parser.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    parser.add_argument("--query-rice-domain-npz", default="results/final_rice_prediction/rice_inputs/rice_w1_w6_domain_merged_esmc_embeddings.npz")
    parser.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    parser.add_argument("--cnn-jple-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    parser.add_argument("--cnn-jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    parser.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_06_pure_cnn_jple_geometry")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--jple-threshold", type=float, default=0.01)
    parser.add_argument("--jple-std", type=float, default=0.2)
    parser.add_argument("--acceptable-pearson", type=float, default=0.775)
    parser.add_argument("--acceptable-ndcg20", type=float, default=0.747)
    parser.add_argument("--skip-crop-ablation", action="store_true")
    parser.add_argument("--skip-residue-neighbor", action="store_true")
    parser.add_argument("--recompute-similarity", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    motif_ids, y_raw, kmers = load_motif_npz(ROOT / args.motif_npz)
    family_idx = build_family_indices(kmers)
    crop_df = pd.DataFrame()
    residue_metrics = pd.DataFrame()
    residue_query = pd.DataFrame()
    if not args.skip_crop_ablation:
        log("run crop/domain ablation")
        crop_df = run_crop_ablation(args, out, kmers, family_idx, device)
    if not args.skip_residue_neighbor:
        log("run residue-level neighbor decoder")
        residue_metrics, residue_query = run_residue_neighbor_decoder(args, out, motif_ids, y_raw, kmers, family_idx, device)
    write_report(out, crop_df, residue_metrics, residue_query)
    log(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
