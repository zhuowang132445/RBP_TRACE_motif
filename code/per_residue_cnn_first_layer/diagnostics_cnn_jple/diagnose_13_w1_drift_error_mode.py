#!/usr/bin/env python3
"""Diagnose whether w1 latent drift is a known training-set error mode.

No training, no checkpoint edits, no decoder changes. This script compares
RNAcompete training residuals with the w1 residual direction.
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

from diagnose_07_cnn_vs_jple_latent_shift import cnn_latent, l2_normalize, load_cnn, mean_pool, resolve, short_id  # noqa: E402
from diagnose_08_representation_failure_analysis import project_jple  # noqa: E402
from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize  # noqa: E402
from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402


FAMILIES = ["U-rich", "CUUCU-like", "UGUGUG-like", "GA-rich", "mixed/other"]


def log(msg: str) -> None:
    print(f"[diagnose-13] {msg}", flush=True)


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["protein_id", "domain_family", "domain_architecture"])
    df = pd.read_csv(path, sep="\t")
    keep = [c for c in ["protein_id", "domain_family", "domain_architecture", "sequence_length"] if c in df.columns]
    return df[keep].drop_duplicates("protein_id")


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0] = np.nan
    return np.sum(a * b, axis=1) / denom


def vector_cosine_to_rows(rows: np.ndarray, vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(rows, axis=1) * np.linalg.norm(vec)
    denom[denom == 0] = np.nan
    return (rows @ vec) / denom


def neighbor_sets(query_w: np.ndarray, train_w: np.ndarray, train_ids: list[str], self_index: int | None = None, k: int = 50) -> tuple[list[str], np.ndarray]:
    dist = cdist(query_w[None, :], train_w, "cosine")[0]
    if self_index is not None:
        dist[self_index] = np.inf
    order = np.argsort(dist)[:k]
    return [train_ids[int(i)] for i in order], dist


def family_counts(ids: list[str], train_family: dict[str, str]) -> dict[str, int]:
    counts = {fam: 0 for fam in FAMILIES}
    for pid in ids:
        counts[train_family[pid]] = counts.get(train_family[pid], 0) + 1
    return counts


def shift_for_protein(
    pid: str,
    idx: int,
    true_w: np.ndarray,
    cnn_w: np.ndarray,
    train_ids: list[str],
    train_family: dict[str, str],
) -> dict[str, Any]:
    exact_ids, _ = neighbor_sets(true_w[idx], true_w, train_ids, self_index=idx, k=50)
    cnn_ids, _ = neighbor_sets(cnn_w[idx], true_w, train_ids, self_index=idx, k=50)
    exact_set = set(exact_ids)
    cnn_set = set(cnn_ids)
    lost = list(exact_set - cnn_set)
    gained = list(cnn_set - exact_set)
    exact_counts = family_counts(exact_ids, train_family)
    cnn_counts = family_counts(cnn_ids, train_family)
    lost_counts = family_counts(lost, train_family)
    gained_counts = family_counts(gained, train_family)
    exact_majority = max(exact_counts, key=exact_counts.get)
    cnn_majority = max(cnn_counts, key=cnn_counts.get)
    row: dict[str, Any] = {
        "protein": pid,
        "neighbor_overlap": len(exact_set & cnn_set) / 50.0,
        "neighbor_loss": 1.0 - len(exact_set & cnn_set) / 50.0,
        "exact_majority_family": exact_majority,
        "cnn_majority_family": cnn_majority,
        "u_to_mixed_shift": bool(exact_majority == "U-rich" and cnn_majority == "mixed/other"),
        "u_to_ga_shift": bool(exact_majority == "U-rich" and cnn_majority == "GA-rich"),
    }
    for fam in FAMILIES:
        row[f"exact_top50_{fam}_fraction"] = exact_counts[fam] / 50.0
        row[f"cnn_top50_{fam}_fraction"] = cnn_counts[fam] / 50.0
        row[f"lost_{fam}_count"] = lost_counts[fam]
        row[f"gained_{fam}_count"] = gained_counts[fam]
    return row


def write_report(
    out: Path,
    residuals: pd.DataFrame,
    top50: pd.DataFrame,
    shifts: pd.DataFrame,
    family_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
) -> None:
    similar_count = int((top50["drift_similarity_to_w1"] >= 0.5).sum())
    similar_u_shift = int(((shifts["u_to_mixed_shift"] | shifts["u_to_ga_shift"])).sum())
    top50_mean_overlap = float(shifts["neighbor_overlap"].mean())
    top50_mean_lost_u = float(shifts["lost_U-rich_count"].mean())
    top50_mean_gained_mixed = float(shifts["gained_mixed/other_count"].mean())
    top10 = top50.head(10)
    top10_fams = top10["family"].value_counts().to_dict()
    if similar_count >= 10 and similar_u_shift >= 3:
        verdict = "Situation A: training set contains a detectable w1-like drift mode; this looks systematically learnable."
    else:
        verdict = "Situation B: few/no training proteins show both w1-like drift direction and similar U-rich-to-mixed/GA neighbor loss; w1 is closer to query-specific OOD drift."
    lines = [
        "# Diagnose 13 w1 Drift Error Mode",
        "",
        "## Direct Answer",
        "",
        verdict,
        "",
        "## Quantitative Evidence",
        "",
        f"- Top50 drift-similar proteins with cosine similarity >= 0.5: {similar_count}",
        f"- Among top50 drift-similar proteins, U-rich -> mixed/GA shifts: {similar_u_shift}",
        f"- Top50 drift-similar mean neighbor overlap: {top50_mean_overlap:.3f}",
        f"- Top50 drift-similar mean lost U-rich count: {top50_mean_lost_u:.2f}",
        f"- Top50 drift-similar mean gained mixed count: {top50_mean_gained_mixed:.2f}",
        f"- Top10 drift-similar family composition: {top10_fams}",
        "",
        "## Top20 w1-like Drift Proteins",
        "",
        top50.head(20).to_markdown(index=False),
        "",
        "## Family Drift Summary",
        "",
        family_summary.to_markdown(index=False),
        "",
        "## Domain Drift Summary",
        "",
        domain_summary.to_markdown(index=False),
        "",
        "## Output Files",
        "",
        "- `training_protein_residuals.tsv`",
        "- `w1_drift_nearest_training_proteins.tsv`",
        "- `drift_similar_protein_neighbor_shift.tsv`",
        "- `family_drift_summary.tsv`",
        "- `domain_drift_summary.tsv`",
    ]
    (out / "diagnose_13_w1_drift_error_mode_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--train-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--cnn-checkpoint", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_cnn/per_residue_cnn_jple_checkpoint.pt")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617/diagnose_13_w1_drift_error_mode")
    p.add_argument("--num-eigenvector", type=int, default=122)
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
    true_w = anchor_w[np.asarray(keep, dtype=int)].astype(np.float32)
    y_train = np.vstack([y_all[motif_index[pid]] for pid in train_ids]).astype(np.float32)
    train_family = {pid: assign_profile_family(y_train[i], kmers, 50) for i, pid in enumerate(train_ids)}
    log(f"train_n={len(train_ids)}")

    log("computing frozen CNN train/w1 latents")
    cnn_model = load_cnn(resolve(args.cnn_checkpoint), device)
    cnn_w = cnn_latent(cnn_model, train_ids, train_map, device, args.batch_size)
    w1_cnn = cnn_latent(cnn_model, [w1_id], query_map, device, args.batch_size)[0]

    log("computing exact JPLE w1 latent and aligning basis")
    x_train = l2_normalize(mean_pool(train_ids, train_map))
    x_w1 = l2_normalize(mean_pool([w1_id], query_map))
    jple = RBPTraceFirstLayer(args.num_eigenvector, threshold=0.01, std=0.2)
    jple.fit(x_train, y_train)
    exact_train = jple.w_train.astype(np.float32)
    w1_exact = project_jple(jple, x_w1)[0]
    signs = np.sign(np.sum(exact_train * true_w, axis=0))
    signs[signs == 0] = 1.0
    w1_exact_aligned = w1_exact * signs

    delta = true_w - cnn_w
    w1_delta = w1_exact_aligned - w1_cnn
    latent_cos = row_cosine(true_w, cnn_w)
    residuals = pd.DataFrame(
        {
            "protein": train_ids,
            "family": [train_family[pid] for pid in train_ids],
            "delta_norm": np.linalg.norm(delta, axis=1),
            "latent_cosine": latent_cos,
            "latent_cosine_distance": 1.0 - latent_cos,
            "drift_similarity_to_w1": vector_cosine_to_rows(delta, w1_delta),
        }
    )
    residuals = residuals.merge(manifest.rename(columns={"protein_id": "protein"}), on="protein", how="left")
    residuals.to_csv(out / "training_protein_residuals.tsv", sep="\t", index=False)

    top50 = residuals.sort_values("drift_similarity_to_w1", ascending=False).head(50).copy()
    top50.to_csv(out / "w1_drift_nearest_training_proteins.tsv", sep="\t", index=False)

    log("computing neighbor shifts for drift-similar proteins")
    id_to_idx = {pid: i for i, pid in enumerate(train_ids)}
    shift_rows = []
    for pid in top50["protein"]:
        idx = id_to_idx[pid]
        row = shift_for_protein(pid, idx, true_w, cnn_w, train_ids, train_family)
        row.update(
            {
                "family": train_family[pid],
                "domain_architecture": residuals.set_index("protein").loc[pid, "domain_architecture"],
                "domain_family": residuals.set_index("protein").loc[pid, "domain_family"],
                "delta_norm": float(residuals.set_index("protein").loc[pid, "delta_norm"]),
                "drift_similarity_to_w1": float(residuals.set_index("protein").loc[pid, "drift_similarity_to_w1"]),
            }
        )
        shift_rows.append(row)
    shifts = pd.DataFrame(shift_rows)
    shifts.to_csv(out / "drift_similar_protein_neighbor_shift.tsv", sep="\t", index=False)

    log("summarizing family/domain drift")
    all_shift_rows = []
    for i, pid in enumerate(train_ids):
        row = shift_for_protein(pid, i, true_w, cnn_w, train_ids, train_family)
        row.update({"family": train_family[pid]})
        all_shift_rows.append(row)
    all_shifts = pd.DataFrame(all_shift_rows)
    merged = residuals.merge(all_shifts[["protein", "neighbor_overlap", "neighbor_loss", "lost_U-rich_count", "gained_mixed/other_count", "gained_GA-rich_count"]], on="protein", how="left")
    family_summary = (
        merged.groupby("family")
        .agg(
            n=("protein", "count"),
            mean_delta_norm=("delta_norm", "mean"),
            median_delta_norm=("delta_norm", "median"),
            mean_drift_similarity_to_w1=("drift_similarity_to_w1", "mean"),
            max_drift_similarity_to_w1=("drift_similarity_to_w1", "max"),
            mean_neighbor_loss=("neighbor_loss", "mean"),
            mean_lost_Urich=("lost_U-rich_count", "mean"),
            mean_gained_mixed=("gained_mixed/other_count", "mean"),
            mean_gained_GArich=("gained_GA-rich_count", "mean"),
        )
        .reset_index()
    )
    family_summary.to_csv(out / "family_drift_summary.tsv", sep="\t", index=False)

    merged["domain_group"] = merged["domain_architecture"].fillna("unknown")
    domain_summary = (
        merged.groupby("domain_group")
        .agg(
            n=("protein", "count"),
            mean_delta_norm=("delta_norm", "mean"),
            mean_neighbor_loss=("neighbor_loss", "mean"),
            mean_drift_similarity_to_w1=("drift_similarity_to_w1", "mean"),
            max_drift_similarity_to_w1=("drift_similarity_to_w1", "max"),
        )
        .reset_index()
        .sort_values(["n", "mean_drift_similarity_to_w1"], ascending=[False, False])
    )
    domain_summary.to_csv(out / "domain_drift_summary.tsv", sep="\t", index=False)

    write_report(out, residuals, top50, shifts, family_summary, domain_summary)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
