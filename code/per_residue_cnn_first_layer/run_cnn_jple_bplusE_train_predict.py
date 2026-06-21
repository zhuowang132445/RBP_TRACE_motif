#!/usr/bin/env python3
"""Train and predict a CNN+JPLE B+E repair branch.

B: cosine-aligned CNN encoder training.
E: residual correction head on top of the trained encoder latent.

The original CNN+JPLE checkpoint and JPLE decoder anchors are not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics_cnn_jple"
sys.path.insert(0, str(DIAG_DIR))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer" / "diagnostics"))

from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from diagnostic_utils import assign_profile_family, load_motif_npz, row_l2_normalize  # noqa: E402
from diagnose_07_cnn_vs_jple_latent_shift import resolve, short_id  # noqa: E402
from diagnose_14_encoder_alignment_benchmark import (  # noqa: E402
    decode_threshold,
    evaluate_strategy,
    family_indices,
    neighbor_metrics,
    predict_encoder,
    profile_metrics,
    seed_all,
    train_encoder_config,
    train_residual_head,
)


def log(msg: str) -> None:
    print(f"[cnn-jple-bplusE] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--jple-anchor-npz", default="results/per_residue_cnn_first_layer/jple_embedding_variants_all348_20260617/per_residue_mean_anchor_jple_all348_model.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/cnn_jple_BplusE_20260617")
    p.add_argument("--encoder-epochs", type=int, default=160)
    p.add_argument("--residual-epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--nce-temperature", type=float, default=0.1)
    p.add_argument("--decoder-threshold", type=float, default=0.01)
    p.add_argument("--decoder-std", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.2)
    p.add_argument("--torch-num-threads", type=int, default=1)
    args = p.parse_args()

    # Reuse diagnose_14 helper argument names.
    args.epochs = args.encoder_epochs
    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
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
    log(f"train_n={len(train_ids)} query_n={len(query_ids)}")

    positive_mask = np.zeros((len(train_ids), len(train_ids)), dtype=np.float32)
    exact_dist = cdist(z_true, z_true, "cosine")
    np.fill_diagonal(exact_dist, np.inf)
    for i in range(len(train_ids)):
        positive_mask[i, np.argsort(exact_dist[i])[:10]] = 1.0

    log("training B cosine-aligned encoder")
    encoder_dir = out / "B_encoder_cos0.5"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    config = {"strategy_id": "BplusE_B_encoder_cos0.5", "route": "BplusE", "cos": 0.5}
    encoder = train_encoder_config(config, train_ids, train_map, z_true, positive_mask, device, args, encoder_dir)
    z_train_b = predict_encoder(encoder, train_ids, train_map, device, args.batch_size)
    z_query_b = predict_encoder(encoder, query_ids, query_map, device, args.batch_size)

    log("training residual correction head on B encoder latent")
    residual_dir = out / "E_residual_head_on_B"
    residual_dir.mkdir(parents=True, exist_ok=True)
    head = train_residual_head(z_train_b, z_true, args, residual_dir)
    with torch.no_grad():
        z_train_be = head(torch.from_numpy(z_train_b.astype(np.float32))).numpy().astype(np.float32)
        z_query_be = head(torch.from_numpy(z_query_b.astype(np.float32))).numpy().astype(np.float32)

    log("evaluating and predicting")
    metrics, q = evaluate_strategy(
        "BplusE_cos0.5_residual",
        "B_cosine_encoder_plus_E_residual",
        z_train_be,
        z_query_be,
        z_true,
        y_train,
        train_ids,
        train_family_list,
        train_family,
        query_ids,
        kmers,
        fam_idx,
        out,
        args,
    )
    pd.DataFrame([metrics]).to_csv(out / "rnacompete_metrics.tsv", sep="\t", index=False)
    q.to_csv(out / "query_prediction.tsv", sep="\t", index=False)
    pred_query = decode_threshold(z_query_be, z_true, y_train, args.decoder_threshold, args.decoder_std, exclude_self=False)
    top_rows = []
    for qid, profile in zip(query_ids, pred_query):
        order = np.argsort(-profile)[:100]
        for rank, idx in enumerate(order, start=1):
            top_rows.append({"query": short_id(qid), "query_id": qid, "rank": rank, "kmer": str(kmers[idx]), "score": float(profile[idx])})
    pd.DataFrame(top_rows).to_csv(out / "query_top100_motifs.tsv", sep="\t", index=False)

    w1 = q[q["query"] == "w1"].iloc[0]
    core_ok = bool(
        q.set_index("query").loc["w3", "expected_family_rank"] <= 20
        and q.set_index("query").loc["w4", "expected_family_rank"] <= 20
        and q.set_index("query").loc["w6", "expected_family_rank"] <= 20
        and q.set_index("query").loc["AtPTBP3", "expected_family_rank"] <= 20
    )
    report = [
        "# CNN+JPLE B+E Train/Predict",
        "",
        "Architecture: per-residue CNN encoder -> residual correction head -> frozen JPLE decoder.",
        "",
        "## RNAcompete Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Query Summary",
        "",
        q[["query", "top1_motif", "top5_motifs", "U-rich_rank", "CUUCU-like_rank", "UGUGUG-like_rank", "expected_family_rank", "RNCMPT00434_rank", "top50_U-rich_fraction"]].to_markdown(index=False),
        "",
        "## Key Judgment",
        "",
        f"- w1 U-rich rank: {int(w1['U-rich_rank'])}",
        f"- w1 RNCMPT00434 rank: {int(w1['RNCMPT00434_rank'])}",
        f"- core queries stable: {core_ok}",
    ]
    (out / "BplusE_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
