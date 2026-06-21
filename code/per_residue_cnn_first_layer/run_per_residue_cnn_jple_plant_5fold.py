#!/usr/bin/env python3
"""Plant 5-fold CV for per-residue CNN -> JPLE latent motif prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))

from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402
from cnn_model_utils import load_h5_features, setup_threads  # noqa: E402
from run_jple_embedding_variants import (  # noqa: E402
    align_train,
    l2_normalize,
    load_h5_mean_embeddings,
    load_motif,
    predict_from_latent,
    resolve,
    row_l2_normalize,
    seed_all,
    standardize_pred,
    write_predictions,
)
from run_per_residue_cnn_jple_plant_loo import (  # noqa: E402
    pearson,
    predict_query_latent,
    spearman,
    top_overlap,
    train_fold_cnn,
)


def log(msg: str) -> None:
    print(f"[cnn-jple-plant-5fold] {msg}", flush=True)


def short_id(qid: str) -> str:
    if qid.startswith("AtPTBP3"):
        return "AtPTBP3"
    if "|original=" in qid:
        return qid.split("|original=", 1)[1].split("|", 1)[0]
    return qid.split("|", 1)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--plant-label-tsv", default="results/species_transfer_analysis/plant_nonplant_label_check.tsv")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/cnn_jple_plant_5fold_20260617")
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--num-eigenvector", type=int, default=122)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--std", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    device = torch.device(args.device)

    out = Path(args.output_dir)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    motif_ids, y_raw, kmers = load_motif(resolve(args.motif_npz))
    y_norm = row_l2_normalize(y_raw)
    y_raw_by_id = {pid: y_raw[i].astype(np.float32) for i, pid in enumerate(motif_ids)}
    y_norm_by_id = {pid: y_norm[i].astype(np.float32) for i, pid in enumerate(motif_ids)}

    x_map, _ = load_h5_features(resolve(args.train_per_residue_h5), None)
    mean_ids, mean_x = load_h5_mean_embeddings(resolve(args.train_per_residue_h5))
    all_ids, all_mean_x, _ = align_train(mean_ids, mean_x, motif_ids, y_norm)
    mean_x_by_id = {pid: all_mean_x[i] for i, pid in enumerate(all_ids)}

    label = pd.read_csv(resolve(args.plant_label_tsv), sep="\t")
    plant_ids = label.loc[label["is_plant"].astype(str).str.lower().isin(["true", "1", "yes"]), "protein_id"].astype(str).tolist()
    plant_ids = [pid for pid in plant_ids if pid in all_ids and pid in x_map and pid in y_raw_by_id]
    rng = np.random.default_rng(args.seed)
    plant_ids = list(np.asarray(plant_ids)[rng.permutation(len(plant_ids))])
    folds = [list(x) for x in np.array_split(np.asarray(plant_ids, dtype=str), args.num_folds)]
    fold_rows = []
    for i, fold in enumerate(folds, start=1):
        for pid in fold:
            fold_rows.append({"fold": i, "protein_id": pid})
    pd.DataFrame(fold_rows).to_csv(out / "plant_5fold_split.tsv", sep="\t", index=False)
    log(f"plant_n={len(plant_ids)} folds={len(folds)} fold_sizes={[len(f) for f in folds]}")

    q_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    at_map, _ = load_h5_features(resolve(args.query_atptbp3_per_residue_h5), None)
    q_map.update(at_map)
    query_ids = list(q_map.keys())
    query_sum = np.zeros((len(query_ids), len(kmers)), dtype=np.float64)

    metric_rows: list[dict[str, object]] = []
    for fold_idx, heldout_ids in enumerate(folds, start=1):
        heldout_set = set(map(str, heldout_ids))
        train_ids = [pid for pid in all_ids if pid not in heldout_set and pid in x_map]
        log(f"fold={fold_idx}/{len(folds)} heldout_n={len(heldout_ids)} train_n={len(train_ids)}")
        train_mean_x = l2_normalize(np.vstack([mean_x_by_id[pid] for pid in train_ids]).astype(np.float32))
        train_y_norm = np.vstack([y_norm_by_id[pid] for pid in train_ids]).astype(np.float32)

        jple = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
        jple.fit(train_mean_x, train_y_norm)
        model = train_fold_cnn(train_ids, x_map, jple.w_train.astype(np.float32), args, device)

        heldout_map = {pid: x_map[pid] for pid in heldout_ids}
        heldout_w = predict_query_latent(model, heldout_map, list(heldout_ids), jple.w_train.shape[1], args.batch_size, device)
        heldout_pred, heldout_dist, _ = predict_from_latent(
            heldout_w,
            jple.w_train.astype(np.float32),
            jple.y_train.astype(np.float32),
            args.threshold,
            args.std,
        )
        heldout_pred = standardize_pred(heldout_pred)
        for i, pid in enumerate(heldout_ids):
            true_raw = y_raw_by_id[pid]
            metric_rows.append(
                {
                    "fold": fold_idx,
                    "heldout_protein_id": pid,
                    "pearson": pearson(heldout_pred[i], true_raw),
                    "spearman": spearman(heldout_pred[i], true_raw),
                    "top20_recovery": top_overlap(heldout_pred[i], true_raw, 20),
                    "top50_recovery": top_overlap(heldout_pred[i], true_raw, 50),
                    "min_latent_distance": float(heldout_dist[i]),
                    "train_n": len(train_ids),
                    "heldout_n": len(heldout_ids),
                }
            )
        pd.DataFrame(metric_rows).to_csv(out / "plant_5fold_validation_metrics.tsv", sep="\t", index=False)

        query_w = predict_query_latent(model, q_map, query_ids, jple.w_train.shape[1], args.batch_size, device)
        query_pred, _, _ = predict_from_latent(
            query_w,
            jple.w_train.astype(np.float32),
            jple.y_train.astype(np.float32),
            args.threshold,
            args.std,
        )
        query_sum += standardize_pred(query_pred).astype(np.float64)

    ensemble_pred = (query_sum / len(folds)).astype(np.float32)
    write_predictions(
        out,
        "cnn_jple_plant_5fold_ensemble",
        query_ids,
        all_ids,
        kmers,
        ensemble_pred,
        np.full(len(query_ids), np.nan),
        pd.DataFrame(),
        args.top_n,
    )
    summary = {
        "folds": len(folds),
        "fold_sizes": [len(f) for f in folds],
        "metrics_path": str(out / "plant_5fold_validation_metrics.tsv"),
        "ensemble_summary": str(out / "cnn_jple_plant_5fold_ensemble_query_summary.tsv"),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log("done")


if __name__ == "__main__":
    main()
