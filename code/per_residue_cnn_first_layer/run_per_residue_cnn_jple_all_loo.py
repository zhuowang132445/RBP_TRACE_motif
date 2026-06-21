#!/usr/bin/env python3
"""All-protein LOO for per-residue CNN -> JPLE latent motif prediction."""

from __future__ import annotations

import argparse
import gzip
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
    print(f"[cnn-jple-all-loo] {msg}", flush=True)


def short_id(qid: str) -> str:
    if qid.startswith("AtPTBP3"):
        return "AtPTBP3"
    if "|original=" in qid:
        return qid.split("|original=", 1)[1].split("|", 1)[0]
    return qid.split("|", 1)[0]


def load_state(path: Path, query_shape: tuple[int, int]) -> tuple[np.ndarray, int, set[str]]:
    if not path.exists():
        return np.zeros(query_shape, dtype=np.float64), 0, set()
    z = np.load(path, allow_pickle=True)
    return (
        np.asarray(z["query_sum"], dtype=np.float64),
        int(z["query_count"]),
        set(np.asarray(z["completed_heldout_ids"]).astype(str).tolist()),
    )


def save_state(path: Path, query_sum: np.ndarray, query_count: int, completed: set[str]) -> None:
    np.savez_compressed(
        path,
        query_sum=query_sum.astype(np.float32),
        query_count=np.asarray(query_count, dtype=np.int64),
        completed_heldout_ids=np.asarray(sorted(completed), dtype=str),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/cnn_jple_all_loo_20260617")
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
    p.add_argument("--max-folds", type=int, default=0, help="0 means all folds")
    p.add_argument("--start-fold", type=int, default=1, help="1-based inclusive")
    p.add_argument("--end-fold", type=int, default=0, help="1-based inclusive; 0 means final fold")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-fold-checkpoints", action="store_true")
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
    all_ids = [pid for pid in all_ids if pid in x_map and pid in y_raw_by_id]
    mean_x_by_id = {pid: all_mean_x[i] for i, pid in enumerate(align_train(mean_ids, mean_x, motif_ids, y_norm)[0]) if pid in all_ids}

    fold_ids = list(all_ids)
    total_available = len(fold_ids)
    if args.max_folds and args.max_folds > 0:
        fold_ids = fold_ids[: args.max_folds]
    end_fold = args.end_fold if args.end_fold and args.end_fold > 0 else len(fold_ids)
    fold_items = [(i, pid) for i, pid in enumerate(fold_ids, start=1) if args.start_fold <= i <= end_fold]
    pd.DataFrame({"fold": list(range(1, len(fold_ids) + 1)), "heldout_protein_id": fold_ids}).to_csv(out / "all_loo_fold_ids.tsv", sep="\t", index=False)
    log(f"available_folds={total_available} selected_folds={len(fold_items)} epochs={args.epochs}")

    q_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    at_map, _ = load_h5_features(resolve(args.query_atptbp3_per_residue_h5), None)
    q_map.update(at_map)
    query_ids = list(q_map.keys())
    state_path = out / "query_ensemble_state.npz"
    if args.resume:
        query_sum, query_count, completed = load_state(state_path, (len(query_ids), len(kmers)))
    else:
        query_sum = np.zeros((len(query_ids), len(kmers)), dtype=np.float64)
        query_count = 0
        completed = set()

    metrics_path = out / "all_loo_validation_metrics.tsv"
    metrics_rows: list[dict[str, object]] = []
    if args.resume and metrics_path.exists():
        metrics_rows = pd.read_csv(metrics_path, sep="\t").to_dict("records")

    for fold_idx, heldout in fold_items:
        if heldout in completed:
            log(f"fold={fold_idx}/{len(fold_ids)} heldout={heldout} skipped")
            continue
        log(f"fold={fold_idx}/{len(fold_ids)} heldout={heldout}")
        train_ids = [pid for pid in all_ids if pid != heldout]
        train_mean_x = l2_normalize(np.vstack([mean_x_by_id[pid] for pid in train_ids]).astype(np.float32))
        train_y_norm = np.vstack([y_norm_by_id[pid] for pid in train_ids]).astype(np.float32)

        jple = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
        jple.fit(train_mean_x, train_y_norm)
        model = train_fold_cnn(train_ids, x_map, jple.w_train.astype(np.float32), args, device)

        heldout_w = predict_query_latent(model, {heldout: x_map[heldout]}, [heldout], jple.w_train.shape[1], args.batch_size, device)
        heldout_pred, heldout_dist, _ = predict_from_latent(
            heldout_w,
            jple.w_train.astype(np.float32),
            jple.y_train.astype(np.float32),
            args.threshold,
            args.std,
        )
        heldout_pred = standardize_pred(heldout_pred)[0]
        true_raw = y_raw_by_id[heldout]
        metrics_rows.append(
            {
                "fold": fold_idx,
                "heldout_protein_id": heldout,
                "pearson": pearson(heldout_pred, true_raw),
                "spearman": spearman(heldout_pred, true_raw),
                "top20_recovery": top_overlap(heldout_pred, true_raw, 20),
                "top50_recovery": top_overlap(heldout_pred, true_raw, 50),
                "min_latent_distance": float(heldout_dist[0]),
                "train_n": len(train_ids),
            }
        )
        pd.DataFrame(metrics_rows).to_csv(metrics_path, sep="\t", index=False)

        query_w = predict_query_latent(model, q_map, query_ids, jple.w_train.shape[1], args.batch_size, device)
        query_pred, _, _ = predict_from_latent(
            query_w,
            jple.w_train.astype(np.float32),
            jple.y_train.astype(np.float32),
            args.threshold,
            args.std,
        )
        query_sum += standardize_pred(query_pred).astype(np.float64)
        query_count += 1
        completed.add(heldout)
        save_state(state_path, query_sum, query_count, completed)

        if args.save_fold_checkpoints:
            ckpt_dir = out / "fold_checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(
                {
                    "fold": fold_idx,
                    "heldout_protein_id": heldout,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "train_ids": train_ids,
                    "jple_w_train": jple.w_train.astype(np.float32),
                    "jple_y_train": jple.y_train.astype(np.float32),
                    "kmers": kmers,
                    "args": vars(args),
                },
                ckpt_dir / f"fold_{fold_idx:04d}_{heldout}.pt",
            )

    if query_count > 0:
        ensemble_pred = (query_sum / query_count).astype(np.float32)
        write_predictions(
            out,
            "cnn_jple_all_loo_ensemble",
            query_ids,
            all_ids,
            kmers,
            ensemble_pred,
            np.full(len(query_ids), np.nan),
            pd.DataFrame(),
            args.top_n,
        )
        with gzip.open(out / "query_ensemble_running_sum.tsv.gz", "wt") as handle:
            handle.write("query_id\tshort_id\tkmer\tscore_sum\tfold_count\n")
            for qi, qid in enumerate(query_ids):
                sid = short_id(qid)
                for kmer, score in zip(kmers, query_sum[qi]):
                    handle.write(f"{qid}\t{sid}\t{kmer}\t{float(score)}\t{query_count}\n")

    summary = {
        "available_folds": total_available,
        "selected_folds": len(fold_items),
        "completed_folds_in_state": query_count,
        "metrics_path": str(metrics_path),
        "ensemble_summary": str(out / "cnn_jple_all_loo_ensemble_query_summary.tsv"),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(f"done completed_folds_in_state={query_count}")


if __name__ == "__main__":
    main()
