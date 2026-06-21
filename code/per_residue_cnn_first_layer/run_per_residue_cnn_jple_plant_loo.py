#!/usr/bin/env python3
"""Strict plant LOO for per-residue CNN -> JPLE latent motif prediction."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "per_residue_cnn_first_layer"))

from rbp_trace_core.model import RBPTraceFirstLayer  # noqa: E402
from cnn_model_utils import PerResidueCnn, collate_batch, load_h5_features, setup_threads  # noqa: E402
from run_jple_embedding_variants import (  # noqa: E402
    LatentTargetDataset,
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


def log(msg: str) -> None:
    print(f"[cnn-jple-plant-loo] {msg}", flush=True)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata_simple(a), rankdata_simple(b))


def top_overlap(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    pred_top = set(np.argsort(-pred)[:k].tolist())
    true_top = set(np.argsort(-true)[:k].tolist())
    return len(pred_top & true_top) / float(k)


def predict_query_latent(model: PerResidueCnn, q_map: dict[str, np.ndarray], query_ids: list[str], latent_dim: int, batch_size: int, device: torch.device) -> np.ndarray:
    dummy = np.zeros((len(query_ids), latent_dim), dtype=np.float32)
    loader = DataLoader(
        LatentTargetDataset(query_ids, q_map, dummy),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for _, x, mask, _ in loader:
            rows.append(model(x.to(device), mask.to(device)).cpu().numpy())
    return np.vstack(rows).astype(np.float32)


def train_fold_cnn(
    train_ids: list[str],
    x_map: dict[str, np.ndarray],
    target_w: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> PerResidueCnn:
    dataset = LatentTargetDataset(train_ids, x_map, target_w.astype(np.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    model = PerResidueCnn(
        input_dim=next(iter(x_map.values())).shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=target_w.shape[1],
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for _, x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            log(f"epoch={epoch} train_latent_loss={np.mean(losses):.6g}")
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--query-rice-per-residue-h5", default="results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5")
    p.add_argument("--query-atptbp3-per-residue-h5", default="results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--plant-label-tsv", default="results/species_transfer_analysis/plant_nonplant_label_check.tsv")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer/cnn_jple_plant_loo_45_20260617")
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
    p.add_argument("--max-folds", type=int, default=0, help="0 means all plant folds")
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
    all_ids, all_mean_x, all_y_norm = align_train(mean_ids, mean_x, motif_ids, y_norm)
    mean_x_by_id = {pid: all_mean_x[i] for i, pid in enumerate(all_ids)}

    label = pd.read_csv(resolve(args.plant_label_tsv), sep="\t")
    plant_ids = label.loc[label["is_plant"].astype(str).str.lower().isin(["true", "1", "yes"]), "protein_id"].astype(str).tolist()
    plant_ids = [pid for pid in plant_ids if pid in all_ids and pid in x_map and pid in y_raw_by_id]
    if args.max_folds and args.max_folds > 0:
        plant_ids = plant_ids[: args.max_folds]
    (out / "plant_fold_ids.tsv").write_text("protein_id\n" + "\n".join(plant_ids) + "\n")
    log(f"plant_folds={len(plant_ids)} epochs={args.epochs}")

    q_map, _ = load_h5_features(resolve(args.query_rice_per_residue_h5), None)
    at_map, _ = load_h5_features(resolve(args.query_atptbp3_per_residue_h5), None)
    q_map.update(at_map)
    query_ids = list(q_map.keys())
    query_sum = np.zeros((len(query_ids), len(kmers)), dtype=np.float64)
    query_count = 0

    metrics_path = out / "plant_loo_validation_metrics.tsv"
    done: set[str] = set()
    if args.resume and metrics_path.exists():
        prev = pd.read_csv(metrics_path, sep="\t")
        if "heldout_protein_id" in prev.columns:
            done = set(prev["heldout_protein_id"].astype(str))

    metrics_rows: list[dict[str, object]] = []
    if args.resume and metrics_path.exists():
        metrics_rows = pd.read_csv(metrics_path, sep="\t").to_dict("records")

    for fold_idx, heldout in enumerate(plant_ids, start=1):
        if heldout in done:
            log(f"fold={fold_idx}/{len(plant_ids)} heldout={heldout} skipped resume")
            continue
        log(f"fold={fold_idx}/{len(plant_ids)} heldout={heldout}")
        train_ids = [pid for pid in all_ids if pid != heldout and pid in x_map]
        train_mean_x = l2_normalize(np.vstack([mean_x_by_id[pid] for pid in train_ids]).astype(np.float32))
        train_y_norm = np.vstack([y_norm_by_id[pid] for pid in train_ids]).astype(np.float32)

        jple = RBPTraceFirstLayer(args.num_eigenvector, args.threshold, args.std)
        jple.fit(train_mean_x, train_y_norm)
        target_w = jple.w_train.astype(np.float32)

        model = train_fold_cnn(train_ids, x_map, target_w, args, device)

        heldout_w = predict_query_latent(model, {heldout: x_map[heldout]}, [heldout], target_w.shape[1], args.batch_size, device)
        heldout_pred, heldout_dist, _ = predict_from_latent(heldout_w, jple.w_train.astype(np.float32), jple.y_train.astype(np.float32), args.threshold, args.std)
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

        query_w = predict_query_latent(model, q_map, query_ids, target_w.shape[1], args.batch_size, device)
        query_pred, _, _ = predict_from_latent(query_w, jple.w_train.astype(np.float32), jple.y_train.astype(np.float32), args.threshold, args.std)
        query_sum += standardize_pred(query_pred).astype(np.float64)
        query_count += 1

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
                ckpt_dir / f"fold_{fold_idx:03d}_{heldout}.pt",
            )

        with gzip.open(out / "query_ensemble_running_sum.tsv.gz", "wt") as handle:
            handle.write("query_id\tshort_id\tkmer\tscore_sum\tfold_count\n")
            for qi, qid in enumerate(query_ids):
                sid = "AtPTBP3" if qid.startswith("AtPTBP3") else (qid.split("|original=", 1)[1].split("|", 1)[0] if "|original=" in qid else qid.split("|", 1)[0])
                for kmer, score in zip(kmers, query_sum[qi]):
                    handle.write(f"{qid}\t{sid}\t{kmer}\t{float(score)}\t{query_count}\n")

    if query_count == 0:
        raise RuntimeError("No folds were run; use without --resume or remove existing metrics")
    ensemble_pred = (query_sum / query_count).astype(np.float32)
    write_predictions(
        out,
        "cnn_jple_plant_loo45_ensemble",
        query_ids,
        all_ids,
        kmers,
        ensemble_pred,
        np.full(len(query_ids), np.nan),
        pd.DataFrame(),
        args.top_n,
    )
    summary = {
        "folds_requested": len(plant_ids),
        "folds_completed_this_run": query_count,
        "metrics_path": str(metrics_path),
        "ensemble_summary": str(out / "cnn_jple_plant_loo45_ensemble_query_summary.tsv"),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(f"done folds={query_count}")


if __name__ == "__main__":
    main()
