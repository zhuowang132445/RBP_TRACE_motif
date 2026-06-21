#!/usr/bin/env python3
"""Train current profile-ranking latent CNN on all RNAcompete proteins, then save a final checkpoint."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from cnn_model_utils import PerResidueCnn, load_h5_features, resolve_path, setup_threads  # noqa: E402
from train_cnn_motif_latent_fixed_split import (  # noqa: E402
    MotifProfileDataset,
    build_latent_and_profile_maps,
    build_topk_decoys,
    collate_profile_batch,
    compute_loss_parts,
    filter_finite_ids,
    load_motif,
)


def log(msg: str) -> None:
    print(f"[all-train-profile-ranking-cnn] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--per-residue-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    p.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=199)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--latent-loss-weight", type=float, default=1.0)
    p.add_argument("--profile-loss-weight", type=float, default=0.5)
    p.add_argument("--ranking-loss-weight", type=float, default=0.1)
    p.add_argument("--ranking-margin", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--decoy-n", type=int, default=200)
    p.add_argument("--loss-type", choices=["smooth_l1", "mse"], default="smooth_l1")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260617)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)

    out = Path(args.output_dir)
    out = ROOT / out if not out.is_absolute() else out
    out.mkdir(parents=True, exist_ok=True)

    x_map, _ = load_h5_features(
        resolve_path(args.per_residue_h5, ["rnacompete_rbd_per_residue_esmc.h5"]),
        resolve_path(args.per_residue_manifest, ["rnacompete_rbd_per_residue_manifest.tsv"], required=False),
    )
    motif_ids, y, kmers, id2y = load_motif(resolve_path(args.motif_npz, ["motif_profiles.npz"]))
    train_ids = [pid for pid in motif_ids if pid in x_map]
    train_ids, dropped = filter_finite_ids(train_ids, y, id2y)
    if not train_ids:
        raise RuntimeError("No trainable proteins found")

    y_train = y[[id2y[pid] for pid in train_ids]]
    scaler = StandardScaler(with_mean=True, with_std=True)
    y_train_scaled = scaler.fit_transform(y_train).astype(np.float32)
    latent_dim = max(2, min(args.latent_dim, len(train_ids) - 1, y.shape[1] - 1))
    svd = TruncatedSVD(n_components=latent_dim, random_state=args.seed)
    svd.fit(y_train_scaled)
    latent_by_id, profile_by_id = build_latent_and_profile_maps(train_ids, y, id2y, scaler, svd)
    topk_by_id, decoy_by_id = build_topk_decoys(train_ids, profile_by_id, args.top_k, args.decoy_n, args.seed)

    loader = DataLoader(
        MotifProfileDataset(train_ids, x_map, latent_by_id, profile_by_id, topk_by_id, decoy_by_id),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_profile_batch,
    )
    model = PerResidueCnn(
        next(iter(x_map.values())).shape[1],
        args.hidden_dim,
        latent_dim,
        args.kernel_size,
        args.num_blocks,
        args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    latent_criterion: nn.Module = nn.SmoothL1Loss() if args.loss_type == "smooth_l1" else nn.MSELoss()
    svd_components_t = torch.tensor(svd.components_.astype(np.float32), device=device)
    scaler_mean_t = torch.tensor(scaler.mean_.astype(np.float32), device=device)
    scaler_scale_t = torch.tensor(scaler.scale_.astype(np.float32), device=device)

    log(
        "all_train_n=%d dropped=%d epochs=%d latent_dim=%d weights=(latent %.3g, profile %.3g, ranking %.3g)"
        % (len(train_ids), len(dropped), args.epochs, latent_dim, args.latent_loss_weight, args.profile_loss_weight, args.ranking_loss_weight)
    )
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        parts_rows = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            total_loss, parts, _ = compute_loss_parts(
                model,
                batch,
                device,
                latent_criterion,
                svd_components_t,
                scaler_mean_t,
                scaler_scale_t,
                args,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            parts_rows.append(parts)
        row = {"epoch": epoch}
        for key in parts_rows[0]:
            row[f"train_{key}"] = float(np.mean([r[key] for r in parts_rows]))
        rows.append(row)
        if epoch == 1 or epoch % 10 == 0:
            log(
                "epoch=%d train_total=%.5g latent=%.5g profile=%.5g ranking=%.5g"
                % (epoch, row["train_total_loss"], row["train_latent_loss"], row["train_profile_corr_loss"], row["train_ranking_loss"])
            )

    pd.DataFrame(rows).to_csv(out / "training_log.tsv", sep="\t", index=False)
    with gzip.open(out / "train_ids_used.tsv.gz", "wt") as handle:
        handle.write("protein_id\n")
        for pid in train_ids:
            handle.write(pid + "\n")

    checkpoint = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": {
            "input_dim": int(next(iter(x_map.values())).shape[1]),
            "hidden_dim": args.hidden_dim,
            "latent_dim": latent_dim,
            "kernel_size": args.kernel_size,
            "num_blocks": args.num_blocks,
            "dropout": args.dropout,
        },
        "training_args": vars(args),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "svd_components": svd.components_.astype(np.float32),
        "kmer_list": kmers,
        "best_epoch": args.epochs,
        "best_validation_metrics": {},
        "selection_metric": "none_all_train_fixed_epoch",
        "train_ids": train_ids,
        "preprocess": {
            "svd_components": svd.components_.astype(np.float32),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
        },
        "kmers": kmers,
        "args": vars(args),
    }
    ckpt_path = out / "all_train_profile_ranking_cnn_checkpoint.pt"
    legacy_path = out / "best_model.pt"
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, legacy_path)
    summary = {
        "checkpoint": str(ckpt_path),
        "legacy_checkpoint": str(legacy_path),
        "train_n": len(train_ids),
        "epochs": args.epochs,
        "seed": args.seed,
        "loss_weights": {
            "latent": args.latent_loss_weight,
            "profile": args.profile_loss_weight,
            "ranking": args.ranking_loss_weight,
        },
        "note": "All 348 aligned RNAcompete proteins are used for training. No validation split or generalization estimate.",
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
