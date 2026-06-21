#!/usr/bin/env python3
"""Summarize per-residue CNN vs RBPTrace plant LOO metrics without retraining."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_cnn_motif_latent_plant_loo import add_bins, summarize_and_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize CNN vs original RBPTrace plant LOO comparison metrics.")
    p.add_argument("--metrics-tsv", default="results/per_residue_cnn_first_layer/plant_loo_cnn_vs_rbp_trace_metrics.tsv")
    p.add_argument("--output-dir", default="results/per_residue_cnn_first_layer")
    p.add_argument("--per-residue-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--motif-npz", default="data/processed/motif_profiles.npz")
    p.add_argument("--plant-label-tsv", default="results/species_transfer_analysis/plant_nonplant_label_check.tsv")
    p.add_argument("--domain-annotation-tsv", default="results/embedding_domain_audit/domain_annotation_check.tsv")
    p.add_argument("--baseline-exp3-tsv", default="results/species_transfer_analysis/exp3_plant_leave_one_out_metrics.tsv")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--torch-num-threads", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_tsv)
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics TSV: {metrics_path}")
    metrics = pd.read_csv(metrics_path, sep="\t")
    if "rbd_length_bin" not in metrics.columns and "rbd_length" in metrics.columns:
        metrics = add_bins(metrics, "rbd_length", "rbd_length_bin", ["short", "medium", "long"])
    if "nearest_distance_bin" not in metrics.columns and "nearest_train_distance" in metrics.columns:
        metrics = add_bins(metrics, "nearest_train_distance", "nearest_distance_bin", ["close", "medium", "distant"])
    summarize_and_report(metrics, {}, args, out_dir)
    print("[cnn-summary] summary=" + str(out_dir / "cnn_vs_rbp_trace_summary.tsv"))
    print("[cnn-summary] report=" + str(out_dir / "PER_RESIDUE_CNN_FIRST_LAYER_REPORT.md"))
    print("[cnn-summary] judgment=" + str(out_dir / "final_cnn_model_judgment.json"))


if __name__ == "__main__":
    main()
