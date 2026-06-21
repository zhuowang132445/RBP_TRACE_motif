#!/usr/bin/env python3
"""Predict query motifs from a fixed-split per-residue CNN checkpoint."""

from __future__ import annotations

import argparse
import gzip
import json
import os
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
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from cnn_model_utils import PerResidueCnn, RbdEmbeddingDataset, collate_batch, load_h5_features, resolve_path, setup_threads  # noqa: E402


def log(msg: str) -> None:
    print(f"[profile-ranking-cnn-predict] {msg}", flush=True)


def latent_to_profile_torch(
    pred_latent: torch.Tensor,
    svd_components: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
) -> torch.Tensor:
    pred_profile_scaled = pred_latent @ svd_components
    return pred_profile_scaled * scaler_scale + scaler_mean


def short_id(pid: str) -> str:
    return pid.split("|", 1)[0]


def contains_u_rich(kmers: list[str]) -> bool:
    return any("UUUUU" in k for k in kmers)


def contains_cu_rich(kmers: list[str]) -> bool:
    patterns = ("CU", "CUU", "UCU", "CUUC", "CUUCU")
    return any(any(p in k for p in patterns) for k in kmers)


def load_target_motifs(path_text: str | None) -> dict[str, str]:
    if not path_text:
        return {}
    path = resolve_path(path_text, [Path(path_text).name], required=False)
    if path is None:
        return {}
    df = pd.read_csv(path, sep="\t")
    id_col = "query_id" if "query_id" in df.columns else "short_id" if "short_id" in df.columns else None
    motif_col = "target_motif" if "target_motif" in df.columns else "motif" if "motif" in df.columns else None
    if id_col is None or motif_col is None:
        raise ValueError("target motif TSV must contain query_id/short_id and target_motif/motif columns")
    return dict(zip(df[id_col].astype(str), df[motif_col].astype(str)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--query-h5", action="append", required=True)
    ap.add_argument("--query-manifest", action="append", default=[])
    ap.add_argument("--query-label", action="append", default=[])
    ap.add_argument("--target-motif-tsv", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--torch-num-threads", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_threads(args.torch_num_threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)

    ckpt_path = resolve_path(args.checkpoint, [Path(args.checkpoint).name])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    if all(k in ckpt for k in ["scaler_mean", "scaler_scale", "svd_components", "kmer_list"]):
        scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        svd_components = np.asarray(ckpt["svd_components"], dtype=np.float32)
        kmers = np.asarray(ckpt["kmer_list"]).astype(str)
    else:
        preprocess = ckpt["preprocess"]
        scaler_mean = np.asarray(preprocess["scaler_mean"], dtype=np.float32)
        scaler_scale = np.asarray(preprocess["scaler_scale"], dtype=np.float32)
        svd_components = np.asarray(preprocess["svd_components"], dtype=np.float32)
        kmers = np.asarray(ckpt["kmers"]).astype(str)

    svd_components_t = torch.tensor(svd_components, device=device)
    scaler_mean_t = torch.tensor(scaler_mean, device=device)
    scaler_scale_t = torch.tensor(scaler_scale, device=device)

    model = PerResidueCnn(
        cfg["input_dim"],
        cfg["hidden_dim"],
        cfg["latent_dim"],
        cfg["kernel_size"],
        cfg["num_blocks"],
        cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    manifests = list(args.query_manifest)
    while len(manifests) < len(args.query_h5):
        manifests.append(None)
    labels = list(args.query_label)
    while len(labels) < len(args.query_h5):
        labels.append(Path(args.query_h5[len(labels)]).stem)

    query_x: dict[str, np.ndarray] = {}
    query_len: dict[str, int] = {}
    source_label: dict[str, str] = {}
    for h5_text, manifest_text, label in zip(args.query_h5, manifests, labels):
        h5_path = resolve_path(h5_text, [Path(h5_text).name])
        manifest_path = resolve_path(manifest_text, [Path(manifest_text).name], required=False) if manifest_text else None
        x_map, len_map = load_h5_features(h5_path, manifest_path)
        for pid, arr in x_map.items():
            key = pid
            if key in query_x:
                key = f"{label}|{pid}"
            query_x[key] = arr
            query_len[key] = len_map[pid]
            source_label[key] = label

    out = Path(args.output_dir)
    out = ROOT / out if not out.is_absolute() else out
    out.mkdir(parents=True, exist_ok=True)

    dummy_ids = {pid: i for i, pid in enumerate(query_x)}
    dummy_y = np.zeros((len(query_x), cfg["latent_dim"]), dtype=np.float32)
    loader = DataLoader(
        RbdEmbeddingDataset(list(query_x), query_x, dummy_y, dummy_ids),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    target_motifs = load_target_motifs(args.target_motif_tsv)

    top_rows = []
    full_rows = []
    summary_rows = []
    matrix_rows = []
    with torch.no_grad():
        for ids, x, mask, _ in loader:
            pred_latent = model(x.to(device), mask.to(device))
            pred_profile = latent_to_profile_torch(pred_latent, svd_components_t, scaler_mean_t, scaler_scale_t).cpu().numpy()
            for pid, pred in zip(ids, pred_profile):
                order = np.argsort(-pred)
                sid = short_id(pid)
                matrix_rows.append((pid, pred))
                top_kmers = [str(kmers[idx]) for idx in order[: args.top_n]]
                for rank, idx in enumerate(order[: args.top_n], 1):
                    top_rows.append(
                        {
                            "query_id": pid,
                            "short_id": sid,
                            "source": source_label.get(pid, ""),
                            "rank": rank,
                            "kmer": str(kmers[idx]),
                            "pred_score": float(pred[idx]),
                            "rbd_length": query_len.get(pid, np.nan),
                        }
                    )
                for idx in order:
                    full_rows.append({"query_id": pid, "short_id": sid, "kmer": str(kmers[idx]), "pred_score": float(pred[idx])})
                target = target_motifs.get(pid, target_motifs.get(sid, ""))
                target_rank = ""
                if target:
                    hit = np.where(kmers == target)[0]
                    if len(hit):
                        rank_pos = np.empty_like(order)
                        rank_pos[order] = np.arange(1, len(order) + 1)
                        target_rank = int(rank_pos[int(hit[0])])
                summary_rows.append(
                    {
                        "query_id": pid,
                        "short_id": sid,
                        "top1_kmer": top_kmers[0] if top_kmers else "",
                        "top5_kmers": ",".join(top_kmers[:5]),
                        "top10_kmers": ",".join(top_kmers[:10]),
                        "contains_u_rich_top20": contains_u_rich(top_kmers[:20]),
                        "contains_cu_rich_top20": contains_cu_rich(top_kmers[:20]),
                        "target_motif": target,
                        "target_motif_rank_if_provided": target_rank,
                    }
                )

    top_df = pd.DataFrame(top_rows)
    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    top_path = out / "query_predicted_top_7mers_profile_ranking_strong.tsv"
    full_path = out / "query_predicted_full_7mer_profile_ranking_strong.tsv"
    summary_path = out / "query_prediction_summary_profile_ranking_strong.tsv"
    top_df.to_csv(top_path, sep="\t", index=False)
    full_df.to_csv(full_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    # Legacy compatibility outputs.
    top_df.rename(columns={"pred_score": "score"}).to_csv(out / "query_cnn_fixed_split_top_predicted_7mers.tsv", sep="\t", index=False)
    with gzip.open(out / "query_cnn_fixed_split_score_matrix.tsv.gz", "wt") as handle:
        handle.write("query_id\t" + "\t".join(kmers) + "\n")
        for pid, pred in matrix_rows:
            handle.write(pid + "\t" + "\t".join(f"{float(v):.6g}" for v in pred) + "\n")
    prediction_summary = {
        "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("best_epoch"),
        "best_validation_metrics": ckpt.get("best_validation_metrics", {}),
        "selection_metric": ckpt.get("selection_metric"),
        "n_query": len(query_x),
        "top_n": args.top_n,
        "top_path": str(top_path),
        "full_path": str(full_path),
        "summary_path": str(summary_path),
    }
    (out / "query_cnn_fixed_split_prediction_summary.json").write_text(json.dumps(prediction_summary, indent=2) + "\n")
    log(f"top_predictions={top_path}")
    log(f"summary={summary_path}")


if __name__ == "__main__":
    main()
