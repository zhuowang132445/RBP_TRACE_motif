#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from rbp_trace_core.model import RBPTraceFirstLayer


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True); denom[denom == 0] = 1.0
    return x / denom


def load_aligned(embedding_npz: Path, motif_npz: Path, out_dir: Path):
    emb = np.load(embedding_npz, allow_pickle=True)
    protein_ids = np.asarray(emb["protein_ids"]).astype(str)
    x = np.asarray(emb["embeddings"], dtype=np.float32)
    motif = np.load(motif_npz, allow_pickle=True)
    profile_ids = np.asarray(motif["profile_ids"]).astype(str)
    kmers = np.asarray(motif["kmers"]).astype(str)
    y_raw = np.asarray(motif["zscores"], dtype=np.float32)
    if set(protein_ids.tolist()) != set(profile_ids.tolist()):
        raise ValueError("embedding protein_ids and motif profile_ids differ")
    order = np.asarray([{pid: i for i, pid in enumerate(profile_ids)}[pid] for pid in protein_ids], dtype=int)
    y = y_raw[order]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"protein_id": protein_ids, "embedding_row": range(len(protein_ids)), "profile_row": order}).to_csv(out_dir / "id_alignment_check.tsv", sep="\t", index=False)
    return protein_ids, l2_normalize(x), kmers, y


def main():
    ap = argparse.ArgumentParser(description="Train final RBP_TRACE_ESMC_input first-layer motif model from ESMC embeddings and RNAcompete profiles.")
    ap.add_argument("--embedding-npz", default="data/embeddings/rnacompete_domain_merged_esmc_embeddings.npz")
    ap.add_argument("--motif-profiles-npz", default="data/processed/motif_profiles.npz")
    ap.add_argument("--out-model", default="models/final/rbp_trace_first_layer_final_model.rebuilt.npz")
    ap.add_argument("--out-dir", default="results/rebuild_rbp_trace_esmc_input")
    ap.add_argument("--num-eigenvector", type=int, default=122)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--std", type=float, default=0.2)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    protein_ids, x, kmers, y = load_aligned(Path(args.embedding_npz), Path(args.motif_profiles_npz), out_dir)
    model = RBPTraceFirstLayer(num_eigenvector=args.num_eigenvector, threshold=args.threshold, std=args.std)
    model.fit(x, y)
    out_model = Path(args.out_model); out_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_model, y_train=model.y_train, x_train_mean=model.x_train_mean, y_train_mean=model.y_train_mean, w_train=model.w_train, v_train=model.v_train, num_eigenvector=args.num_eigenvector, threshold=args.threshold, std=args.std)
    cfg = vars(args); cfg.update({"n_proteins": int(len(protein_ids)), "embedding_dim": int(x.shape[1]), "n_kmers": int(len(kmers)), "predictor": "RBP_TRACE_ESMC_input"})
    (out_dir / "rebuild_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {out_model}")

if __name__ == "__main__":
    main()
