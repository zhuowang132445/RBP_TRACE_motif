#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, json, sys
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


def top_indices(v, n):
    n = min(int(n), len(v)); idx = np.argpartition(-v, n - 1)[:n]
    return idx[np.argsort(-v[idx])]


def load_model(path: Path) -> RBPTraceFirstLayer:
    z = np.load(path, allow_pickle=True)
    model = RBPTraceFirstLayer(num_eigenvector=int(z["num_eigenvector"]) if "num_eigenvector" in z.files else 122, threshold=float(z["threshold"]) if "threshold" in z.files else 0.01, std=float(z["std"]) if "std" in z.files else 0.2)
    model.load(y_train=np.asarray(z["y_train"], dtype=np.float32), x_train_mean=np.asarray(z["x_train_mean"], dtype=np.float32), y_train_mean=np.asarray(z["y_train_mean"], dtype=np.float32), w_train=np.asarray(z["w_train"], dtype=np.float32), v_train=np.asarray(z["v_train"], dtype=np.float32))
    return model


def main():
    ap = argparse.ArgumentParser(description="Predict 7-mer motif scores with final RBP_TRACE_ESMC_input model from query ESMC embedding npz.")
    ap.add_argument("--query-embedding-npz", required=True, help="npz with protein_ids and embeddings arrays, e.g. domain_merged ESMC embeddings")
    ap.add_argument("--motif-profiles-npz", default="data/processed/motif_profiles.npz")
    ap.add_argument("--model", default="models/final/rbp_trace_first_layer_final_model.npz")
    ap.add_argument("--output-dir", default="results/rbp_trace_prediction")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    motif = np.load(args.motif_profiles_npz, allow_pickle=True)
    kmers = np.asarray(motif["kmers"]).astype(str)
    q = np.load(args.query_embedding_npz, allow_pickle=True)
    ids = np.asarray(q["protein_ids"]).astype(str)
    x = l2_normalize(np.asarray(q["embeddings"], dtype=np.float32))
    model = load_model(Path(args.model))
    pred, _, neigh = model.predict_protein(x)
    std = np.std(pred, axis=1); std[std == 0] = 1.0
    pred = (pred / std[:, None]).astype(np.float32)
    with gzip.open(out / "rbp_trace_7mer_scores.tsv.gz", "wt") as handle:
        handle.write("query_id\tkmer\tscore\n")
        for pid, row in zip(ids, pred):
            for kmer, score in zip(kmers, row):
                handle.write(f"{pid}\t{kmer}\t{float(score)}\n")
    rows = []
    for pid, row in zip(ids, pred):
        for rank, idx in enumerate(top_indices(row, args.top_n), start=1):
            rows.append({"query_id": pid, "rank": rank, "kmer": str(kmers[idx]), "score": float(row[idx])})
    pd.DataFrame(rows).to_csv(out / "top_predicted_7mers.tsv", sep="\t", index=False)
    try:
        neigh.to_csv(out / "rbp_trace_neighbor_table.tsv", sep="\t", index=False)
    except Exception:
        pass
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"wrote predictions to {out}")

if __name__ == "__main__":
    main()
