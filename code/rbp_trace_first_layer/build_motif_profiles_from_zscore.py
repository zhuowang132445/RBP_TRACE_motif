#!/usr/bin/env python3
from __future__ import annotations
import argparse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def read_zscore(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".tsv")]
            if not names:
                raise ValueError(f"No .tsv found in {path}")
            with zf.open(names[0]) as handle:
                return pd.read_csv(handle, sep="\t", index_col=0)
    return pd.read_csv(path, sep="\t", index_col=0)


def main():
    ap = argparse.ArgumentParser(description="Build motif_profiles.npz/tsv from RNAcompete/RBP_TRACE zscore table and protein FASTA metadata.")
    ap.add_argument("--zscore", default="data/original/rbp_trace/processed_reference/zscore_train.zip")
    ap.add_argument("--seq-fasta", default="data/original/rbp_trace/processed_reference/seq_train.fasta")
    ap.add_argument("--metadata", default="data/original/rbp_trace/raw/rnacompete_metadata_eupri.tsv")
    ap.add_argument("--out-npz", default="data/processed/motif_profiles.npz")
    ap.add_argument("--out-tsv", default="data/processed/motif_profiles.tsv")
    args = ap.parse_args()
    z = read_zscore(Path(args.zscore)).dropna()
    kmers = z.index.astype(str).str.upper().str.replace("T", "U", regex=False).to_numpy(dtype=str)
    profile_ids = z.columns.astype(str).to_numpy(dtype=str)
    scores = z.to_numpy(dtype=np.float32).T
    mask = np.isfinite(scores).astype(np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out_npz = Path(args.out_npz); out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, profile_ids=profile_ids, kmers=kmers, zscores=scores, zscore_mask=mask)
    meta_rows = []
    seqs = {}
    current = None; chunks = []
    with open(args.seq_fasta, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line: continue
            if line.startswith(">"):
                if current is not None: seqs[current] = "".join(chunks)
                current = line[1:].split()[0]; chunks = []
            else:
                chunks.append(line)
        if current is not None: seqs[current] = "".join(chunks)
    meta = None
    if Path(args.metadata).exists():
        meta = pd.read_csv(args.metadata, sep="\t")
        if "rnacompete_id" in meta.columns:
            meta = meta.drop_duplicates("rnacompete_id").set_index("rnacompete_id")
    for pid in profile_ids:
        row = {"rnacompete_id": pid, "protein_sequence": seqs.get(pid, ""), "protein_length": len(seqs.get(pid, ""))}
        if meta is not None and pid in meta.index:
            for col in ["gene_name", "protein_id", "tax_name"]:
                if col in meta.columns:
                    row[col] = meta.loc[pid, col]
        meta_rows.append(row)
    pd.DataFrame(meta_rows).to_csv(args.out_tsv, sep="\t", index=False)
    print(f"wrote {out_npz} and {args.out_tsv}: profiles={len(profile_ids)} kmers={len(kmers)}")

if __name__ == "__main__":
    main()
