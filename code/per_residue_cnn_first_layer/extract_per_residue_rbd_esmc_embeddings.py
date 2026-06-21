#!/usr/bin/env python3
"""Extract per-residue RBD ESMC embeddings into HDF5 for exploratory CNN models."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(f"[per-residue-extract] {msg}", flush=True)


def resolve_path(path_text: str | None, patterns: list[str], required: bool = False) -> Path | None:
    if path_text:
        p = Path(path_text)
        for c in [p, ROOT / p]:
            if c.exists():
                return c
        matches = list(ROOT.rglob(p.name))
        if matches:
            return matches[0]
    for pat in patterns:
        matches = list(ROOT.rglob(pat))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Could not resolve {path_text or patterns}")
    return None


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    cur = None
    seq: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    records[cur] = "".join(seq).replace(" ", "").upper()
                cur = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if cur is not None:
        records[cur] = "".join(seq).replace(" ", "").upper()
    return records


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def parse_ranges(value: Any) -> list[int]:
    text = "" if value is None or pd.isna(value) else str(value)
    out = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            pass
    return out


def build_rbd_sequences_from_annotation(protein_fasta: Path, domain_tsv: Path) -> dict[str, str]:
    full = read_fasta(protein_fasta)
    ann = pd.read_csv(domain_tsv, sep="\t")
    id_col = first_col(ann, ["protein_id", "rice_rbp_id", "rbp_id", "id"])
    start_col = first_col(ann, ["domain_start", "domain_starts", "rbd_start", "start"])
    end_col = first_col(ann, ["domain_end", "domain_ends", "rbd_end", "end"])
    if id_col is None or start_col is None or end_col is None:
        raise ValueError("domain annotation needs protein_id/domain_start/domain_end columns to extract RBD sequence")
    out: dict[str, str] = {}
    for _, row in ann.iterrows():
        pid = str(row[id_col])
        if pid not in full:
            continue
        starts = parse_ranges(row[start_col])
        ends = parse_ranges(row[end_col])
        pieces = []
        for s, e in zip(starts, ends):
            s = max(1, s)
            e = min(len(full[pid]), e)
            if e >= s:
                pieces.append(full[pid][s - 1 : e])
        if pieces:
            out[pid] = "".join(pieces)
    return out


def load_annotation(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["protein_id", "domain_family", "domain_architecture"])
    return pd.read_csv(path, sep="\t")


def annotation_lookup(ann: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if ann.empty:
        return {}
    id_col = first_col(ann, ["protein_id", "rice_rbp_id", "rbp_id", "id"])
    if id_col is None:
        return {}
    fam_col = first_col(ann, ["domain_family", "family", "rbp_family"])
    arch_col = first_col(ann, ["domain_architecture", "architecture"])
    out = {}
    for _, row in ann.iterrows():
        pid = str(row[id_col])
        out[pid] = {
            "domain_family": str(row[fam_col]) if fam_col else "Unknown",
            "domain_architecture": str(row[arch_col]) if arch_col else "Unknown",
        }
    return out


def safe_key(pid: str) -> str:
    return pid.replace("/", "__slash__")


def find_existing_token(path_or_dir: Path | None, pid: str) -> Path | None:
    if path_or_dir is None:
        return None
    if path_or_dir.is_file():
        return path_or_dir
    matches = sorted(path_or_dir.glob(f"{pid}*.pt")) + sorted(path_or_dir.glob(f"{pid}*.npy"))
    return matches[0] if matches else None


def load_token_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".pt":
        import torch
        arr = torch.load(path, map_location="cpu")
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().float().numpy()
        else:
            arr = np.asarray(arr)
    else:
        arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"per-residue embedding must be 2D LxD, got shape {arr.shape} from {path}")
    return arr


def setup_torch_threads(n: int) -> Any:
    import torch
    torch.set_num_threads(max(1, int(n)))
    torch.set_num_interop_threads(max(1, int(n)))
    return torch


def load_transformers_model(model_dir: Path, device: str, dtype_name: str, torch_num_threads: int):
    torch = setup_torch_threads(torch_num_threads)
    from transformers import AutoModel, AutoTokenizer
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"
    dtype = torch.float32
    if device != "cpu" and dtype_name == "float16":
        dtype = torch.float16
    elif device != "cpu" and dtype_name == "bfloat16":
        dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, torch_dtype=dtype)
    model.eval().to(device)
    return torch, tokenizer, model, device


def embed_sequence_per_residue(torch, tokenizer, model, seq: str, device: str, max_len: int) -> np.ndarray:
    if len(seq) > max_len:
        seq = seq[:max_len]
    with torch.inference_mode():
        encoded = tokenizer([seq], return_tensors="pt", padding=True, truncation=False, return_special_tokens_mask=True)
        special = encoded.pop("special_tokens_mask")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        special = special.to(device).bool()
        outputs = model(**encoded)
        token_mask = encoded["attention_mask"].bool() & (~special)
        mat = outputs.last_hidden_state[0][token_mask[0]].detach().cpu().float().numpy().astype(np.float32)
    return mat


def write_h5(records: list[tuple[str, str, np.ndarray, str, str, str, str]], output_h5: Path, output_manifest: Path) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_h5, "w") as h5:
        emb_grp = h5.create_group("embeddings")
        seq_grp = h5.create_group("sequences")
        meta_grp = h5.create_group("metadata")
        pids = []
        embedding_dim = 0
        for pid, seq, mat, fam, arch, status, msg in records:
            pids.append(pid)
            if status == "ok":
                emb_grp.create_dataset(safe_key(pid), data=mat, compression="gzip", compression_opts=4)
                seq_grp.create_dataset(safe_key(pid), data=seq, dtype=str_dt)
                embedding_dim = int(mat.shape[1])
                seq_len = int(mat.shape[0])
            else:
                seq_len = len(seq)
            rows.append({
                "protein_id": pid,
                "sequence_length": seq_len,
                "embedding_dim": embedding_dim if status == "ok" else "",
                "domain_family": fam,
                "domain_architecture": arch,
                "status": status,
                "message": msg,
            })
        meta_grp.create_dataset("protein_ids", data=np.asarray(pids, dtype=object), dtype=str_dt)
        meta_grp.create_dataset("embedding_dim", data=int(embedding_dim))
    pd.DataFrame(rows).to_csv(output_manifest, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract per-residue RBD ESMC embeddings to HDF5.")
    p.add_argument("--rbd-fasta", default=None)
    p.add_argument("--protein-fasta", default=None)
    p.add_argument("--metadata-tsv", default="data/original/rbp_trace/raw/rnacompete_metadata_eupri.tsv")
    p.add_argument("--domain-annotation-tsv", default="results/embedding_domain_audit/domain_annotation_check.tsv")
    p.add_argument("--existing-token-dir", default=None, help="Directory containing existing 2D .pt/.npy per-residue embeddings named by protein_id.")
    p.add_argument("--model-dir", default=None, help="Local ESMC/transformers model dir. Required if no existing token files are supplied.")
    p.add_argument("--output-h5", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5")
    p.add_argument("--output-manifest", default="results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=1, help="Kept for CLI consistency; extraction is intentionally serial.")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--max-len", type=int, default=2046)
    p.add_argument("--max-proteins", type=int, default=None)
    p.add_argument("--protein-id-list", default=None, help="Optional text file with protein IDs to extract, one per line.")
    p.add_argument("--quick-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rbd_fasta = resolve_path(args.rbd_fasta, ["*rbd*.fasta", "*domain*merged*.fasta", "seq_train.fasta"], required=False)
    domain_path = resolve_path(args.domain_annotation_tsv, ["domain_annotation_check.tsv"], required=False)
    existing_dir = Path(args.existing_token_dir) if args.existing_token_dir else None
    if existing_dir is not None and not existing_dir.exists():
        existing_dir = resolve_path(args.existing_token_dir, [Path(args.existing_token_dir).name], required=False)

    if rbd_fasta is not None:
        seqs = read_fasta(rbd_fasta)
        log(f"Using RBD fasta: {rbd_fasta} ({len(seqs)} sequences)")
    else:
        protein_fasta = resolve_path(args.protein_fasta, ["seq_train.fasta", "*.fasta"], required=False)
        if protein_fasta is None or domain_path is None:
            raise SystemExit("Could not find RBD sequences. Provide --rbd-fasta or --protein-fasta + --domain-annotation-tsv.")
        seqs = build_rbd_sequences_from_annotation(protein_fasta, domain_path)
        log(f"Extracted RBD sequences from {protein_fasta} and {domain_path}: {len(seqs)} sequences")
    if args.protein_id_list:
        id_path = Path(args.protein_id_list)
        wanted = [line.strip().split()[0] for line in id_path.read_text().splitlines() if line.strip() and not line.startswith("#")]
        missing = [pid for pid in wanted if pid not in seqs]
        seqs = {pid: seqs[pid] for pid in wanted if pid in seqs}
        if missing:
            log(f"protein-id-list missing {len(missing)} IDs from fasta; first_missing={missing[:5]}")
    if args.quick_test:
        args.max_proteins = args.max_proteins or None
    if args.max_proteins:
        seqs = dict(list(seqs.items())[: args.max_proteins])

    ann = annotation_lookup(load_annotation(domain_path))
    model_tuple = None
    if existing_dir is None:
        if not args.model_dir:
            raise SystemExit(
                "No existing per-residue token embeddings found/provided. Provide --existing-token-dir with 2D .pt/.npy files, "
                "or provide --model-dir for local ESMC/transformers extraction. This script will not use pooled 1152D embeddings as a CNN substitute."
            )
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise SystemExit(f"--model-dir not found: {model_dir}")
        model_tuple = load_transformers_model(model_dir, args.device, args.dtype, args.torch_num_threads)

    records = []
    for i, (pid, seq) in enumerate(seqs.items(), start=1):
        fam = ann.get(pid, {}).get("domain_family", "Unknown")
        arch = ann.get(pid, {}).get("domain_architecture", "Unknown")
        try:
            token_path = find_existing_token(existing_dir, pid)
            if token_path is not None:
                mat = load_token_matrix(token_path)
                msg = f"loaded_existing_token_embedding:{token_path}"
            else:
                torch, tokenizer, model, device = model_tuple
                mat = embed_sequence_per_residue(torch, tokenizer, model, seq, device, args.max_len)
                msg = "extracted_with_local_transformers_model"
            if mat.shape[0] != min(len(seq), args.max_len):
                msg += f";length_warning_seq_{len(seq)}_embedding_{mat.shape[0]}"
            records.append((pid, seq[: args.max_len], mat, fam, arch, "ok", msg))
        except Exception as exc:
            records.append((pid, seq, np.zeros((0, 0), dtype=np.float32), fam, arch, "failed", f"{type(exc).__name__}:{str(exc)[:300]}"))
        if i % 25 == 0:
            log(f"processed {i}/{len(seqs)}")
    write_h5(records, Path(args.output_h5), Path(args.output_manifest))
    ok = sum(r[5] == "ok" for r in records)
    log(f"Wrote H5: {args.output_h5}")
    log(f"Wrote manifest: {args.output_manifest}")
    log(f"status ok={ok} failed={len(records)-ok}")


if __name__ == "__main__":
    main()
