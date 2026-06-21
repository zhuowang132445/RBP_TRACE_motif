#!/public/home/wz/anaconda3/bin/python
"""Extract ESMC full-length and RBD/domain embeddings.

Frozen inference only. No training, no checkpoint modification.

Outputs one mean-pooled embedding per protein per embedding scope:
  - full_length: full protein sequence, with long sequences handled by windows
  - domain_merged: HMM-detected RBD/domain regions merged per protein
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL_DIR = "models/esmc/ESMC-600M"
DEFAULT_OUTPUT_DIR = "results/esmc_embeddings_rnacompete_clip"
DEFAULT_RNACOMPETE_FASTA = (
    "/public/home/wz/workplace/cursor/modle/"
    "RBP_TRACE_portable_full_20260607_175714/data/raw/motif_train/seq_train.fasta"
)
DEFAULT_CLIP_FASTA = (
    "/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/"
    "07_training_data/window_dataset_top1000_strict_fixed/rbp_proteins_for_domain_scan.fasta"
)
DEFAULT_CLIP_ALL_FASTA = (
    "/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/"
    "04_rbp_protein/MuSIC_RBP_proteins_all.strict_repaired.fasta"
)
DEFAULT_HMM_PATH = "data/original/rbp_trace/processed_reference/domain_rbp.hmm"
DEFAULT_RBP_TRACE_PATH = "code/rbp_trace_core"
STANDARD_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYX]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen ESMC embeddings for RNAcompete and CLIP proteins.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--rnacompete-fasta", default=DEFAULT_RNACOMPETE_FASTA)
    parser.add_argument("--clip-fasta", default=DEFAULT_CLIP_FASTA)
    parser.add_argument("--extra-fasta", action="append", default=[], metavar="DATASET:PATH")
    parser.add_argument("--include-clip-all-strict-repaired", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-scope", choices=["full", "domain", "both"], default="both")
    parser.add_argument("--hmm-path", default=DEFAULT_HMM_PATH, help="Comma-separated HMM files for domain scan.")
    parser.add_argument("--rbp-trace-path", default=DEFAULT_RBP_TRACE_PATH)
    parser.add_argument("--domain-types", default="all", help="Comma-separated HMM domain names, or 'all'.")
    parser.add_argument("--domain-flank", type=int, default=15)
    parser.add_argument("--max-aa-per-window", type=int, default=2046, help="ESMC context-safe AA window length.")
    parser.add_argument("--window-overlap", type=int, default=0)
    parser.add_argument("--long-sequence-policy", choices=["window_mean", "truncate", "skip"], default="window_mean")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--save-per-token", action="store_true", help="Only saved for single-window items; can be large.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failed", action="store_true", help="Return 0 even if some embeddings failed.")
    return parser.parse_args()


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_id: Optional[str] = None
    chunks: List[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                current_id = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records.append((current_id, "".join(chunks)))
    return records


def clean_sequence(seq: str) -> Tuple[str, str]:
    original = seq.upper().replace("*", "")
    cleaned = STANDARD_AA_RE.sub("X", original)
    note = "" if cleaned == original else "non_20aa_or_non_X_replaced_by_X"
    return cleaned, note


def safe_name(*parts: str) -> str:
    raw = "\t".join(parts)
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", parts[-1])[:80].strip("._") or "protein"
    return f"{stem}.{digest}"


def dtype_from_name(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_model(model_dir: Path, device: str, dtype_name: str):
    import torch

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing ESMC dependencies. Install with:\n"
            "  /public/home/wz/anaconda3/bin/python -m pip install --user --ignore-requires-python "
            "\"esm@git+https://github.com/Biohub/esm.git@main\"\n"
        ) from exc

    torch_dtype = dtype_from_name(dtype_name)
    if device == "cpu":
        torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    model.eval().to(device)
    return tokenizer, model


def pool_mean_hidden(last_hidden, encoded):
    attention_mask = encoded["attention_mask"].bool()
    special_mask = encoded.get("special_tokens_mask")
    token_mask = attention_mask & (~special_mask.bool()) if special_mask is not None else attention_mask
    token_mask_f = token_mask.unsqueeze(-1).to(last_hidden.dtype)
    denom = token_mask_f.sum(dim=1).clamp_min(1.0)
    return (last_hidden * token_mask_f).sum(dim=1) / denom, token_mask


def make_windows(seq: str, max_len: int, overlap: int, policy: str) -> Tuple[List[str], List[int], str]:
    if len(seq) <= max_len:
        return [seq], [len(seq)], "single_window"
    if policy == "skip":
        return [], [], f"skipped_long_sequence_len_{len(seq)}"
    if policy == "truncate":
        return [seq[:max_len]], [max_len], f"truncated_from_{len(seq)}_to_{max_len}"
    if overlap >= max_len:
        raise ValueError("--window-overlap must be smaller than --max-aa-per-window")
    step = max_len - overlap
    windows = [seq[start : start + max_len] for start in range(0, len(seq), step)]
    weights = [len(window) for window in windows]
    return windows, weights, f"window_mean_{len(windows)}_windows_len_{len(seq)}"


def embed_sequence(tokenizer, model, seq: str, device, max_len: int, overlap: int, policy: str, batch_size: int):
    import torch

    windows, weights, note = make_windows(seq, max_len, overlap, policy)
    if not windows:
        return None, note, None
    pooled_chunks: List[np.ndarray] = []
    per_token_hidden = None
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model_inputs = {
                key: value for key, value in encoded.items()
                if key != "special_tokens_mask"
            }
            outputs = model(**model_inputs)
            pooled, token_mask = pool_mean_hidden(outputs.last_hidden_state, encoded)
            pooled_chunks.append(pooled.detach().cpu().float().numpy())
            if len(windows) == 1:
                per_token_hidden = outputs.last_hidden_state[0][token_mask[0]].detach().cpu().float()
    chunk_mat = np.concatenate(pooled_chunks, axis=0)
    weights_np = np.array(weights, dtype=np.float32)
    emb = (chunk_mat * weights_np[:, None]).sum(axis=0) / weights_np.sum()
    return emb.astype(np.float32), note, per_token_hidden


def _to_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def scan_domains(records: List[Tuple[str, str]], hmm_path: Path, rbp_trace_path: Path) -> Dict[str, List[Dict]]:
    del rbp_trace_path
    import pyhmmer

    hmm_paths = [Path(item) for item in str(hmm_path).split(",") if item.strip()]
    alphabet = pyhmmer.easel.Alphabet.amino()
    seqs = []
    for protein_id, seq in records:
        text_seq = pyhmmer.easel.TextSequence(name=protein_id.encode(), sequence=seq)
        seqs.append(text_seq.digitize(alphabet))

    out: Dict[str, List[Dict]] = {}
    seen = set()
    for one_hmm_path in hmm_paths:
        with pyhmmer.plan7.HMMFile(str(one_hmm_path)) as hmm_file:
            hmms = list(hmm_file)
        source = one_hmm_path.name
        for hits in pyhmmer.hmmscan(seqs, hmms, E=0.01, domE=0.01):
            for hit in hits:
                for domain in hit.domains:
                    if not domain.reported:
                        continue
                    protein_id = _to_text(domain.alignment.target_name)
                    domain_type = _to_text(domain.alignment.hmm_name)
                    start = int(domain.env_from)
                    end = int(domain.env_to)
                    if end < start:
                        start, end = end, start
                    key = (protein_id, source, domain_type, start, end)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.setdefault(protein_id, []).append(
                        {
                            "domain_source": source,
                            "domain_type": domain_type,
                            "domain_start": start,
                            "domain_end": end,
                            "domain_seq": "",
                        }
                    )
    for protein_id in out:
        out[protein_id] = sorted(
            out[protein_id],
            key=lambda row: (row["domain_start"], row["domain_end"], row["domain_source"], row["domain_type"]),
        )
    return out

def merge_domain_regions(seq: str, domains: List[Dict], flank: int, selected_domains: Optional[set]) -> Optional[Dict]:
    filtered = [d for d in domains if selected_domains is None or d["domain_type"] in selected_domains]
    if not filtered:
        return None
    intervals = []
    domain_sources = []
    domain_types = []
    raw_starts = []
    raw_ends = []
    for d in sorted(filtered, key=lambda x: x["domain_start"]):
        start0 = max(0, d["domain_start"] - 1 - flank)
        end0 = min(len(seq), d["domain_end"] + flank)
        intervals.append([start0, end0])
        domain_sources.append(d.get("domain_source", ""))
        domain_types.append(d["domain_type"])
        raw_starts.append(str(d["domain_start"]))
        raw_ends.append(str(d["domain_end"]))
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    merged_seq = "".join(seq[start:end] for start, end in merged)
    return {
        "sequence": merged_seq,
        "domain_source": ";".join(domain_sources),
        "domain_type": ";".join(domain_types),
        "domain_start": ";".join(raw_starts),
        "domain_end": ";".join(raw_ends),
        "domain_region_start": ";".join(str(start + 1) for start, _ in merged),
        "domain_region_end": ";".join(str(end) for _, end in merged),
        "domain_region_count": len(merged),
    }


def build_inputs(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    inputs = [("rnacompete", Path(args.rnacompete_fasta)), ("clip_music_top1000_strict", Path(args.clip_fasta))]
    if args.include_clip_all_strict_repaired:
        inputs.append(("clip_music_all_strict_repaired", Path(DEFAULT_CLIP_ALL_FASTA)))
    for item in args.extra_fasta:
        if ":" not in item:
            raise SystemExit(f"--extra-fasta must be DATASET:PATH, got: {item}")
        dataset, path = item.split(":", 1)
        inputs.append((dataset, Path(path)))
    return inputs


def write_manifest(path: Path, rows: List[Dict]) -> None:
    columns = [
        "dataset",
        "protein_id",
        "embedding_scope",
        "domain_source",
        "domain_type",
        "domain_start",
        "domain_end",
        "domain_region_start",
        "domain_region_end",
        "domain_region_count",
        "source_fasta",
        "sequence_length",
        "embedding_sequence_length",
        "embedding_dim",
        "embedding_path",
        "per_token_path",
        "status",
        "note",
    ]
    with path.open("w") as out:
        out.write("\t".join(columns) + "\n")
        for row in rows:
            out.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def consolidate(output_dir: Path, manifest_rows: List[Dict]) -> None:
    ok_rows = [row for row in manifest_rows if row.get("status") == "ok"]
    for key in sorted({(row["dataset"], row["embedding_scope"]) for row in ok_rows}):
        dataset, scope = key
        rows = [row for row in ok_rows if row["dataset"] == dataset and row["embedding_scope"] == scope]
        embeddings = [np.load(row["embedding_path"]).astype(np.float32) for row in rows]
        np.savez_compressed(
            output_dir / f"{dataset}_{scope}_esmc_embeddings.npz",
            protein_ids=np.array([row["protein_id"] for row in rows], dtype=object),
            embedding_scopes=np.array([row["embedding_scope"] for row in rows], dtype=object),
            domain_sources=np.array([row.get("domain_source", "") for row in rows], dtype=object),
            domain_types=np.array([row.get("domain_type", "") for row in rows], dtype=object),
            domain_starts=np.array([row.get("domain_start", "") for row in rows], dtype=object),
            domain_ends=np.array([row.get("domain_end", "") for row in rows], dtype=object),
            embeddings=np.stack(embeddings, axis=0),
        )
    if ok_rows:
        embeddings = [np.load(row["embedding_path"]).astype(np.float32) for row in ok_rows]
        np.savez_compressed(
            output_dir / "combined_esmc_embeddings.npz",
            protein_ids=np.array([row["protein_id"] for row in ok_rows], dtype=object),
            datasets=np.array([row["dataset"] for row in ok_rows], dtype=object),
            embedding_scopes=np.array([row["embedding_scope"] for row in ok_rows], dtype=object),
            domain_sources=np.array([row.get("domain_source", "") for row in ok_rows], dtype=object),
            domain_types=np.array([row.get("domain_type", "") for row in ok_rows], dtype=object),
            domain_starts=np.array([row.get("domain_start", "") for row in ok_rows], dtype=object),
            domain_ends=np.array([row.get("domain_end", "") for row in ok_rows], dtype=object),
            embeddings=np.stack(embeddings, axis=0),
        )


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_embedding_dir = output_dir / "per_embedding"
    per_token_dir = output_dir / "per_token_hidden"
    per_embedding_dir.mkdir(parents=True, exist_ok=True)
    if args.save_per_token:
        per_token_dir.mkdir(parents=True, exist_ok=True)

    selected_domains = None if args.domain_types == "all" else set(x.strip() for x in args.domain_types.split(",") if x.strip())
    inputs = build_inputs(args)
    run_config = vars(args).copy()
    run_config["resolved_model_dir"] = str(model_dir)
    run_config["inputs"] = [[dataset, str(path)] for dataset, path in inputs]
    (output_dir / "esmc_embedding_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n")

    raw_records_by_dataset: Dict[str, List[Tuple[str, str, str, str, str]]] = {}
    for dataset, fasta_path in inputs:
        if not fasta_path.exists():
            raise FileNotFoundError(f"Missing FASTA for {dataset}: {fasta_path}")
        rows = []
        for protein_id, seq in read_fasta(fasta_path):
            cleaned, clean_note = clean_sequence(seq)
            rows.append((protein_id, seq, cleaned, clean_note, str(fasta_path)))
        raw_records_by_dataset[dataset] = rows

    domain_hits_by_dataset: Dict[str, Dict[str, List[Dict]]] = {}
    if args.embedding_scope in {"domain", "both"}:
        for dataset, rows in raw_records_by_dataset.items():
            scan_input = [(protein_id, cleaned) for protein_id, _, cleaned, _, _ in rows]
            domain_hits_by_dataset[dataset] = scan_domains(scan_input, Path(args.hmm_path), Path(args.rbp_trace_path))

    jobs: List[Dict] = []
    manifest_rows: List[Dict] = []
    for dataset, rows in raw_records_by_dataset.items():
        for protein_id, original_seq, cleaned_seq, clean_note, source_fasta in rows:
            scopes = []
            if args.embedding_scope in {"full", "both"}:
                scopes.append(("full_length", cleaned_seq, {}))
            if args.embedding_scope in {"domain", "both"}:
                merged = merge_domain_regions(
                    cleaned_seq,
                    domain_hits_by_dataset.get(dataset, {}).get(protein_id, []),
                    args.domain_flank,
                    selected_domains,
                )
                if merged is None:
                    row = {
                        "dataset": dataset,
                        "protein_id": protein_id,
                        "embedding_scope": "domain_merged",
                        "source_fasta": source_fasta,
                        "sequence_length": len(original_seq),
                        "embedding_sequence_length": 0,
                        "status": "no_domain",
                        "note": clean_note,
                    }
                    manifest_rows.append(row)
                else:
                    meta = {k: v for k, v in merged.items() if k != "sequence"}
                    scopes.append(("domain_merged", merged["sequence"], meta))
            for scope, emb_seq, meta in scopes:
                stem = safe_name(dataset, scope, protein_id)
                emb_path = per_embedding_dir / dataset / scope / f"{stem}.npy"
                token_path = per_token_dir / dataset / scope / f"{stem}.pt" if args.save_per_token else None
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                if token_path is not None:
                    token_path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "dataset": dataset,
                    "protein_id": protein_id,
                    "embedding_scope": scope,
                    "source_fasta": source_fasta,
                    "sequence_length": len(original_seq),
                    "embedding_sequence_length": len(emb_seq),
                    "embedding_path": str(emb_path),
                    "per_token_path": str(token_path) if token_path is not None else "",
                    "status": "pending",
                    "note": clean_note,
                    **meta,
                }
                if emb_path.exists() and not args.overwrite:
                    emb = np.load(emb_path, mmap_mode="r")
                    row["embedding_dim"] = int(emb.shape[-1])
                    row["status"] = "ok"
                    manifest_rows.append(row)
                else:
                    jobs.append({"row": row, "sequence": emb_seq})

    if args.dry_run:
        manifest_rows.extend(job["row"] for job in jobs)
        write_manifest(output_dir / "esmc_embedding_manifest.dry_run.tsv", manifest_rows)
        with (output_dir / "esmc_embedding_input_summary.tsv").open("w") as out:
            out.write("dataset\tembedding_scope\tn_records\tn_pending_or_existing\tn_no_domain\n")
            for dataset in sorted(raw_records_by_dataset):
                for scope in ["full_length", "domain_merged"]:
                    rows = [r for r in manifest_rows if r["dataset"] == dataset and r["embedding_scope"] == scope]
                    if rows:
                        out.write(f"{dataset}\t{scope}\t{len(raw_records_by_dataset[dataset])}\t{len(rows)}\t{sum(r['status']=='no_domain' for r in rows)}\n")
        print(f"dry_run_ok\tjobs={len(jobs)}\tmanifest={output_dir / 'esmc_embedding_manifest.dry_run.tsv'}")
        return

    if not jobs:
        write_manifest(output_dir / "esmc_embedding_manifest.tsv", manifest_rows)
        consolidate(output_dir, manifest_rows)
        with (output_dir / "esmc_embedding_summary.tsv").open("w") as out:
            out.write("dataset\tembedding_scope\tn_rows\tn_ok\tn_failed\tn_no_domain\tembedding_dim\n")
            keys = sorted({(r["dataset"], r["embedding_scope"]) for r in manifest_rows})
            for dataset, scope in keys:
                rows = [r for r in manifest_rows if r["dataset"] == dataset and r["embedding_scope"] == scope]
                ok = [r for r in rows if r["status"] == "ok"]
                out.write(
                    f"{dataset}\t{scope}\t{len(rows)}\t{len(ok)}\t"
                    f"{sum(r['status']=='failed' for r in rows)}\t{sum(r['status']=='no_domain' for r in rows)}\t"
                    f"{ok[0].get('embedding_dim','') if ok else ''}\n"
                )
        print(f"no_pending_jobs\tmanifest={output_dir / 'esmc_embedding_manifest.tsv'}")
        return

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    tokenizer, model = load_model(model_dir, args.device, args.dtype)

    for job in jobs:
        row = job["row"]
        seq = job["sequence"]
        try:
            emb, note, per_token_hidden = embed_sequence(
                tokenizer,
                model,
                seq,
                device,
                args.max_aa_per_window,
                args.window_overlap,
                args.long_sequence_policy,
                args.batch_size,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and args.device == "cuda":
                torch.cuda.empty_cache()
            try:
                emb, note, per_token_hidden = embed_sequence(
                    tokenizer,
                    model,
                    seq,
                    device,
                    args.max_aa_per_window,
                    args.window_overlap,
                    args.long_sequence_policy,
                    1,
                )
                note = f"single_retry_after_batch_failure;{note}"
            except Exception as exc_single:
                row["status"] = "failed"
                row["note"] = f"{row.get('note','')};{type(exc_single).__name__}:{str(exc_single).replace(chr(9),' ')[:300]}"
                manifest_rows.append(row)
                write_manifest(output_dir / "esmc_embedding_manifest.tsv", manifest_rows)
                continue
        except Exception as exc:
            row["status"] = "failed"
            row["note"] = f"{row.get('note','')};{type(exc).__name__}:{str(exc).replace(chr(9),' ')[:300]}"
            manifest_rows.append(row)
            write_manifest(output_dir / "esmc_embedding_manifest.tsv", manifest_rows)
            continue

        if emb is None:
            row["status"] = "failed"
            row["note"] = f"{row.get('note','')};{note}"
        else:
            np.save(row["embedding_path"], emb.astype(np.float32))
            if args.save_per_token and per_token_hidden is not None and row.get("per_token_path"):
                torch.save(per_token_hidden, row["per_token_path"])
            row["embedding_dim"] = int(emb.shape[-1])
            row["status"] = "ok"
            row["note"] = f"{row.get('note','')};{note}" if row.get("note") else note
        manifest_rows.append(row)
        write_manifest(output_dir / "esmc_embedding_manifest.tsv", manifest_rows)

    write_manifest(output_dir / "esmc_embedding_manifest.tsv", manifest_rows)
    consolidate(output_dir, manifest_rows)
    failed_total = sum(row.get("status") == "failed" for row in manifest_rows)
    with (output_dir / "esmc_embedding_summary.tsv").open("w") as out:
        out.write("dataset\tembedding_scope\tn_rows\tn_ok\tn_failed\tn_no_domain\tembedding_dim\n")
        keys = sorted({(r["dataset"], r["embedding_scope"]) for r in manifest_rows})
        for dataset, scope in keys:
            rows = [r for r in manifest_rows if r["dataset"] == dataset and r["embedding_scope"] == scope]
            ok = [r for r in rows if r["status"] == "ok"]
            out.write(
                f"{dataset}\t{scope}\t{len(rows)}\t{len(ok)}\t"
                f"{sum(r['status']=='failed' for r in rows)}\t{sum(r['status']=='no_domain' for r in rows)}\t"
                f"{ok[0].get('embedding_dim','') if ok else ''}\n"
            )
    if failed_total and not args.allow_failed:
        raise SystemExit(f"embedding_failed_count={failed_total}; see {output_dir / 'esmc_embedding_manifest.tsv'}")


if __name__ == "__main__":
    main()
