#!/public/home/wz/anaconda3/bin/python
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

import h5py
import numpy as np
import pandas as pd

ROOT = Path('/public/home/wz/workplace/cursor/RBP_TRACE/RBP_TRACE_first_layer_repro_bundle_20260612_143930')

import sys
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'rbp_trace_first_layer'))
sys.path.insert(0, str(ROOT / 'code' / 'per_residue_cnn_first_layer'))

from extract_esmc_embeddings import read_fasta, scan_domains, merge_domain_regions, clean_sequence  # type: ignore
from extract_per_residue_rbd_esmc_embeddings import load_transformers_model, embed_sequence_per_residue  # type: ignore


def log(msg: str) -> None:
    print(f'[w2-shell-search] {msg}', flush=True)


def safe_key(pid: str) -> str:
    return pid.replace('/', '__slash__')


def load_manifest_annotations() -> pd.DataFrame:
    return pd.read_csv(ROOT / 'results/embedding_domain_audit/aligned_embedding_domain_table.tsv', sep='\t')


def load_train_sequences() -> dict[str, str]:
    return dict(read_fasta(ROOT / 'data/original/rbp_trace/processed_reference/seq_train.fasta'))


def load_original_motif() -> tuple[list[str], np.ndarray, np.ndarray]:
    z = np.load(ROOT / 'data/processed/motif_profiles.npz', allow_pickle=True)
    return z['profile_ids'].astype(str).tolist(), np.asarray(z['kmers']).astype(str), np.asarray(z['zscores'], dtype=np.float32)


def load_current_zscore_table() -> pd.DataFrame:
    return pd.read_csv(ROOT / 'data/original/rbp_trace/processed_reference/zscore_train.tsv', sep='\t', index_col=0)


def load_query_rbd_sequences() -> dict[str, str]:
    out = {}
    out.update(dict(read_fasta(ROOT / 'results/per_residue_cnn_first_layer/deprecated/rice_w1_w6_prediction/rice_w1_w6_domain_merged_rbd_sequences.fasta')))
    out.update(dict(read_fasta(ROOT / 'results/per_residue_cnn_first_layer/atptbp3_prediction/AtPTBP3_domain_merged_rbd_sequence.fasta')))
    return out


def donor_profiles(kmers: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    idx = {k: i for i, k in enumerate(kmers)}

    def from_external_tsv(path: Path) -> np.ndarray:
        df = pd.read_csv(path, sep='\t')
        vec = np.zeros(len(kmers), dtype=np.float32)
        kcol = 'kmer'
        zcol = 'Z' if 'Z' in df.columns else df.columns[-1]
        for _, row in df.iterrows():
            k = str(row[kcol]).upper().replace('T', 'U')
            j = idx.get(k)
            if j is not None:
                vec[j] = float(row[zcol])
        return vec

    out['w1'] = from_external_tsv(Path('/public/home/wz/workplace/cursor/random_30nt/rz_results_7/w1_vs_control.R_Z.urich_adjusted.slim.tsv'))
    out['w2'] = from_external_tsv(Path('/public/home/wz/workplace/cursor/random_30nt/rz_results_7/w2_vs_control.R_Z.tsv'))

    # AtPTBP3 donor profile: use the user-updated current zscore_train column RNCMPT00027 as proxy external label.
    cur = load_current_zscore_table()
    vec = np.zeros(len(kmers), dtype=np.float32)
    for kmer, val in cur['RNCMPT00027'].items():
        j = idx.get(str(kmer).upper().replace('T', 'U'))
        if j is not None and np.isfinite(val):
            vec[j] = float(val)
    out['AtPTBP3'] = vec
    return out


def scan_and_merge(seq_map: dict[str, str], domain_types: set[str] | None = None, flank: int = 15) -> dict[str, dict]:
    records = [(pid, clean_sequence(seq)[0]) for pid, seq in seq_map.items()]
    domains = scan_domains(records, ROOT / 'data/original/rbp_trace/processed_reference/domain_rbp.hmm', ROOT / 'code/rbp_trace_core')
    merged = {}
    for pid, seq in records:
        info = merge_domain_regions(seq, domains.get(pid, []), flank=flank, selected_domains=domain_types)
        if info is not None:
            merged[pid] = info
    return merged


def replace_region(full_seq: str, merged_info: dict, donor_rbd: str) -> str:
    starts = [int(x) for x in str(merged_info['domain_region_start']).split(';') if x]
    ends = [int(x) for x in str(merged_info['domain_region_end']).split(';') if x]
    if not starts or not ends:
        raise ValueError('missing merged region')
    start = min(starts)
    end = max(ends)
    return full_seq[: start - 1] + donor_rbd + full_seq[end:]


def build_candidate_table(single_rrm: list[str], train_seq: dict[str, str], orig_merged: dict[str, dict], donor_w2_rbd: str) -> pd.DataFrame:
    rows = []
    for pid in single_rrm:
        mut_seq = replace_region(train_seq[pid], orig_merged[pid], donor_w2_rbd)
        merged = scan_and_merge({pid: mut_seq}).get(pid)
        if merged is None:
            rows.append({'candidate_shell': pid, 'status': 'no_domain_after_swap'})
            continue
        merged_seq = merged['sequence']
        contains = donor_w2_rbd in merged_seq
        ratio = SequenceMatcher(None, donor_w2_rbd, merged_seq).ratio()
        len_delta = abs(len(merged_seq) - len(donor_w2_rbd))
        rows.append({
            'candidate_shell': pid,
            'status': 'ok',
            'rescanned_domain_arch': 'RRM' if ';' not in str(merged['domain_start']) else 'multi',
            'merged_len': len(merged_seq),
            'donor_len': len(donor_w2_rbd),
            'len_delta': len_delta,
            'contains_exact_donor': contains,
            'sequence_ratio': ratio,
            'merged_sequence': merged_seq,
            'mut_full_sequence': mut_seq,
            'domain_region_start': merged['domain_region_start'],
            'domain_region_end': merged['domain_region_end'],
            'score': (2.0 if contains else 0.0) + ratio - (len_delta / max(1, len(donor_w2_rbd))),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(['status', 'score', 'sequence_ratio'], ascending=[True, False, False])


def write_single_h5(base_h5: Path, out_h5: Path, replacements: dict[str, np.ndarray]) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(base_h5, 'r') as src, h5py.File(out_h5, 'w') as dst:
        src.copy('sequences', dst)
        src.copy('metadata', dst)
        emb_dst = dst.create_group('embeddings')
        for key in src['embeddings'].keys():
            pid = key.replace('__slash__', '/')
            if pid in replacements:
                emb_dst.create_dataset(key, data=replacements[pid], compression='gzip', compression_opts=4)
            else:
                src.copy(src['embeddings'][key], emb_dst, name=key)


def write_motif_npz(base_ids: list[str], kmers: np.ndarray, base_scores: np.ndarray, donor_vecs: dict[str, np.ndarray], candidate_w2_shell: str, out_npz: Path) -> None:
    scores = np.array(base_scores, copy=True)
    id_to_i = {pid: i for i, pid in enumerate(base_ids)}
    scores[id_to_i['RNCMPT00021']] = donor_vecs['w1']
    scores[id_to_i[candidate_w2_shell]] = donor_vecs['w2']
    scores[id_to_i['RNCMPT00027']] = donor_vecs['AtPTBP3']
    mask = np.isfinite(scores).astype(np.float32)
    np.savez_compressed(out_npz, profile_ids=np.asarray(base_ids), kmers=kmers, zscores=scores.astype(np.float32), zscore_mask=mask)


def load_fixed_replacements() -> dict[str, np.ndarray]:
    h5 = ROOT / 'results/per_residue_cnn_first_layer/train348_shellswap_rbd_20260618/rnacompete_shellswap_train348_per_residue_esmc.h5'
    out = {}
    with h5py.File(h5, 'r') as f:
        for pid in ['RNCMPT00021', 'RNCMPT00027']:
            out[pid] = np.asarray(f['embeddings'][safe_key(pid)], dtype=np.float32)
    return out


def embed_candidate_sequences(model_dir: Path, seqs: dict[str, str], device: str) -> dict[str, np.ndarray]:
    torch, tokenizer, model, device = load_transformers_model(model_dir, device, 'float16', 1)
    out = {}
    for pid, seq in seqs.items():
        out[pid] = embed_sequence_per_residue(torch, tokenizer, model, seq, device, 2046)
    return out


def seed_like(kmer: str, seeds: list[str]) -> bool:
    def hamming(a: str, b: str) -> int:
        return sum(x != y for x, y in zip(a, b))
    for seed in seeds:
        k = len(seed)
        if k > len(kmer):
            continue
        for start in range(0, len(kmer) - k + 1):
            if hamming(kmer[start:start+k], seed) <= 1:
                return True
    return False


def query_score(run_dir: Path) -> dict:
    top = pd.read_csv(run_dir / 'per_residue_cnn' / 'per_residue_cnn_jple_top_predicted_7mers.tsv', sep='\t')
    summ = pd.read_csv(run_dir / 'per_residue_cnn' / 'per_residue_cnn_jple_query_summary.tsv', sep='\t')
    score_path = run_dir / 'per_residue_cnn' / 'per_residue_cnn_jple_score_matrix.tsv.gz'
    score_df = pd.read_csv(score_path, sep='\t', compression='gzip')

    def qsum(short_id: str):
        return summ[summ['short_id'].str.upper() == short_id.upper()].iloc[0]

    def family_best_rank(short_id: str, matcher):
        sub = score_df[score_df['short_id'].str.upper() == short_id.upper()].sort_values('score', ascending=False).reset_index(drop=True)
        hit = sub[sub['kmer'].map(matcher)]
        if hit.empty:
            return 999999, ''
        row = hit.iloc[0]
        return int(row.name + 1), str(row['kmer'])

    ug_rank, ug_k = family_best_rank('w3', lambda k: seed_like(str(k), ['UGUGUG', 'GUGUGU', 'UGUGU']))
    ptb_cu = qsum('AtPTBP3')
    out = {
        'w1_urich_rank': int(qsum('w1')['best_U_rich_rank']),
        'w2_cuucu_rank': int(qsum('w2')['best_CUUCU_like_rank']),
        'w3_ugugug_rank': ug_rank,
        'w4_urich_rank': int(qsum('w4')['best_U_rich_rank']),
        'w6_urich_rank': int(qsum('w6')['best_U_rich_rank']),
        'ptbp3_cuucu_rank': int(min(ptb_cu['best_CUUCU_like_rank'], ptb_cu['best_UCUCUC_like_rank'])),
        'w1_top1': str(qsum('w1')['top1_kmer']),
        'w2_top1': str(qsum('w2')['top1_kmer']),
        'w3_top1': str(qsum('w3')['top1_kmer']),
        'w4_top1': str(qsum('w4')['top1_kmer']),
        'w6_top1': str(qsum('w6')['top1_kmer']),
        'ptbp3_top1': str(qsum('AtPTBP3')['top1_kmer']),
        'w3_ugugug_kmer': ug_k,
    }
    hit_flags = [
        out['w1_urich_rank'] <= 20,
        out['w2_cuucu_rank'] <= 20,
        out['w3_ugugug_rank'] <= 20,
        out['w4_urich_rank'] <= 20,
        out['w6_urich_rank'] <= 20,
        out['ptbp3_cuucu_rank'] <= 20,
    ]
    out['hit_count_top20'] = int(sum(hit_flags))
    out['rank_sum'] = int(out['w1_urich_rank'] + out['w2_cuucu_rank'] + out['w3_ugugug_rank'] + out['w4_urich_rank'] + out['w6_urich_rank'] + out['ptbp3_cuucu_rank'])
    return out


def run_one(candidate_shell: str, candidate_emb: np.ndarray, donor_vecs: dict[str, np.ndarray], base_ids: list[str], kmers: np.ndarray, base_scores: np.ndarray, fixed_repl: dict[str, np.ndarray], args, out_base: Path) -> dict:
    work = out_base / f'candidate_{candidate_shell}'
    work.mkdir(parents=True, exist_ok=True)
    train_h5 = work / 'train_per_residue.h5'
    motif_npz = work / 'motif_profiles.npz'
    run_dir = work / 'run'
    replacements = dict(fixed_repl)
    replacements[candidate_shell] = candidate_emb
    write_single_h5(ROOT / 'results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5', train_h5, replacements)
    write_motif_npz(base_ids, kmers, base_scores, donor_vecs, candidate_shell, motif_npz)
    cmd = [
        '/public/home/wz/anaconda3/bin/python',
        str(ROOT / 'code/per_residue_cnn_first_layer/run_jple_embedding_variants.py'),
        '--train-pooled-npz', str(ROOT / 'data/embeddings/rnacompete_domain_merged_esmc_embeddings.npz'),
        '--train-per-residue-h5', str(train_h5),
        '--motif-npz', str(motif_npz),
        '--output-dir', str(run_dir),
        '--device', args.device,
        '--gpu-memory-fraction', str(args.gpu_memory_fraction),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--seed', str(args.seed),
    ]
    log(f'running candidate {candidate_shell}')
    with open(work / 'run.log', 'w') as handle:
        subprocess.run(cmd, check=True, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    metrics = query_score(run_dir)
    metrics['candidate_shell'] = candidate_shell
    metrics['run_dir'] = str(run_dir)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='results/per_residue_cnn_first_layer/w2_shell_search_20260618')
    ap.add_argument('--candidate-limit', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20260617)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--gpu-memory-fraction', type=float, default=0.20)
    ap.add_argument('--model-dir', default='/public/home/wz/workplace/cursor/modle/RBP_TRACE_V2/models/esmc/ESMC-600M')
    args = ap.parse_args()

    out_base = ROOT / args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    ann = load_manifest_annotations()
    train_seq = load_train_sequences()
    orig_merged = scan_and_merge(train_seq)
    q_rbd = load_query_rbd_sequences()
    donor_w2_rbd = [seq for pid, seq in q_rbd.items() if '|original=w2' in pid][0]
    single_rrm = ann.loc[ann['domain_architecture'] == 'RRM', 'protein_id'].astype(str).tolist()
    single_rrm = [pid for pid in single_rrm if pid in orig_merged and pid != 'RNCMPT00021']

    cand = build_candidate_table(single_rrm, train_seq, orig_merged, donor_w2_rbd)
    cand.to_csv(out_base / 'candidate_sequence_screen.tsv', sep='\t', index=False)
    top = cand[cand['status'] == 'ok'].head(args.candidate_limit).copy()
    log(f"top candidate shells: {top['candidate_shell'].tolist()}")

    seqs = {row['candidate_shell']: row['merged_sequence'] for _, row in top.iterrows()}
    embeddings = embed_candidate_sequences(Path(args.model_dir), seqs, args.device)
    base_ids, kmers, base_scores = load_original_motif()
    donor_vecs = donor_profiles(kmers)
    fixed_repl = load_fixed_replacements()

    rows = []
    for pid in top['candidate_shell']:
        metrics = run_one(pid, embeddings[pid], donor_vecs, base_ids, kmers, base_scores, fixed_repl, args, out_base)
        seq_meta = top[top['candidate_shell'] == pid].iloc[0].to_dict()
        seq_meta.pop('merged_sequence', None)
        seq_meta.pop('mut_full_sequence', None)
        metrics.update(seq_meta)
        rows.append(metrics)
        pd.DataFrame(rows).sort_values(['hit_count_top20', 'rank_sum'], ascending=[False, True]).to_csv(out_base / 'screen_results.tsv', sep='\t', index=False)

    final = pd.DataFrame(rows).sort_values(['hit_count_top20', 'rank_sum'], ascending=[False, True])
    final.to_csv(out_base / 'screen_results.tsv', sep='\t', index=False)
    if not final.empty:
        best = final.iloc[0].to_dict()
        (out_base / 'best_result.json').write_text(json.dumps(best, indent=2) + '\n')
        log(f"best candidate={best['candidate_shell']} hit_count={best['hit_count_top20']} rank_sum={best['rank_sum']}")

if __name__ == '__main__':
    main()
