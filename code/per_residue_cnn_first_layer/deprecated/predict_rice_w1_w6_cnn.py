#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_cnn_motif_latent_plant_loo import (  # noqa: E402
    PerResidueCnn,
    RbdEmbeddingDataset,
    collate_batch,
    eval_model_on_ids,
    first_key,
    load_h5_features,
    metric_profile,
    resolve_path,
    setup_threads,
)


def log(msg: str) -> None:
    print('[cnn-rice-predict] ' + msg, flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_motif(path: Path):
    z = np.load(path, allow_pickle=True)
    ids = np.asarray(z[first_key(z, ['profile_ids', 'protein_ids', 'ids', 'names'])]).astype(str)
    y = np.asarray(z[first_key(z, ['zscores', 'scores', 'Y', 'profiles'])], dtype=np.float32)
    kmers = np.asarray(z['kmers']).astype(str) if 'kmers' in z.files else np.arange(y.shape[1]).astype(str)
    return ids, y, kmers, {pid: i for i, pid in enumerate(ids)}


def split_train_val(ids: list[str], seed: int, val_fraction: float):
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_val = max(8, int(round(len(ids) * val_fraction)))
    return ids[n_val:], ids[:n_val]


def main() -> None:
    ap = argparse.ArgumentParser(description='Train final exploratory per-residue CNN and predict W1-W6 rice RBPs.')
    ap.add_argument('--train-h5', default='results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5')
    ap.add_argument('--train-manifest', default='results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv')
    ap.add_argument('--rice-h5', default='results/per_residue_cnn_first_layer/rice_w1_w6_prediction/rice_w1_w6_per_residue_esmc.h5')
    ap.add_argument('--rice-manifest', default='results/per_residue_cnn_first_layer/rice_w1_w6_prediction/rice_w1_w6_per_residue_manifest.tsv')
    ap.add_argument('--rice-annotation-tsv', default='results/final_rice_prediction/rice_inputs/rice_w1_w6_rbd_domain_annotation.tsv')
    ap.add_argument('--motif-npz', default='data/processed/motif_profiles.npz')
    ap.add_argument('--output-dir', default='results/per_residue_cnn_first_layer/rice_w1_w6_prediction')
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'auto'])
    ap.add_argument('--gpu-memory-fraction', type=float, default=0.20)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--gradient-clip', type=float, default=1.0)
    ap.add_argument('--latent-dim', type=int, default=50)
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--kernel-size', type=int, default=5)
    ap.add_argument('--num-blocks', type=int, default=2)
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--top-n', type=int, default=20)
    ap.add_argument('--torch-num-threads', type=int, default=1)
    ap.add_argument('--seed', type=int, default=20260615)
    args = ap.parse_args()

    setup_threads(args.torch_num_threads)
    seed_all(args.seed)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        log('CUDA requested but unavailable; falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    if device.type == 'cuda' and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_memory_fraction), device=0)
        log(f'Set CUDA memory fraction to {args.gpu_memory_fraction}')

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_h5 = resolve_path(args.train_h5, ['rnacompete_rbd_per_residue_esmc.h5'])
    train_manifest = resolve_path(args.train_manifest, ['rnacompete_rbd_per_residue_manifest.tsv'], required=False)
    rice_h5 = resolve_path(args.rice_h5, ['rice_w1_w6_per_residue_esmc.h5'])
    rice_manifest = resolve_path(args.rice_manifest, ['rice_w1_w6_per_residue_manifest.tsv'], required=False)
    motif_path = resolve_path(args.motif_npz, ['motif_profiles.npz'])

    train_x, train_len = load_h5_features(train_h5, train_manifest)
    rice_x, rice_len = load_h5_features(rice_h5, rice_manifest)
    motif_ids, y, kmers, id_to_y = load_motif(motif_path)
    train_ids = [pid for pid in motif_ids if pid in train_x]
    if len(train_ids) < 20:
        raise SystemExit('Too few aligned RNAcompete training IDs')
    log(f'train proteins={len(train_ids)} rice proteins={len(rice_x)}')

    train_inner, val_ids = split_train_val(train_ids, args.seed, args.val_fraction)
    latent_dim = max(2, min(args.latent_dim, len(train_ids) - 1, y.shape[1] - 1))
    svd = TruncatedSVD(n_components=latent_dim, random_state=args.seed)
    svd.fit(y[[id_to_y[p] for p in train_ids]])
    train_lat = svd.transform(y[[id_to_y[p] for p in train_inner]]).astype(np.float32)
    latent_mean = train_lat.mean(axis=0, keepdims=True).astype(np.float32)
    latent_std = train_lat.std(axis=0, keepdims=True).astype(np.float32)
    latent_std[latent_std < 1e-6] = 1.0

    z_y = np.zeros((len(id_to_y), latent_dim), dtype=np.float32)
    for pid in train_ids:
        lat = svd.transform(y[[id_to_y[pid]]]).astype(np.float32)
        z_y[id_to_y[pid]] = ((lat - latent_mean) / latent_std)[0]

    train_loader = DataLoader(RbdEmbeddingDataset(train_inner, train_x, z_y, id_to_y), batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(RbdEmbeddingDataset(val_ids, train_x, z_y, id_to_y), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    input_dim = next(iter(train_x.values())).shape[1]
    model = PerResidueCnn(input_dim, args.hidden_dim, latent_dim, args.kernel_size, args.num_blocks, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    best_state = None
    best_val = -1e9
    best_epoch = 0
    patience_left = args.patience
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _, x, mask, target in train_loader:
            x = x.to(device); mask = mask.to(device); target = target.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = mse(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        _, val_loss, val_metric = eval_model_on_ids(model, val_loader, device, svd, latent_mean, latent_std, y, id_to_y)
        score = val_metric if np.isfinite(val_metric) else -val_loss
        curve.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_loss': val_loss, 'val_pearson': val_metric})
        if score > best_val:
            best_val = float(score); best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch % 10 == 0:
            log(f'epoch={epoch} train_loss={np.mean(losses):.4g} val_pearson={val_metric:.4g} best_epoch={best_epoch}')
        if patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    rice_ids = list(rice_x.keys())
    dummy_id_to_y = {pid: i for i, pid in enumerate(rice_ids)}
    dummy_y = np.zeros((len(rice_ids), latent_dim), dtype=np.float32)
    rice_loader = DataLoader(RbdEmbeddingDataset(rice_ids, rice_x, dummy_y, dummy_id_to_y), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)
    pred_rows = []
    matrix = []
    with torch.no_grad():
        for ids, x, mask, _ in rice_loader:
            x = x.to(device); mask = mask.to(device)
            pred_lat_z = model(x, mask).detach().cpu().numpy()
            pred_lat = pred_lat_z * latent_std + latent_mean
            pred_y = svd.inverse_transform(pred_lat)[0]
            matrix.append(pred_y)
            pid = ids[0]
            top_idx = np.argsort(-pred_y)[:args.top_n]
            for rank, idx in enumerate(top_idx, start=1):
                pred_rows.append({'rice_rbp_id': pid, 'short_id': pid.split('|')[0].upper(), 'rank': rank, 'kmer': kmers[idx], 'score': float(pred_y[idx]), 'rbd_length': rice_len.get(pid, np.nan)})
    top = pd.DataFrame(pred_rows)
    top.to_csv(out_dir / 'rice_w1_w6_cnn_top_predicted_7mers.tsv', sep='\t', index=False)
    with gzip.open(out_dir / 'rice_w1_w6_cnn_predicted_7mer_score_matrix.tsv.gz', 'wt') as handle:
        handle.write('rice_rbp_id\t' + '\t'.join(kmers) + '\n')
        for pid, row in zip(rice_ids, matrix):
            handle.write(pid + '\t' + '\t'.join(f'{float(v):.6g}' for v in row) + '\n')
    pd.DataFrame(curve).to_csv(out_dir / 'rice_w1_w6_cnn_training_curve.tsv', sep='\t', index=False)
    summary = {'train_proteins': len(train_ids), 'rice_proteins': len(rice_ids), 'latent_dim': latent_dim, 'best_epoch': best_epoch, 'best_val_pearson': best_val, 'model_note': 'exploratory_per_residue_cnn_not_final_recommended_model'}
    (out_dir / 'rice_w1_w6_cnn_prediction_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    torch.save({'model_state': model.state_dict(), 'latent_dim': latent_dim, 'best_epoch': best_epoch}, out_dir / 'rice_w1_w6_cnn_final_model.pt')
    log('Top motifs: ' + str(out_dir / 'rice_w1_w6_cnn_top_predicted_7mers.tsv'))
    log('Summary: ' + str(out_dir / 'rice_w1_w6_cnn_prediction_summary.json'))


if __name__ == '__main__':
    main()
