#!/usr/bin/env python3
"""Final exploratory per-residue RBD CNN query predictor.

This is the retained CNN logic: train on all aligned RNAcompete RBD
per-residue embeddings, fit motif SVD on the training set, standardize motif
targets, and optionally apply a weak poly-U ranking regularizer. It is for
exploratory query checks; the official first-layer model remains the original
RBPTraceFirstLayer when CNN validation is not superior.
"""

from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['NUMEXPR_NUM_THREADS']='1'; os.environ['VECLIB_MAXIMUM_THREADS']='1'
import argparse, gzip, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR))
from cnn_model_utils import PerResidueCnn, RbdEmbeddingDataset, collate_batch, first_key, load_h5_features, resolve_path, setup_threads

def log(x): print('[final-cnn-query] '+x, flush=True)
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def load_motif(path):
    z=np.load(path, allow_pickle=True); ids=np.asarray(z[first_key(z,['profile_ids','protein_ids','ids','names'])]).astype(str)
    y=np.asarray(z[first_key(z,['zscores','scores','Y','profiles'])], dtype=np.float32); km=np.asarray(z['kmers']).astype(str)
    return ids,y,km,{p:i for i,p in enumerate(ids)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-h5',default='results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5')
    ap.add_argument('--train-manifest',default='results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv')
    ap.add_argument('--query-h5',required=True)
    ap.add_argument('--query-manifest',default=None)
    ap.add_argument('--motif-npz',default='data/processed/motif_profiles.npz')
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--gpu-memory-fraction',type=float,default=0.20)
    ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--latent-dim',type=int,default=50); ap.add_argument('--hidden-dim',type=int,default=64)
    ap.add_argument('--dropout',type=float,default=0.3); ap.add_argument('--kernel-size',type=int,default=5); ap.add_argument('--num-blocks',type=int,default=2)
    ap.add_argument('--learning-rate',type=float,default=1e-4); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--gradient-clip',type=float,default=1.0)
    ap.add_argument('--ranking-loss-weight',type=float,default=0.03); ap.add_argument('--ranking-top-k',type=int,default=20); ap.add_argument('--ranking-margin',type=float,default=0.5)
    ap.add_argument('--top-n',type=int,default=50); ap.add_argument('--torch-num-threads',type=int,default=1); ap.add_argument('--seed',type=int,default=20260615)
    args=ap.parse_args(); setup_threads(args.torch_num_threads); seed_all(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available(): args.device='cpu'
    device=torch.device(args.device)
    if device.type=='cuda' and args.gpu_memory_fraction>0: torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,0)
    out=Path(args.output_dir); out=ROOT/out if not out.is_absolute() else out; out.mkdir(parents=True,exist_ok=True)
    train_x,_=load_h5_features(resolve_path(args.train_h5,['rnacompete_rbd_per_residue_esmc.h5']), resolve_path(args.train_manifest,['rnacompete_rbd_per_residue_manifest.tsv'],required=False))
    query_x,query_len=load_h5_features(resolve_path(args.query_h5,[Path(args.query_h5).name]), resolve_path(args.query_manifest,[Path(args.query_manifest).name],required=False) if args.query_manifest else None)
    motif_ids,y,kmers,id2y=load_motif(resolve_path(args.motif_npz,['motif_profiles.npz']))
    train_ids=[p for p in motif_ids if p in train_x]
    y_train=y[[id2y[p] for p in train_ids]]
    scaler=StandardScaler(with_mean=True, with_std=True)
    y_scaled=scaler.fit_transform(y_train).astype(np.float32)
    latent_dim=max(2,min(args.latent_dim,len(train_ids)-1,y.shape[1]-1))
    svd=TruncatedSVD(n_components=latent_dim, random_state=args.seed)
    lat=svd.fit_transform(y_scaled).astype(np.float32)
    lat_mean=lat.mean(axis=0,keepdims=True).astype(np.float32); lat_std=lat.std(axis=0,keepdims=True).astype(np.float32); lat_std[lat_std<1e-6]=1
    z_y=np.zeros((len(id2y),latent_dim),dtype=np.float32)
    for pid,row in zip(train_ids,lat): z_y[id2y[pid]]=(row-lat_mean[0])/lat_std[0]
    loader=DataLoader(RbdEmbeddingDataset(train_ids,train_x,z_y,id2y),batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate_batch)
    kmer_to_idx={k:i for i,k in enumerate(kmers)}
    decoy_kmers=['UUUUUUU','UUUUUUC','UUUUUUG','CUUUUUU','AUUUUUU','GUUUUUU','UUUUUCU','UUUUUCG']
    decoy_idx=[kmer_to_idx[k] for k in decoy_kmers if k in kmer_to_idx]
    pos_idx_map={pid:np.argsort(-y[id2y[pid]])[:args.ranking_top_k].astype(int) for pid in train_ids}
    model=PerResidueCnn(next(iter(train_x.values())).shape[1],args.hidden_dim,latent_dim,args.kernel_size,args.num_blocks,args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); mse=nn.MSELoss(); curve=[]
    comp_t=torch.tensor(svd.components_,device=device,dtype=torch.float32)
    scale_t=torch.tensor(scaler.scale_,device=device,dtype=torch.float32)
    mean_t=torch.tensor(scaler.mean_,device=device,dtype=torch.float32)
    lat_mean_t=torch.tensor(lat_mean,device=device,dtype=torch.float32)
    lat_std_t=torch.tensor(lat_std,device=device,dtype=torch.float32)
    log(f'train={len(train_ids)} query={len(query_x)} epochs={args.epochs} target_standardized=True ranking_weight={args.ranking_loss_weight}')
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch_ids,x,mask,target in loader:
            x=x.to(device); mask=mask.to(device); target=target.to(device); opt.zero_grad(set_to_none=True)
            pred=model(x,mask); loss=mse(pred,target)
            if args.ranking_loss_weight>0 and decoy_idx:
                pred_lat=pred*lat_std_t+lat_mean_t
                pred_scaled=pred_lat @ comp_t
                pred_y=pred_scaled*scale_t+mean_t
                rloss=[]
                for bi,pid in enumerate(batch_ids):
                    pos_np=pos_idx_map[pid]
                    pos_set=set(int(v) for v in pos_np)
                    neg_np=[j for j in decoy_idx if j not in pos_set] or decoy_idx
                    pos=torch.tensor(pos_np,device=device,dtype=torch.long)
                    neg=torch.tensor(neg_np,device=device,dtype=torch.long)
                    rloss.append(torch.relu(torch.tensor(args.ranking_margin,device=device)-pred_y[bi,pos].mean()+pred_y[bi,neg].mean()))
                loss=loss+args.ranking_loss_weight*torch.stack(rloss).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip); opt.step(); losses.append(float(loss.detach().cpu()))
        curve.append({'epoch':ep,'train_loss':float(np.mean(losses))})
        if ep%10==0: log(f'epoch={ep} train_loss={np.mean(losses):.4g}')
    # predict
    dummy_ids={p:i for i,p in enumerate(query_x)}; dummy=np.zeros((len(query_x),latent_dim),dtype=np.float32)
    qloader=DataLoader(RbdEmbeddingDataset(list(query_x),query_x,dummy,dummy_ids),batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_batch)
    rows=[]; matrix=[]; model.eval()
    with torch.no_grad():
        for ids,x,mask,_ in qloader:
            x=x.to(device); mask=mask.to(device); predz=model(x,mask).cpu().numpy(); pred_lat=predz*lat_std+lat_mean
            pred_scaled=svd.inverse_transform(pred_lat); pred=scaler.inverse_transform(pred_scaled)[0]; matrix.append(pred); pid=ids[0]
            for rank,idx in enumerate(np.argsort(-pred)[:args.top_n],1): rows.append({'query_id':pid,'short_id':pid.split('|')[0],'rank':rank,'kmer':kmers[idx],'score':float(pred[idx]),'rbd_length':query_len.get(pid,np.nan)})
    pd.DataFrame(rows).to_csv(out/'query_cnn_fixed_all_train_top_predicted_7mers.tsv',sep='\t',index=False)
    with gzip.open(out/'query_cnn_fixed_all_train_score_matrix.tsv.gz','wt') as h:
        h.write('query_id\t'+'\t'.join(kmers)+'\n')
        for pid,row in zip(list(query_x),matrix): h.write(pid+'\t'+'\t'.join(f'{float(v):.6g}' for v in row)+'\n')
    pd.DataFrame(curve).to_csv(out/'query_cnn_fixed_all_train_curve.tsv',sep='\t',index=False)
    (out/'query_cnn_fixed_all_train_summary.json').write_text(json.dumps({'train_proteins':len(train_ids),'query_proteins':len(query_x),'epochs':args.epochs,'latent_dim':latent_dim,'target_standardized':True,'ranking_loss_weight':args.ranking_loss_weight,'ranking_top_k':args.ranking_top_k,'ranking_margin':args.ranking_margin},indent=2)+'\n')
    log('Top motifs: '+str(out/'query_cnn_fixed_all_train_top_predicted_7mers.tsv'))
if __name__=='__main__': main()
