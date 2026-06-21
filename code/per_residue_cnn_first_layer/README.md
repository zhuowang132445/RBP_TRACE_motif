# Per-Residue RBD CNN Final Exploratory Logic

This module has been cleaned so the active CNN code path is explicit.

## Retained entry points

- `extract_per_residue_rbd_esmc_embeddings.py`: extract per-residue RBD ESMC embeddings into HDF5.
- `run_final_cnn_query_prediction.py`: final exploratory CNN query predictor.
- `cnn_model_utils.py`: shared lightweight model/data utilities used by the final predictor.

Deprecated one-off scripts and validation scripts were moved to `deprecated/`.

## Final CNN setting

The retained CNN logic is:

- input: per-residue RBD ESMC embedding, shape `L x D` per RBP
- training set: all aligned RNAcompete RBP with motif profiles
- target: kmer-wise standardized motif profiles
- motif dimension reduction: all-train `TruncatedSVD`, default `latent_dim=50`
- CNN: lightweight 1D convolution along RBD residue positions
- prediction: latent motif prediction followed by inverse SVD and inverse standardization
- regularization: weak ranking loss against generic U-rich decoys, default `--ranking-loss-weight 0.03`
- default training: 60 epochs, batch size 8, single CPU thread, CUDA allowed with memory cap

This CNN remains exploratory. Full plant LOO did not stably outperform the original `RBPTraceFirstLayer`, so the official first-layer model is still original RBPTrace + architecture/distance confidence. Use this CNN for query-level checks such as PTBP3/W2 UC-rich sensitivity tests.

## Recommended command

```bash
nice -n 10 env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 \
  python code/per_residue_cnn_first_layer/run_final_cnn_query_prediction.py \
    --train-h5 results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_esmc.h5 \
    --train-manifest results/per_residue_cnn_first_layer/features/rnacompete_rbd_per_residue_manifest.tsv \
    --query-h5 path/to/query_per_residue_esmc.h5 \
    --query-manifest path/to/query_per_residue_manifest.tsv \
    --motif-npz data/processed/motif_profiles.npz \
    --output-dir results/per_residue_cnn_first_layer/query_prediction \
    --device cuda \
    --gpu-memory-fraction 0.20 \
    --batch-size 8 \
    --epochs 60 \
    --latent-dim 50 \
    --ranking-loss-weight 0.03 \
    --torch-num-threads 1
```
