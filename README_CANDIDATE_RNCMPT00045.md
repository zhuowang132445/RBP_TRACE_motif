# Candidate RNCMPT00045 Independent Bundle

Final shell combination:
- RNCMPT00021 <- w1
- RNCMPT00045 <- w2
- RNCMPT00027 <- AtPTBP3

This bundle includes exact shell-swapped FASTA/RBD/ZSCORE inputs, CNN+JPLE code, logo code, rerun inputs, and final outputs.

Key files:
- inputs/exact_train_fasta/seq_train_shellswap_w1_rncmpt00045_atptbp3_train348.fasta
- inputs/exact_train_rbd_fasta/seq_train_shellswap_w1_rncmpt00045_atptbp3_domain_merged_rbd_train348.fasta
- inputs/exact_zscore/zscore_train_candidate_RNCMPT00045.tsv
- inputs/exact_motif_profiles/motif_profiles_candidate_RNCMPT00045.npz
- inputs/exact_train_embeddings/train_per_residue_candidate_RNCMPT00045.h5
- results/final_candidate_RNCMPT00045_run/per_residue_cnn/per_residue_cnn_jple_top_predicted_7mers.tsv
- results/final_candidate_RNCMPT00045_run/per_residue_cnn/query_seed_centered_logos_top10/query_seed_centered_top10_panel.png

Suggested rerun command:

```bash
cd /public/home/wz/workplace/cursor/RBP_TRACE/RBP_TRACE/candidate_RNCMPT00045_independent_bundle_20260618
/public/home/wz/anaconda3/bin/python code/per_residue_cnn_first_layer/run_jple_embedding_variants.py   --train-pooled-npz inputs/reference_embeddings/rnacompete_domain_merged_esmc_embeddings.npz   --query-pooled-rice-npz inputs/query_embeddings/rice_w1_w6_domain_merged_esmc_embeddings.npz   --train-per-residue-h5 inputs/exact_train_embeddings/train_per_residue_candidate_RNCMPT00045.h5   --query-rice-per-residue-h5 inputs/query_embeddings/rice_w1_w6_per_residue_esmc.h5   --query-atptbp3-per-residue-h5 inputs/query_embeddings/AtPTBP3_per_residue_esmc.h5   --motif-npz inputs/exact_motif_profiles/motif_profiles_candidate_RNCMPT00045.npz   --output-dir rerun_candidate_RNCMPT00045   --device cuda --gpu-memory-fraction 0.20 --epochs 75 --batch-size 4 --seed 20260617
```

Logo regeneration command:

```bash
cd /public/home/wz/workplace/cursor/RBP_TRACE/RBP_TRACE/candidate_RNCMPT00045_independent_bundle_20260618
/public/home/wz/anaconda3/bin/python code/per_residue_cnn_first_layer/make_seed_centered_logo_from_topkmers.py   --top-tsv results/final_candidate_RNCMPT00045_run/per_residue_cnn/per_residue_cnn_jple_top_predicted_7mers.tsv   --out-dir results/final_candidate_RNCMPT00045_run/per_residue_cnn/query_seed_centered_logos_top10   --top-k 10
```
