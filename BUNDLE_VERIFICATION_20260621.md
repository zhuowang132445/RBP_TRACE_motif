# Bundle Verification 2026-06-21

Status: PASS for independent execution.

What was verified:
- The bundle reruns the CNN+JPLE training/prediction script using only files inside the bundle.
- The rerun completed successfully and produced new pooled and per-residue CNN+JPLE outputs under rerun_candidate_RNCMPT00045/.
- The logo regeneration script also reran successfully from the rerun top-kmer table.

Important caveat:
- The rerun is not byte-identical to the packaged original GPU run.
- Differences are small and consistent with GPU non-determinism / training-order variation.
- Core motif-family conclusions are preserved:
  - w1 remains U-rich top1
  - w2 remains CUUCU-like top1/family
  - w3 remains UGUGUG-like family
  - w4 remains U-rich top1
  - w6 remains U-rich top1
  - AtPTBP3 remains CU/U-rich with top1 CUUUCUU

Observed rerun differences vs packaged original:
- w3 top1 changed within the same UGUGUG-like family (GUGUGUG vs UGUGUGG)
- Minor top10 reorderings for w1/w2/w4/w6/AtPTBP3
- run_config.json differs only by input/output paths, because rerun uses bundle-relative paths.

Interpretation:
- The bundle is operationally independent.
- It is suitable for migration and rerunning.
- Do not expect strict bitwise reproducibility from the GPU training step unless deterministic training settings are explicitly enforced.
