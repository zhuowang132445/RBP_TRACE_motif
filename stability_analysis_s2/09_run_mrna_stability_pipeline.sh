#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/public/home/zhuowang/smoke/egwas}"
OUT_DIR="${2:-${ROOT_DIR}/stability_analysis}"
THREADS="${THREADS:-16}"
STRANDNESS="${STRANDNESS:-2}"
BAM_DIR="${BAM_DIR:-${ROOT_DIR}/02_StarAlign}"
BAM_PATTERN="${BAM_PATTERN:-*Aligned.sortedByCoord.out.bam}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURECOUNTS_BIN="${FEATURECOUNTS_BIN:-featureCounts}"
R_BIN="${R_BIN:-Rscript}"
PAIRED_END="${PAIRED_END:-1}"
STRICT_SAMPLE_MAP="${STRICT_SAMPLE_MAP:-0}"

GTF="${GTF:-${ROOT_DIR}/rice_all_genomes_v7.gtf}"
EXPR_RDATA="${EXPR_RDATA:-${ROOT_DIR}/salmon_cdna_mirna200_tpm_filtered_log_tpmc3_rankn.RData}"
EXPR_OBJECT="${EXPR_OBJECT:-tpmc_mn_3}"
GENE_LIST="${GENE_LIST:-${ROOT_DIR}/RBP_gene_list.txt}"
MAPPING_FILE="${MAPPING_FILE:-${ROOT_DIR}/mapping_table.txt}"
SAMPLE_BAM_MAP="${SAMPLE_BAM_MAP:-${ROOT_DIR}/as_reanalysis/metadata/sample_bam_base.tsv}"
BAM_METRICS="${BAM_METRICS:-${ROOT_DIR}/as_reanalysis/metadata/bam_metrics_by_sample.tsv}"
SCRIPT_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-${SCRIPT_DIR_DEFAULT}}"

mkdir -p "${OUT_DIR}/annotation" "${OUT_DIR}/counts" "${OUT_DIR}/results"

set +u
source /etc/profile >/dev/null 2>&1 || true
set -u
module load R/4.3.2 >/dev/null 2>&1 || true

if command -v module >/dev/null 2>&1; then
  module use /public/home/software/opt/bio/modules/all >/dev/null 2>&1 || true
  module load Subread/2.0.0 >/dev/null 2>&1 || \
    module load Subread >/dev/null 2>&1 || \
    module load subread >/dev/null 2>&1 || true
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[ERROR] Neither python3 nor python is available." >&2
    exit 1
  fi
fi

if ! command -v "${FEATURECOUNTS_BIN}" >/dev/null 2>&1 && [ ! -x "${FEATURECOUNTS_BIN}" ]; then
  echo "[ERROR] featureCounts not found. Try 'module avail subread' or install subread in your conda env." >&2
  exit 1
fi

if ! command -v "${R_BIN}" >/dev/null 2>&1 && [ ! -x "${R_BIN}" ]; then
  echo "[ERROR] Rscript not found. Set R_BIN to an absolute Rscript path if needed." >&2
  exit 1
fi

mapfile -t BAM_FILES < <(find "${BAM_DIR}" -maxdepth 1 -type f -name "${BAM_PATTERN}" | sort)
if [ "${#BAM_FILES[@]}" -eq 0 ]; then
  echo "[ERROR] No BAM files matched ${BAM_DIR}/${BAM_PATTERN}" >&2
  exit 1
fi
printf "%s\n" "${BAM_FILES[@]}" > "${OUT_DIR}/counts/bam_inputs.txt"
echo "[INFO] BAM files to count: ${#BAM_FILES[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/07_make_eisa_saf.py" \
  --gtf "${GTF}" \
  --outdir "${OUT_DIR}/annotation"

FC_ARGS=(-T "${THREADS}" -F SAF -s "${STRANDNESS}")
if [ "${PAIRED_END}" = "1" ]; then
  FC_ARGS+=(-p -B -C)
fi

"${FEATURECOUNTS_BIN}" \
  "${FC_ARGS[@]}" \
  -a "${OUT_DIR}/annotation/exon_regions.saf" \
  -o "${OUT_DIR}/counts/exon_counts.txt" \
  "${BAM_FILES[@]}"

"${FEATURECOUNTS_BIN}" \
  "${FC_ARGS[@]}" \
  -a "${OUT_DIR}/annotation/intron_regions.saf" \
  -o "${OUT_DIR}/counts/intron_counts.txt" \
  "${BAM_FILES[@]}"

"${R_BIN}" "${SCRIPT_DIR}/08_fit_mrna_stability_models.R" \
  --expr_rdata "${EXPR_RDATA}" \
  --expr_object "${EXPR_OBJECT}" \
  --gene_list "${GENE_LIST}" \
  --mapping "${MAPPING_FILE}" \
  --sample_bam_map "${SAMPLE_BAM_MAP}" \
  --bam_metrics "${BAM_METRICS}" \
  --exon_counts "${OUT_DIR}/counts/exon_counts.txt" \
  --intron_counts "${OUT_DIR}/counts/intron_counts.txt" \
  --out_dir "${OUT_DIR}/results" \
  --strict_sample_map "${STRICT_SAMPLE_MAP}"
