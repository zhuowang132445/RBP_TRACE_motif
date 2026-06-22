suppressWarnings(suppressMessages({
  options(stringsAsFactors = FALSE)
}))

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[idx + 1]
}

expr_rdata <- get_arg("--expr_rdata", "salmon_cdna_mirna200_tpm_filtered_log_tpmc3_rankn.RData")
expr_object <- get_arg("--expr_object", "tpmc_mn_3")
gene_list_file <- get_arg("--gene_list", "RBP_gene_list.txt")
mapping_file <- get_arg("--mapping", "mapping_table.txt")
sample_bam_map_file <- get_arg("--sample_bam_map", "as_reanalysis/metadata/sample_bam_base.tsv")
bam_metrics_file <- get_arg("--bam_metrics", "as_reanalysis/metadata/bam_metrics_by_sample.tsv")
exon_counts_file <- get_arg("--exon_counts", "stability_analysis/counts/exon_counts.txt")
intron_counts_file <- get_arg("--intron_counts", "stability_analysis/counts/intron_counts.txt")
out_dir <- get_arg("--out_dir", "stability_analysis/results")
proteins_arg <- get_arg("--proteins", "")
group_frac <- as.numeric(get_arg("--group_frac", "0.25"))
min_total_counts <- as.numeric(get_arg("--min_total_counts", "10"))
latent_pcs <- as.integer(get_arg("--latent_pcs", "5"))
strict_sample_map <- tolower(get_arg("--strict_sample_map", "false")) %in% c("1", "true", "yes")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

trim_cols <- function(x) {
  colnames(x) <- trimws(colnames(x))
  x
}

sort_fdbr <- function(x) {
  x <- unique(trimws(x))
  x <- x[x != ""]
  num <- suppressWarnings(as.integer(sub("^FDBR", "", x)))
  x[order(num, x, na.last = TRUE)]
}

safe_scale <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) {
    return(rep(0, length(x)))
  }
  (x - mean(x, na.rm = TRUE)) / s
}

read_mapping <- function(path) {
  df <- read.table(path, header = FALSE, skip = 1, sep = "", fill = TRUE, stringsAsFactors = FALSE)
  if (ncol(df) < 2) {
    stop("mapping_table.txt should contain at least 2 columns after the header.")
  }
  df <- df[, 1:2]
  colnames(df) <- c("FDBR", "Sample")
  df$FDBR <- trimws(df$FDBR)
  df$Sample <- trimws(df$Sample)
  unique(df[df$FDBR != "" & df$Sample != "", ])
}

read_sample_bam_map <- function(path) {
  if (!file.exists(path)) {
    return(NULL)
  }
  df <- read.delim(path, header = TRUE, check.names = FALSE)
  colnames(df) <- trimws(colnames(df))
  if (!("Sample" %in% colnames(df))) {
    df_raw <- read.delim(path, header = FALSE, check.names = FALSE)
    if (ncol(df_raw) >= 4) {
      df <- df_raw[, 1:4, drop = FALSE]
      colnames(df) <- c("Sample", "FDBR", "CRR", "BAMBase")
    } else if (ncol(df_raw) >= 2) {
      df <- df_raw[, 1:2, drop = FALSE]
      colnames(df) <- c("Sample", "BAMBase")
    } else {
      return(NULL)
    }
  }
  bam_col <- intersect(c("BAMBase", "CRR", "BAM", "bam_base"), colnames(df))
  if (length(bam_col) == 0) {
    return(NULL)
  }
  bam_col <- bam_col[1]
  df <- df[, c("Sample", bam_col), drop = FALSE]
  colnames(df) <- c("Sample", "BAMBase")
  df$Sample <- trimws(df$Sample)
  df$BAMBase <- trimws(df$BAMBase)
  unique(df[df$Sample != "" & df$BAMBase != "", ])
}

read_featurecounts <- function(path) {
  df <- read.delim(path, comment.char = "#", check.names = FALSE)
  df <- trim_cols(df)
  count_cols <- setdiff(colnames(df), c("Geneid", "Chr", "Start", "End", "Strand", "Length"))
  mat <- as.matrix(df[, count_cols, drop = FALSE])
  storage.mode(mat) <- "numeric"
  rownames(mat) <- df$Geneid
  colnames(mat) <- basename(colnames(mat))
  mat
}

read_bam_metrics <- function(path) {
  if (!file.exists(path)) {
    return(NULL)
  }
  df <- read.delim(path, header = TRUE, check.names = FALSE)
  colnames(df) <- trimws(colnames(df))
  if (!("Sample" %in% colnames(df))) {
    return(NULL)
  }
  df$Sample <- trimws(df$Sample)
  if ("mapped_reads" %in% colnames(df)) {
    df$mapped_reads <- suppressWarnings(as.numeric(df$mapped_reads))
    df$log_mapped_reads <- log10(df$mapped_reads + 1)
  }
  if ("batch" %in% colnames(df)) {
    df$batch <- trimws(df$batch)
    df$batch[df$batch == ""] <- "unknown"
  }
  keep_cols <- intersect(c("Sample", "mapped_reads", "log_mapped_reads", "batch"), colnames(df))
  unique(df[, keep_cols, drop = FALSE])
}

collapse_region_counts <- function(mat, suffix) {
  if (nrow(mat) == 0) {
    return(mat)
  }
  # New SAF files already use gene-level GeneID directly, so no further
  # collapsing is needed unless legacy exon/intron suffixes are present.
  if (!any(grepl(paste0("\\|", suffix, "\\|"), rownames(mat), perl = TRUE))) {
    return(mat)
  }
  base_gene <- sub(paste0("\\|", suffix, "\\|.*$"), "", rownames(mat))
  split_idx <- split(seq_len(nrow(mat)), base_gene)
  out <- vapply(split_idx, function(idx) colSums(mat[idx, , drop = FALSE], na.rm = TRUE), numeric(ncol(mat)))
  out <- t(out)
  if (is.null(dim(out))) {
    out <- matrix(out, nrow = 1, dimnames = list(names(split_idx)[1], colnames(mat)))
  }
  rownames(out) <- names(split_idx)
  colnames(out) <- colnames(mat)
  out
}

log_cpm <- function(mat, prior = 0.5) {
  libsizes <- colSums(mat, na.rm = TRUE)
  scale_factor <- libsizes / 1e6
  sweep(mat + prior, 2, scale_factor + 1e-8, "/") |> log2()
}

clean_bam_base <- function(x) {
  x <- basename(x)
  sub("(_new)?\\.Aligned\\.sortedByCoord\\.out\\.bam$", "", x)
}

infer_sample_bam_map <- function(mapping_df, bam_names) {
  fdbr_ordered <- sort_fdbr(mapping_df$FDBR)
  sample_ordered <- mapping_df$Sample[match(fdbr_ordered, mapping_df$FDBR)]
  bam_bases <- clean_bam_base(sort(unique(bam_names)))
  if (length(fdbr_ordered) != length(bam_bases)) {
    stop(sprintf(
      paste(
        "Cannot infer sample-BAM mapping automatically:",
        "mapping_table has %d entries but featureCounts has %d BAM columns.",
        "Provide --sample_bam_map with an explicit Sample/BAMBase table or narrow the BAM input set."
      ),
      length(fdbr_ordered), length(bam_bases)
    ))
  }
  data.frame(
    Sample = sample_ordered,
    FDBR = fdbr_ordered,
    BAMBase = bam_bases,
    Inference = "mapping_order_vs_featureCounts_columns",
    stringsAsFactors = FALSE
  )
}

load(expr_rdata)
if (!exists(expr_object)) {
  stop(sprintf("Object %s not found in %s", expr_object, expr_rdata))
}
expr_mat <- get(expr_object)
colnames(expr_mat) <- trimws(colnames(expr_mat))
rownames(expr_mat) <- trimws(rownames(expr_mat))

mapping_df <- read_mapping(mapping_file)
bam_metrics_df <- read_bam_metrics(bam_metrics_file)
gene_df <- read.table(gene_list_file, header = FALSE, stringsAsFactors = FALSE)
gene_df <- gene_df[, 1:2]
colnames(gene_df) <- c("GeneID", "Protein")
gene_df$GeneID <- trimws(gene_df$GeneID)
gene_df$Protein <- trimws(gene_df$Protein)
if (proteins_arg != "") {
  keep_prot <- trimws(strsplit(proteins_arg, ",")[[1]])
  gene_df <- gene_df[gene_df$Protein %in% keep_prot, , drop = FALSE]
}

exon_mat_raw <- read_featurecounts(exon_counts_file)
intron_mat_raw <- read_featurecounts(intron_counts_file)
exon_mat <- collapse_region_counts(exon_mat_raw, "exon")
intron_mat <- collapse_region_counts(intron_mat_raw, "intron")
shared_genes <- intersect(rownames(exon_mat), rownames(intron_mat))
shared_bams <- intersect(colnames(exon_mat), colnames(intron_mat))
exon_mat <- exon_mat[shared_genes, shared_bams, drop = FALSE]
intron_mat <- intron_mat[shared_genes, shared_bams, drop = FALSE]

sample_bam_map <- read_sample_bam_map(sample_bam_map_file)
if (is.null(sample_bam_map)) {
  sample_bam_map <- infer_sample_bam_map(mapping_df, shared_bams)
}
sample_bam_map$Sample <- trimws(sample_bam_map$Sample)
sample_bam_map$BAMBase <- clean_bam_base(sample_bam_map$BAMBase)
write.table(
  sample_bam_map,
  file.path(out_dir, "sample_bam_map_used.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

bam_lookup <- clean_bam_base(shared_bams)
sample_lookup <- setNames(sample_bam_map$Sample, sample_bam_map$BAMBase)
matched_samples <- sample_lookup[bam_lookup]
valid_idx <- !is.na(matched_samples)
unmatched_bams <- bam_lookup[!valid_idx]
write.table(
  data.frame(BAMBase = unmatched_bams, stringsAsFactors = FALSE),
  file.path(out_dir, "unmatched_bam_bases.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
if (length(unmatched_bams) > 0) {
  msg <- paste(
    "Unmatched BAM bases were dropped:",
    paste(utils::head(unmatched_bams, 20), collapse = ", ")
  )
  warning(msg, call. = FALSE)
  if (strict_sample_map) {
    stop(sprintf(
      "Strict sample map enabled and %d BAMs were unmatched. See %s",
      length(unmatched_bams),
      file.path(out_dir, "unmatched_bam_bases.tsv")
    ))
  }
}
exon_mat <- exon_mat[, valid_idx, drop = FALSE]
intron_mat <- intron_mat[, valid_idx, drop = FALSE]
matched_samples <- matched_samples[valid_idx]
colnames(exon_mat) <- matched_samples
colnames(intron_mat) <- matched_samples

shared_samples <- Reduce(intersect, list(colnames(exon_mat), colnames(intron_mat), colnames(expr_mat)))
if (length(shared_samples) < 50) {
  stop("Too few shared samples between counts and expression matrix.")
}
shared_samples <- sort(shared_samples)
exon_mat <- exon_mat[, shared_samples, drop = FALSE]
intron_mat <- intron_mat[, shared_samples, drop = FALSE]
expr_mat <- expr_mat[, shared_samples, drop = FALSE]

exon_totals <- rowSums(exon_mat, na.rm = TRUE)
intron_totals <- rowSums(intron_mat, na.rm = TRUE)
keep_genes <- shared_genes[exon_totals >= min_total_counts & intron_totals >= min_total_counts]
exon_mat <- exon_mat[keep_genes, , drop = FALSE]
intron_mat <- intron_mat[keep_genes, , drop = FALSE]

exon_logcpm <- log_cpm(exon_mat)
intron_logcpm <- log_cpm(intron_mat)

all_rbp_ids <- unique(gene_df$GeneID)
expr_for_pca <- expr_mat[setdiff(rownames(expr_mat), all_rbp_ids), , drop = FALSE]
expr_for_pca <- expr_for_pca[, shared_samples, drop = FALSE]
gene_sd <- apply(expr_for_pca, 1, function(x) sd(as.numeric(x), na.rm = TRUE))
expr_for_pca <- expr_for_pca[is.finite(gene_sd) & gene_sd > 0, , drop = FALSE]
if (nrow(expr_for_pca) < 2) {
  stop("Too few variable genes remain for PCA after removing target RBP genes.")
}
pc_fit <- prcomp(t(expr_for_pca), center = TRUE, scale. = FALSE)
pc_keep <- seq_len(min(latent_pcs, ncol(pc_fit$x)))
pc_scores <- pc_fit$x[, pc_keep, drop = FALSE]
pc_df <- data.frame(Sample = rownames(pc_scores), pc_scores, check.names = FALSE)
colnames(pc_df)[-1] <- paste0("ExprPC", seq_len(ncol(pc_scores)))

cov_df <- merge(data.frame(Sample = shared_samples, stringsAsFactors = FALSE), pc_df, by = "Sample", all.x = TRUE, sort = FALSE)
cov_df$SampleClass <- substr(cov_df$Sample, 1, 1)
if (!is.null(bam_metrics_df)) {
  cov_df <- merge(cov_df, bam_metrics_df, by = "Sample", all.x = TRUE, sort = FALSE)
}
if ("batch" %in% names(cov_df)) {
  cov_df$batch[is.na(cov_df$batch) | cov_df$batch == ""] <- "unknown"
  cov_df$batch <- factor(cov_df$batch)
}
if ("log_mapped_reads" %in% names(cov_df)) {
  missing_lmr <- is.na(cov_df$log_mapped_reads)
  if (any(missing_lmr)) {
    med_lmr <- median(cov_df$log_mapped_reads, na.rm = TRUE)
    if (is.finite(med_lmr)) {
      cov_df$log_mapped_reads[missing_lmr] <- med_lmr
    }
  }
}
write.table(
  cov_df,
  file.path(out_dir, "sample_covariates_used.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

build_covariate_terms <- function(df, include_group = FALSE) {
  terms <- character()
  if (include_group) {
    terms <- c(terms, "Group")
  } else {
    terms <- c(terms, "RBPExpression")
  }
  terms <- c(terms, "IntronLogCPM", "SampleClass")
  if ("log_mapped_reads" %in% names(df)) {
    terms <- c(terms, "log_mapped_reads")
  }
  if ("batch" %in% names(df)) {
    batch_n <- length(unique(as.character(df$batch[!is.na(df$batch)])))
    if (batch_n > 1) {
      terms <- c(terms, "batch")
    }
  }
  c(terms, grep("^ExprPC", names(df), value = TRUE))
}

fit_residual_model <- function(prot, gene_id) {
  if (!(gene_id %in% rownames(expr_mat))) {
    return(NULL)
  }
  rbp_expr <- as.numeric(expr_mat[gene_id, cov_df$Sample])
  rbp_expr <- safe_scale(rbp_expr)
  names(rbp_expr) <- cov_df$Sample
  rhs_terms <- build_covariate_terms(cov_df, include_group = FALSE)
  form <- as.formula(paste("ExonLogCPM ~", paste(rhs_terms, collapse = " + ")))

  out <- vector("list", nrow(exon_logcpm))
  for (i in seq_len(nrow(exon_logcpm))) {
    gene_name <- rownames(exon_logcpm)[i]
    if (identical(gene_name, gene_id)) next
    work <- cov_df
    work$ExonLogCPM <- as.numeric(exon_logcpm[gene_name, work$Sample])
    work$IntronLogCPM <- as.numeric(intron_logcpm[gene_name, work$Sample])
    work$RBPExpression <- rbp_expr[work$Sample]
    work <- work[complete.cases(work[, c("ExonLogCPM", "IntronLogCPM", "RBPExpression")]), , drop = FALSE]
    if (nrow(work) < 80) next
    fit <- tryCatch(lm(form, data = work), error = function(e) NULL)
    if (is.null(fit)) next
    coefs <- summary(fit)$coefficients
    if (!("RBPExpression" %in% rownames(coefs))) next
    out[[i]] <- data.frame(
      RBP = prot,
      RBP_GeneID = gene_id,
      GeneID = gene_name,
      N = nrow(work),
      Beta_RBP = coefs["RBPExpression", "Estimate"],
      SE_RBP = coefs["RBPExpression", "Std. Error"],
      T_RBP = coefs["RBPExpression", "t value"],
      PValue_RBP = coefs["RBPExpression", "Pr(>|t|)"],
      Beta_Intron = if ("IntronLogCPM" %in% rownames(coefs)) coefs["IntronLogCPM", "Estimate"] else NA_real_,
      stringsAsFactors = FALSE
    )
  }
  res <- do.call(rbind, out)
  if (is.null(res) || nrow(res) == 0) return(NULL)
  res$FDR_RBP <- p.adjust(res$PValue_RBP, method = "BH")
  res <- res[order(res$FDR_RBP, -abs(res$Beta_RBP)), ]
  res
}

fit_group_eisa <- function(prot, gene_id) {
  if (!(gene_id %in% rownames(expr_mat))) {
    return(NULL)
  }
  expr_vec <- as.numeric(expr_mat[gene_id, cov_df$Sample])
  names(expr_vec) <- cov_df$Sample
  low_cut <- quantile(expr_vec, probs = group_frac, na.rm = TRUE)
  high_cut <- quantile(expr_vec, probs = 1 - group_frac, na.rm = TRUE)
  group <- ifelse(expr_vec <= low_cut, "low", ifelse(expr_vec >= high_cut, "high", NA))
  keep_samples <- names(group)[!is.na(group)]
  work_cov <- cov_df[cov_df$Sample %in% keep_samples, , drop = FALSE]
  work_cov$Group <- factor(group[work_cov$Sample], levels = c("low", "high"))
  rhs_terms <- build_covariate_terms(work_cov, include_group = TRUE)
  form <- as.formula(paste("ExonLogCPM ~", paste(rhs_terms, collapse = " + ")))

  out <- vector("list", nrow(exon_logcpm))
  for (i in seq_len(nrow(exon_logcpm))) {
    gene_name <- rownames(exon_logcpm)[i]
    if (identical(gene_name, gene_id)) next
    work <- work_cov
    work$ExonLogCPM <- as.numeric(exon_logcpm[gene_name, work$Sample])
    work$IntronLogCPM <- as.numeric(intron_logcpm[gene_name, work$Sample])
    work <- work[complete.cases(work[, c("ExonLogCPM", "IntronLogCPM", "Group")]), , drop = FALSE]
    if (nrow(work) < 40) next
    if (length(unique(work$Group)) < 2) next
    fit <- tryCatch(lm(form, data = work), error = function(e) NULL)
    if (is.null(fit)) next
    coefs <- summary(fit)$coefficients
    if (!("Grouphigh" %in% rownames(coefs))) next
    high_samples <- work$Sample[work$Group == "high"]
    low_samples <- work$Sample[work$Group == "low"]
    delta_exon <- mean(exon_logcpm[gene_name, high_samples]) - mean(exon_logcpm[gene_name, low_samples])
    delta_intron <- mean(intron_logcpm[gene_name, high_samples]) - mean(intron_logcpm[gene_name, low_samples])
    out[[i]] <- data.frame(
      RBP = prot,
      RBP_GeneID = gene_id,
      GeneID = gene_name,
      N = nrow(work),
      HighN = length(high_samples),
      LowN = length(low_samples),
      DeltaExon = delta_exon,
      DeltaIntron = delta_intron,
      DeltaPost = delta_exon - delta_intron,
      Beta_Group = coefs["Grouphigh", "Estimate"],
      PValue_Group = coefs["Grouphigh", "Pr(>|t|)"],
      stringsAsFactors = FALSE
    )
  }
  res <- do.call(rbind, out)
  if (is.null(res) || nrow(res) == 0) return(NULL)
  res$FDR_Group <- p.adjust(res$PValue_Group, method = "BH")
  res <- res[order(res$FDR_Group, -abs(res$DeltaPost)), ]
  res
}

summary_rows <- list()
for (i in seq_len(nrow(gene_df))) {
  gene_id <- gene_df$GeneID[i]
  prot <- gene_df$Protein[i]
  residual_df <- fit_residual_model(prot, gene_id)
  group_df <- fit_group_eisa(prot, gene_id)
  if (!is.null(residual_df)) {
    write.table(residual_df,
      file.path(out_dir, paste0(prot, "_residual_continuous.tsv")),
      sep = "\t", quote = FALSE, row.names = FALSE
    )
  }
  if (!is.null(group_df)) {
    write.table(group_df,
      file.path(out_dir, paste0(prot, "_eisa_quartile.tsv")),
      sep = "\t", quote = FALSE, row.names = FALSE
    )
  }
  summary_rows[[prot]] <- data.frame(
    Protein = prot,
    GeneID = gene_id,
    SharedSamples = length(shared_samples),
    TestedGenes = nrow(exon_logcpm),
    ResidualHitsFDR05 = if (is.null(residual_df)) 0 else sum(residual_df$FDR_RBP < 0.05, na.rm = TRUE),
    QuartileHitsFDR05 = if (is.null(group_df)) 0 else sum(group_df$FDR_Group < 0.05, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

summary_df <- do.call(rbind, summary_rows)
write.table(summary_df,
  file.path(out_dir, "summary_by_rbp.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)
write.table(
  data.frame(Sample = shared_samples, stringsAsFactors = FALSE),
  file.path(out_dir, "shared_samples_used.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

message("[INFO] Shared samples: ", length(shared_samples))
message("[INFO] Genes tested: ", nrow(exon_logcpm))
