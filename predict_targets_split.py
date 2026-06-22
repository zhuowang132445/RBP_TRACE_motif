#!/usr/bin/python3
import csv
import os
import re
import statistics
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTOME_FASTA = os.path.join(SCRIPT_DIR, "rice_v7_transcripts.fa")
SEED_ASSIGNMENT_FILE = os.path.join(
    SCRIPT_DIR, "pwm_spacing_analysis", "article_strict_seed_assignments.tsv"
)
REGION_SCAN_FILE = os.path.join(SCRIPT_DIR, "region_scan_summary.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "split_target_predictions")

PROTEINS = ["w1", "w2", "w3", "w4", "w5", "w6"]
FIVE_UTR_CLASS = "5UTR_spacing"
THREE_UTR_CLASS = "3UTR_density"
SHORT_GAP_MAX = 5
WINDOW_SIZE = 30


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_top_motifs(path):
    motifs = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein = row["protein"]
            if protein not in PROTEINS:
                continue
            if row.get("is_top1_motif", "0").strip() != "1":
                continue
            motifs[protein] = {
                "motif": row["assigned_motif"].strip().upper(),
                "pwm_file": row["pwm_file"].strip(),
                "pwm_rank": int(row["pwm_rank"]),
                "selection_rule": row.get("top1_selection_rule", "").strip(),
            }
    missing = [protein for protein in PROTEINS if protein not in motifs]
    if missing:
        raise ValueError(f"Missing top motifs for proteins: {', '.join(missing)}")
    return motifs


def load_region_scan(path):
    stats = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein = row["Protein"]
            region = row["Region"]
            if protein not in PROTEINS:
                continue
            try:
                enrichment = float(row["EnrichmentRatio"]) if row["EnrichmentRatio"] else 0.0
            except ValueError:
                enrichment = 0.0
            try:
                fdr = float(row["FDR"]) if row["FDR"] else 1.0
            except ValueError:
                fdr = 1.0
            stats[protein][region] = {
                "enrichment_ratio": enrichment,
                "fdr": fdr,
                "hit_fraction": float(row["HitFraction"]) if row["HitFraction"] else 0.0,
                "gene_count_with_hit": int(row["GeneCountWithHit"]) if row["GeneCountWithHit"] else 0,
            }
    return stats


def assign_classes(region_stats):
    assignments = {}
    for protein in PROTEINS:
        five = region_stats[protein].get("5UTR", {"enrichment_ratio": 0.0, "fdr": 1.0})
        three = region_stats[protein].get("3UTR", {"enrichment_ratio": 0.0, "fdr": 1.0})
        if five["fdr"] < 0.05 and five["enrichment_ratio"] >= three["enrichment_ratio"]:
            assignments[protein] = FIVE_UTR_CLASS
        elif three["fdr"] < 0.05:
            assignments[protein] = THREE_UTR_CLASS
        elif five["enrichment_ratio"] >= three["enrichment_ratio"]:
            assignments[protein] = FIVE_UTR_CLASS
        else:
            assignments[protein] = THREE_UTR_CLASS
    return assignments


def load_fasta(path):
    with open(path, "r", encoding="utf-8") as handle:
        header = None
        seq_parts = []
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts).upper()
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts).upper()


def parse_header(header):
    tx_id = header[1:].split()[0]
    gene_id = re.sub(r"\.[0-9]+$", "", tx_id)
    match = re.search(r"CDS=(\d+)-(\d+)", header)
    if match:
        cds_start = int(match.group(1))
        cds_end = int(match.group(2))
    else:
        cds_start = None
        cds_end = None
    return tx_id, gene_id, cds_start, cds_end


def split_transcript_regions(seq, cds_start, cds_end):
    if cds_start is None or cds_end is None:
        return "", "", ""
    cds_s = max(0, min(len(seq), cds_start - 1))
    cds_e = max(cds_s, min(len(seq), cds_end))
    return seq[:cds_s], seq[cds_s:cds_e], seq[cds_e:]


def find_overlapping_hits(seq, motif):
    starts = []
    search_from = 0
    while True:
        idx = seq.find(motif, search_from)
        if idx == -1:
            break
        starts.append(idx)
        search_from = idx + 1
    return starts


def compute_closest_gap(starts, motif_len):
    if len(starts) < 2:
        return None
    gaps = [
        starts[idx + 1] - (starts[idx] + motif_len)
        for idx in range(len(starts) - 1)
    ]
    return min(gaps)


def format_positions(starts):
    return ",".join(str(pos) for pos in starts)


def longest_run(seq, base="T"):
    best = 0
    current = 0
    for char in seq:
        if char == base:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def max_hits_in_window(starts, motif_len, window_size):
    if not starts:
        return 0
    best = 1
    left = 0
    for right in range(len(starts)):
        while starts[right] + motif_len - starts[left] > window_size:
            left += 1
        size = right - left + 1
        if size > best:
            best = size
    return best


def max_overlap_cluster_hits(starts, motif_len):
    if not starts:
        return 0
    best = 1
    current = 1
    cluster_end = starts[0] + motif_len
    for idx in range(1, len(starts)):
        start = starts[idx]
        if start < cluster_end:
            current += 1
            cluster_end = max(cluster_end, start + motif_len)
        else:
            current = 1
            cluster_end = start + motif_len
        if current > best:
            best = current
    return best


def score_five_utr(hit_count, closest_gap):
    if hit_count >= 2 and closest_gap is not None and closest_gap <= SHORT_GAP_MAX:
        return 3000 - closest_gap
    if hit_count >= 2:
        return 2000 + hit_count
    return 1000 + hit_count


def score_three_utr(hit_count, density_per_kb, max_window_hits, max_t_run):
    return (
        hit_count * 1000
        + max_window_hits * 100
        + int(density_per_kb * 10)
        + max_t_run
    )


def build_five_utr_row(protein, motif, tx_id, gene_id, utr5):
    starts = find_overlapping_hits(utr5, motif)
    if not starts:
        return None
    closest_gap = compute_closest_gap(starts, len(motif))
    row = {
        "protein": protein,
        "analysis_class": FIVE_UTR_CLASS,
        "gene_id": gene_id,
        "transcript_id": tx_id,
        "motif": motif,
        "region": "5UTR",
        "region_length_nt": len(utr5),
        "motif_count": len(starts),
        "hit_positions_0based": format_positions(starts),
        "closest_gap": closest_gap if closest_gap is not None else "",
        "short_gap_le5": int(closest_gap is not None and closest_gap <= SHORT_GAP_MAX),
        "configuration": (
            "fused"
            if closest_gap is not None and closest_gap <= 0
            else "dispersed"
            if closest_gap is not None and closest_gap <= SHORT_GAP_MAX
            else ""
        ),
        "score": score_five_utr(len(starts), closest_gap if closest_gap is not None else 999),
        "candidate_tier": (
            "tier1_short_gap_le5"
            if closest_gap is not None and closest_gap <= SHORT_GAP_MAX
            else "tier2_two_plus"
            if len(starts) >= 2
            else "tier3_one_hit"
        ),
    }
    return row


def build_three_utr_row(protein, motif, tx_id, gene_id, utr3):
    starts = find_overlapping_hits(utr3, motif)
    if not starts:
        return None
    density_per_kb = (len(starts) / (len(utr3) / 1000.0)) if len(utr3) > 0 else 0.0
    max_window_hits = max_hits_in_window(starts, len(motif), WINDOW_SIZE)
    row = {
        "protein": protein,
        "analysis_class": THREE_UTR_CLASS,
        "gene_id": gene_id,
        "transcript_id": tx_id,
        "motif": motif,
        "region": "3UTR",
        "region_length_nt": len(utr3),
        "motif_count": len(starts),
        "motif_density_per_kb": density_per_kb,
        "hit_positions_0based": format_positions(starts),
        "max_hits_in_30nt_window": max_window_hits,
        "max_overlap_cluster_hits": max_overlap_cluster_hits(starts, len(motif)),
        "longest_t_run": longest_run(utr3, "T"),
        "score": score_three_utr(len(starts), density_per_kb, max_window_hits, longest_run(utr3, "T")),
        "candidate_tier": (
            "tier1_clustered_or_multi"
            if len(starts) >= 2 or max_window_hits >= 2
            else "tier2_single"
        ),
    }
    return row


def choose_best_rows(rows):
    best = {}
    for row in rows:
        gene_id = row["gene_id"]
        current = best.get(gene_id)
        if current is None or row["score"] > current["score"]:
            best[gene_id] = row
    return sorted(best.values(), key=lambda row: (-row["score"], row["gene_id"]))


def write_tsv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float):
                    output[field] = f"{value:.6f}"
                else:
                    output[field] = value
            writer.writerow(output)


def write_gene_list(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['gene_id']}\n")


def main():
    ensure_dir(OUTPUT_DIR)
    five_dir = os.path.join(OUTPUT_DIR, FIVE_UTR_CLASS)
    three_dir = os.path.join(OUTPUT_DIR, THREE_UTR_CLASS)
    ensure_dir(five_dir)
    ensure_dir(three_dir)

    top_motifs = load_top_motifs(SEED_ASSIGNMENT_FILE)
    region_stats = load_region_scan(REGION_SCAN_FILE)
    assignments = assign_classes(region_stats)

    assignment_rows = []
    transcript_rows = defaultdict(list)
    for protein in PROTEINS:
        motif_info = top_motifs[protein]
        five = region_stats[protein].get("5UTR", {"enrichment_ratio": 0.0, "fdr": 1.0})
        three = region_stats[protein].get("3UTR", {"enrichment_ratio": 0.0, "fdr": 1.0})
        assignment_rows.append(
            {
                "protein": protein,
                "assigned_class": assignments[protein],
                "top_motif": motif_info["motif"],
                "pwm_file": motif_info["pwm_file"],
                "pwm_rank": motif_info["pwm_rank"],
                "selection_rule": motif_info["selection_rule"],
                "five_utr_enrichment_ratio": five["enrichment_ratio"],
                "five_utr_fdr": five["fdr"],
                "three_utr_enrichment_ratio": three["enrichment_ratio"],
                "three_utr_fdr": three["fdr"],
            }
        )

    for header, seq in load_fasta(TRANSCRIPTOME_FASTA):
        tx_id, gene_id, cds_start, cds_end = parse_header(header)
        utr5, _, utr3 = split_transcript_regions(seq, cds_start, cds_end)
        for protein in PROTEINS:
            motif = top_motifs[protein]["motif"]
            if assignments[protein] == FIVE_UTR_CLASS:
                row = build_five_utr_row(protein, motif, tx_id, gene_id, utr5)
            else:
                row = build_three_utr_row(protein, motif, tx_id, gene_id, utr3)
            if row is not None:
                transcript_rows[protein].append(row)

    write_tsv(
        os.path.join(OUTPUT_DIR, "class_assignment.tsv"),
        [
            "protein",
            "assigned_class",
            "top_motif",
            "pwm_file",
            "pwm_rank",
            "selection_rule",
            "five_utr_enrichment_ratio",
            "five_utr_fdr",
            "three_utr_enrichment_ratio",
            "three_utr_fdr",
        ],
        assignment_rows,
    )

    summary_rows = []
    for protein in PROTEINS:
        rows = sorted(transcript_rows[protein], key=lambda row: (-row["score"], row["gene_id"], row["transcript_id"]))
        gene_rows = choose_best_rows(rows)
        class_name = assignments[protein]
        outdir = five_dir if class_name == FIVE_UTR_CLASS else three_dir

        if class_name == FIVE_UTR_CLASS:
            fieldnames = [
                "protein",
                "analysis_class",
                "gene_id",
                "transcript_id",
                "motif",
                "region",
                "region_length_nt",
                "motif_count",
                "hit_positions_0based",
                "closest_gap",
                "short_gap_le5",
                "configuration",
                "candidate_tier",
                "score",
            ]
        else:
            fieldnames = [
                "protein",
                "analysis_class",
                "gene_id",
                "transcript_id",
                "motif",
                "region",
                "region_length_nt",
                "motif_count",
                "motif_density_per_kb",
                "hit_positions_0based",
                "max_hits_in_30nt_window",
                "max_overlap_cluster_hits",
                "longest_t_run",
                "candidate_tier",
                "score",
            ]

        write_tsv(os.path.join(outdir, f"{protein}_transcript_candidates.tsv"), fieldnames, rows)
        write_tsv(os.path.join(outdir, f"{protein}_gene_candidates.tsv"), fieldnames, gene_rows)
        write_gene_list(os.path.join(outdir, f"{protein}_candidate_genes.txt"), gene_rows)

        motif_counts = [row["motif_count"] for row in rows]
        summary_rows.append(
            {
                "protein": protein,
                "assigned_class": class_name,
                "top_motif": top_motifs[protein]["motif"],
                "transcript_hits": len(rows),
                "gene_hits": len(gene_rows),
                "tier1_genes": sum(row["candidate_tier"].startswith("tier1") for row in gene_rows),
                "median_motif_count": statistics.median(motif_counts) if motif_counts else 0.0,
                "max_score": max((row["score"] for row in gene_rows), default=0),
            }
        )

    write_tsv(
        os.path.join(OUTPUT_DIR, "split_target_summary.tsv"),
        [
            "protein",
            "assigned_class",
            "top_motif",
            "transcript_hits",
            "gene_hits",
            "tier1_genes",
            "median_motif_count",
            "max_score",
        ],
        summary_rows,
    )

    notes_path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(notes_path, "w", encoding="utf-8") as handle:
        handle.write("Split target prediction pipeline\n")
        handle.write("================================\n")
        handle.write("Class assignment source: corrected region_scan_summary.csv\n")
        handle.write("5UTR_spacing: proteins with stronger significant 5UTR enrichment than 3UTR enrichment\n")
        handle.write("3UTR_density: proteins with stronger or fallback 3UTR enrichment\n")
        handle.write("Top motif source: pwm_spacing_analysis/article_strict_seed_assignments.tsv\n")
        handle.write("Scanning mode: exact overlapping motif search in transcript orientation\n")
        handle.write("5UTR_spacing ranking: short-gap <=5 > other 2x > 1x\n")
        handle.write("3UTR_density ranking: motif count, local 30-nt clustering, density, and T-run length\n")

    print(os.path.join(OUTPUT_DIR, "class_assignment.tsv"))
    print(os.path.join(OUTPUT_DIR, "split_target_summary.tsv"))
    print(notes_path)


if __name__ == "__main__":
    main()
