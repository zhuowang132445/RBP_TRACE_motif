#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict


ATTR_PAT = re.compile(r'(\S+)\s+"([^"]+)"')


def parse_attrs(attr_text):
    attrs = {}
    for key, value in ATTR_PAT.findall(attr_text):
        attrs[key] = value
    return attrs


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        prev = merged[-1]
        if start <= prev[1] + 1:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def write_saf(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("GeneID\tChr\tStart\tEnd\tStrand\n")
        for row in rows:
            handle.write(
                f"{row['GeneID']}\t{row['Chr']}\t{row['Start']}\t{row['End']}\t{row['Strand']}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Build gene-level exon and intron SAF files from a GTF.")
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gene-id-key", default="gene_id")
    parser.add_argument("--feature", default="exon")
    parser.add_argument("--min-intron-len", type=int, default=20)
    args = parser.parse_args()

    gene_meta = {}
    exons_by_gene = defaultdict(list)

    with open(args.gtf, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs_text = parts
            if feature != args.feature:
                continue
            attrs = parse_attrs(attrs_text)
            gene_id = attrs.get(args.gene_id_key)
            if not gene_id:
                continue
            start = int(start)
            end = int(end)
            exons_by_gene[gene_id].append((start, end))
            if gene_id not in gene_meta:
                gene_meta[gene_id] = {"Chr": chrom, "Strand": strand}

    exon_rows = []
    intron_rows = []
    for gene_id, intervals in sorted(exons_by_gene.items()):
        meta = gene_meta[gene_id]
        merged_exons = merge_intervals(intervals)
        for start, end in merged_exons:
            exon_rows.append(
                {
                    # Reuse the same GeneID across all exon blocks so featureCounts
                    # aggregates directly to gene-level mature-transcript counts.
                    "GeneID": gene_id,
                    "Chr": meta["Chr"],
                    "Start": start,
                    "End": end,
                    "Strand": meta["Strand"],
                }
            )
        if len(merged_exons) < 2:
            continue
        for idx in range(len(merged_exons) - 1):
            intron_start = merged_exons[idx][1] + 1
            intron_end = merged_exons[idx + 1][0] - 1
            if intron_end - intron_start + 1 < args.min_intron_len:
                continue
            intron_rows.append(
                {
                    # Likewise, intron blocks for one gene should collapse to a
                    # gene-level proxy for transcriptional input.
                    "GeneID": gene_id,
                    "Chr": meta["Chr"],
                    "Start": intron_start,
                    "End": intron_end,
                    "Strand": meta["Strand"],
                }
            )

    os.makedirs(args.outdir, exist_ok=True)
    write_saf(os.path.join(args.outdir, "exon_regions.saf"), exon_rows)
    write_saf(os.path.join(args.outdir, "intron_regions.saf"), intron_rows)
    print(f"[INFO] exon_regions={len(exon_rows)} intron_regions={len(intron_rows)}")


if __name__ == "__main__":
    main()
