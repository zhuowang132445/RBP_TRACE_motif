#!/public/home/wz/anaconda3/bin/python
from __future__ import annotations

import argparse
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

LOGO_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7931E", "U": "#D62828"}
ALPHABET = ["A", "C", "G", "U"]
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}
FONT = FontProperties(weight="bold")


def get_bits(freqs):
    entropy = 0.0
    for f in freqs:
        if f > 0:
            entropy -= f * math.log2(f)
    return 2.0 - entropy


def softmax_weights(scores: np.ndarray) -> np.ndarray:
    x = scores - scores.max()
    w = np.exp(x)
    return w / w.sum()


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def best_offset_to_seed(seed: str, kmer: str, max_shift: int = 2) -> tuple[int, int, int]:
    canvas_len = len(seed) + 2 * max_shift
    seed_start = max_shift
    best = None
    for off in range(-max_shift, max_shift + 1):
        start = seed_start + off
        end = start + len(kmer)
        if start < 0 or end > canvas_len:
            continue
        overlap_start = max(seed_start, start)
        overlap_end = min(seed_start + len(seed), end)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue
        seed_sub = seed[overlap_start - seed_start : overlap_end - seed_start]
        kmer_sub = kmer[overlap_start - start : overlap_end - start]
        mism = hamming(seed_sub, kmer_sub)
        key = (mism, -overlap, abs(off))
        if best is None or key < best[0]:
            best = (key, off, overlap, mism)
    if best is None:
        return 0, 0, len(seed)
    return best[1], best[3], best[2]


def build_pwm(seed: str, kmers: list[str], weights: np.ndarray, max_shift: int = 2):
    canvas_len = len(seed) + 2 * max_shift
    pwm = np.full((canvas_len, 4), 1e-6, dtype=float)
    coverage = np.zeros(canvas_len, dtype=float)
    seed_start = max_shift
    rows = []
    for rank, (kmer, w) in enumerate(zip(kmers, weights), start=1):
        off, mism, overlap = best_offset_to_seed(seed, kmer, max_shift=max_shift)
        start = seed_start + off
        aligned = ["."] * canvas_len
        for i, ch in enumerate(kmer):
            pos = start + i
            if 0 <= pos < canvas_len and ch in BASE_TO_IDX:
                pwm[pos, BASE_TO_IDX[ch]] += w
                coverage[pos] += w
                aligned[pos] = ch
        rows.append({
            "rank": rank,
            "kmer": kmer,
            "weight": float(w),
            "offset_vs_seed": int(off),
            "overlap_with_seed": int(overlap),
            "mismatch_in_overlap": int(mism),
            "aligned": "".join(aligned),
        })
    pwm = pwm / pwm.sum(axis=1, keepdims=True)
    return pwm, coverage, pd.DataFrame(rows)


def draw_logo(ax, pwm, coverage, min_bits=0.05):
    informative = []
    for pos in range(pwm.shape[0]):
        bits = get_bits(pwm[pos])
        if bits >= min_bits and coverage[pos] > 0.05 * coverage.max():
            informative.append(pos)
    if informative:
        display = list(range(min(informative), max(informative) + 1))
    else:
        display = list(range(pwm.shape[0]))
    for x_idx, pos in enumerate(display, start=1):
        freq = {b: float(pwm[pos, i]) for i, b in enumerate(ALPHABET)}
        bits = get_bits(freq.values())
        bases_sorted = sorted(freq.items(), key=lambda x: x[1])
        y_offset = 0.0
        for base, f in bases_sorted:
            height = f * bits
            if height < 0.02:
                continue
            path = TextPath((0, 0), base, size=1, prop=FONT)
            bbox = path.get_extents()
            trans = (Affine2D().translate(-bbox.x0, -bbox.y0).scale(0.85 / bbox.width, height / bbox.height).translate(x_idx - 0.42, y_offset))
            patch = PathPatch(path, transform=trans + ax.transData, color=LOGO_COLORS.get(base, "black"), lw=0)
            ax.add_patch(patch)
            y_offset += height
    ax.set_xlim(0.3, len(display) + 0.7)
    ax.set_ylim(0, 2.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    top = pd.read_csv(args.top_tsv, sep="\t")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"w1", "w2", "w3", "w4", "w6", "AtPTBP3"}
    order = []
    seen = set()
    for q in top["short_id"].astype(str):
        if q in wanted and q not in seen:
            seen.add(q)
            order.append(q)

    panel_rows = math.ceil(len(order) / 2)
    fig, axes = plt.subplots(panel_rows, 2, figsize=(10, 3.0 * panel_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    summary = []

    for ax, q in zip(axes, order):
        sub = top[top["short_id"].astype(str) == q].sort_values("rank").head(args.top_k).copy()
        kmers = sub["kmer"].astype(str).tolist()
        scores = sub["score"].astype(float).to_numpy()
        weights = softmax_weights(scores)
        seed = kmers[0]
        pwm, coverage, align_df = build_pwm(seed, kmers, weights)
        draw_logo(ax, pwm, coverage)
        ax.set_title(f"{q} | seed={seed}", fontsize=11, fontweight="bold")
        align_df.to_csv(out_dir / f"{q}_top{args.top_k}_seed_aligned.tsv", sep="\t", index=False)
        pwm_df = pd.DataFrame({"Pos": np.arange(1, pwm.shape[0] + 1), "A": pwm[:,0], "C": pwm[:,1], "G": pwm[:,2], "T": pwm[:,3]})
        pwm_df.to_csv(out_dir / f"{q}_PWM_1.txt", sep="\t", index=False, float_format="%.6f")
        single_fig, single_ax = plt.subplots(figsize=(5.5, 1.8))
        draw_logo(single_ax, pwm, coverage)
        single_ax.set_title(f"{q} | seed={seed}", fontsize=11, fontweight="bold")
        single_fig.savefig(out_dir / f"{q}_seed_centered_top{args.top_k}_logo.png", dpi=300, bbox_inches="tight")
        plt.close(single_fig)
        summary.append({"Protein": q, "Motif_ID": 1, "Family_Label": "seed_centered_top10", "Representative_RNA": seed, "Seed_RNA": seed, "N_Selected_Members": len(kmers), "Selected_Total_Weight": float(weights.sum()), "N_Family_Members": len(kmers), "Family_Total_Weight": float(weights.sum()), "Top_Baseline_Z": float(scores[0]), "Weight_Fraction": 1.0})

    for ax in axes[len(order):]:
        ax.axis("off")
    fig.savefig(out_dir / "query_seed_centered_top10_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary).to_csv(out_dir / "motif_logo_summary_table.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
