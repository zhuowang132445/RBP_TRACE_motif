#!/public/home/wz/anaconda3/bin/python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

COLORS = {"A": "#2E86DE", "C": "#27AE60", "G": "#F39C12", "U": "#C0392B"}
ALPHABET = ["A", "C", "G", "U"]
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}
FONT = FontProperties(family="DejaVu Sans", weight="bold")


def ordered_queries(df: pd.DataFrame) -> list[str]:
    seen = set()
    out = []
    for q in df["short_id"].astype(str):
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def softmax_weights(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    x = scores - scores.max()
    w = np.exp(x)
    return w / w.sum()


def aligned_positions(kmer_len: int, canvas_len: int, offset: int) -> list[int]:
    start = (canvas_len - kmer_len) // 2 + offset
    return list(range(start, start + kmer_len))


def init_pwm(canvas_len: int) -> np.ndarray:
    return np.full((canvas_len, 4), 1e-6, dtype=float)


def update_pwm(kmers: list[str], weights: np.ndarray, offsets: list[int], canvas_len: int) -> tuple[np.ndarray, np.ndarray]:
    pwm = init_pwm(canvas_len)
    coverage = np.zeros(canvas_len, dtype=float)
    for kmer, w, off in zip(kmers, weights, offsets):
        pos = aligned_positions(len(kmer), canvas_len, off)
        for p, ch in zip(pos, kmer):
            if 0 <= p < canvas_len and ch in BASE_TO_IDX:
                pwm[p, BASE_TO_IDX[ch]] += w
                coverage[p] += w
    norm = pwm.sum(axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    pwm = pwm / norm
    return pwm, coverage


def alignment_score(kmer: str, pwm: np.ndarray, offset: int) -> float:
    pos = aligned_positions(len(kmer), pwm.shape[0], offset)
    score = 0.0
    for p, ch in zip(pos, kmer):
        if 0 <= p < pwm.shape[0] and ch in BASE_TO_IDX:
            score += float(np.log(max(pwm[p, BASE_TO_IDX[ch]], 1e-9)))
    return score


def optimize_offsets(kmers: list[str], weights: np.ndarray, canvas_len: int, max_shift: int, rounds: int = 6) -> tuple[list[int], np.ndarray, np.ndarray]:
    offsets = [0] * len(kmers)
    pwm, coverage = update_pwm(kmers, weights, offsets, canvas_len)
    for _ in range(rounds):
        new_offsets = []
        for kmer in kmers:
            best_off = 0
            best_score = None
            for off in range(-max_shift, max_shift + 1):
                s = alignment_score(kmer, pwm, off)
                if best_score is None or s > best_score:
                    best_score = s
                    best_off = off
            new_offsets.append(best_off)
        offsets = new_offsets
        pwm, coverage = update_pwm(kmers, weights, offsets, canvas_len)
    return offsets, pwm, coverage


def consensus_from_pwm(pwm: np.ndarray, coverage: np.ndarray, min_cov: float) -> tuple[str, int, int]:
    keep = np.where(coverage >= min_cov)[0]
    if len(keep) == 0:
        keep = np.arange(pwm.shape[0])
    start, end = int(keep[0]), int(keep[-1])
    letters = [ALPHABET[int(np.argmax(pwm[i]))] for i in range(start, end + 1)]
    return "".join(letters), start, end


def draw_letter(ax, letter: str, x: float, y: float, height: float, width: float = 0.9) -> None:
    if height <= 0:
        return
    tp = TextPath((0, 0), letter, size=1, prop=FONT)
    bb = tp.get_extents()
    sx = width / bb.width
    sy = height / bb.height
    trans = Affine2D().scale(sx, sy).translate(x + (1 - width) / 2 - bb.x0 * sx, y - bb.y0 * sy)
    patch = PathPatch(tp, transform=trans + ax.transData, color=COLORS[letter], lw=0)
    ax.add_patch(patch)


def draw_logo(ax, pwm: np.ndarray, coverage: np.ndarray, title: str, min_cov: float) -> None:
    keep = np.where(coverage >= min_cov)[0]
    if len(keep) == 0:
        keep = np.arange(pwm.shape[0])
    start, end = int(keep[0]), int(keep[-1])
    view = pwm[start:end + 1]
    for i in range(view.shape[0]):
        pairs = sorted([(ALPHABET[j], view[i, j]) for j in range(4)], key=lambda x: x[1])
        y = 0.0
        for base, h in pairs:
            draw_letter(ax, base, i, y, h)
            y += h
    ax.set_xlim(0, view.shape[0])
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(view.shape[0]) + 0.5)
    ax.set_xticklabels([str(i + 1) for i in range(view.shape[0])], fontsize=9)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("weight", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def write_alignment_table(path: Path, kmers: list[str], weights: np.ndarray, offsets: list[int], canvas_len: int) -> None:
    rows = []
    for rank, (kmer, w, off) in enumerate(zip(kmers, weights, offsets), start=1):
        chars = ["."] * canvas_len
        for p, ch in zip(aligned_positions(len(kmer), canvas_len, off), kmer):
            if 0 <= p < canvas_len:
                chars[p] = ch
        rows.append({
            "rank": rank,
            "kmer": kmer,
            "weight": float(w),
            "offset": off,
            "aligned": "".join(chars),
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--canvas-len", type=int, default=11)
    ap.add_argument("--max-shift", type=int, default=2)
    args = ap.parse_args()

    top = pd.read_csv(args.top_tsv, sep="\t")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = [q for q in ordered_queries(top) if q in {"w1", "w2", "w3", "w4", "w6", "AtPTBP3"}]

    summary = []
    ncols = 2
    nrows = int(np.ceil(len(queries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, q in zip(axes, queries):
        sub = top[top["short_id"].astype(str) == q].sort_values("rank").head(args.top_k).copy()
        kmers = sub["kmer"].astype(str).tolist()
        weights = softmax_weights(sub["score"].astype(float).to_numpy())
        offsets, pwm, coverage = optimize_offsets(kmers, weights, args.canvas_len, args.max_shift)
        min_cov = float(max(0.15, 0.25 * coverage.max()))
        consensus, start, end = consensus_from_pwm(pwm, coverage, min_cov)
        draw_logo(ax, pwm, coverage, f"{q} | {consensus}", min_cov)
        write_alignment_table(out_dir / f"{q}_top{args.top_k}_aligned_kmers.tsv", kmers, weights, offsets, args.canvas_len)
        pd.DataFrame(pwm, columns=ALPHABET).to_csv(out_dir / f"{q}_top{args.top_k}_aligned_pwm.tsv", sep="\t", index=False)
        single_fig, single_ax = plt.subplots(1, 1, figsize=(7, 2.5), constrained_layout=True)
        draw_logo(single_ax, pwm, coverage, f"{q} | {consensus}", min_cov)
        single_fig.savefig(out_dir / f"{q}_top{args.top_k}_shifted_logo.png", dpi=220)
        plt.close(single_fig)
        summary.append({"query": q, "consensus": consensus, "kept_start": start + 1, "kept_end": end + 1, "top1": kmers[0]})

    for ax in axes[len(queries):]:
        ax.axis("off")
    fig.savefig(out_dir / f"query_shifted_logos_top{args.top_k}_panel.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(summary).to_csv(out_dir / "shifted_logo_summary.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
