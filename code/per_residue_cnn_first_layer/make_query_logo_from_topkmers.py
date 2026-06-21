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
FONT = FontProperties(family="DejaVu Sans", weight="bold")


def short_queries(df: pd.DataFrame) -> list[str]:
    order = []
    seen = set()
    for q in df["short_id"].astype(str):
        if q not in seen:
            seen.add(q)
            order.append(q)
    return order


def pwm_from_topk(sub: pd.DataFrame, top_k: int = 50) -> np.ndarray:
    sub = sub.sort_values("rank").head(top_k).copy()
    kmers = sub["kmer"].astype(str).tolist()
    scores = sub["score"].astype(float).to_numpy()
    if len(kmers) == 0:
        return np.zeros((7, 4), dtype=float)
    scores = scores - scores.max()
    weights = np.exp(scores)
    weights = weights / weights.sum()
    pwm = np.zeros((len(kmers[0]), 4), dtype=float)
    idx = {b: i for i, b in enumerate(ALPHABET)}
    for kmer, w in zip(kmers, weights):
        for pos, ch in enumerate(kmer):
            if ch in idx:
                pwm[pos, idx[ch]] += w
    return pwm


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


def draw_logo(ax, pwm: np.ndarray, title: str) -> None:
    for pos in range(pwm.shape[0]):
        pairs = sorted([(ALPHABET[i], pwm[pos, i]) for i in range(4)], key=lambda x: x[1])
        y = 0.0
        for base, h in pairs:
            draw_letter(ax, base, pos, y, h)
            y += h
    ax.set_xlim(0, pwm.shape[0])
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(pwm.shape[0]) + 0.5)
    ax.set_xticklabels([str(i + 1) for i in range(pwm.shape[0])], fontsize=9)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("weight", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    top = pd.read_csv(args.top_tsv, sep="\t")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = [q for q in short_queries(top) if q in {"w1", "w2", "w3", "w4", "w6", "AtPTBP3"}]

    summary_rows = []
    ncols = 2
    nrows = int(np.ceil(len(queries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, q in zip(axes, queries):
        sub = top[top["short_id"].astype(str) == q]
        pwm = pwm_from_topk(sub, top_k=args.top_k)
        draw_logo(ax, pwm, q)
        np.savetxt(out_dir / f"{q}_top{args.top_k}_logo_pwm.tsv", pwm, fmt="%.6f", delimiter="\t")
        summary_rows.append({"query": q, "top1": sub.sort_values("rank").iloc[0]["kmer"], "top_k": args.top_k})
        single_fig, single_ax = plt.subplots(1, 1, figsize=(7, 2.4), constrained_layout=True)
        draw_logo(single_ax, pwm, q)
        single_fig.savefig(out_dir / f"{q}_top{args.top_k}_logo.png", dpi=220)
        plt.close(single_fig)

    for ax in axes[len(queries):]:
        ax.axis("off")
    fig.savefig(out_dir / f"query_logos_top{args.top_k}_panel.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(out_dir / "logo_summary.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
