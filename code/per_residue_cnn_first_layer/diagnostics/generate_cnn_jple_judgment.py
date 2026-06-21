#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


def short(q):
    q = str(q)
    if q.startswith("AtPTBP3"):
        return "AtPTBP3"
    if "|original=" in q:
        return q.split("|original=", 1)[1].split("|", 1)[0]
    return q.split("|", 1)[0]


def main():
    out = Path("results/per_residue_cnn_first_layer/diagnostics_cnn_jple_20260617")
    support = pd.read_csv(out / "diagnostic3_query_neighbor_support.tsv", sep="\t")
    manual = {
        "w1": {
            "trust": "LOW",
            "judgment": "CNN+JPLE 当前预测为 CAU/mixed，不支持把它解释成 U-rich。",
            "reason": "query 的 RBD nearest-neighbor family 主要是 mixed/other，CNN+JPLE 输出也落在 mixed/CAU-rich；U-rich family 虽然总体可恢复，但 w1 在 CNN+JPLE latent 中没有被拉到 U-rich 区域。",
            "display": "不建议展示为可靠 motif；若展示，只能作为 CNN+JPLE 的低置信 CAU-rich/mixed 预测。",
        },
        "w2": {
            "trust": "LOW",
            "judgment": "当前预测 GA-rich，CUUCU-like 不可信。",
            "reason": "top50 RBD 近邻 GA-rich 占主导；CUUCU-like 在 RNAcompete family 诊断中样本极少且 pseudo-query top-k 恢复差，因此模型缺乏稳定训练支持。",
            "display": "不建议作为主要结果展示。",
        },
        "w3": {
            "trust": "MODERATE_HIGH",
            "judgment": "GAGUGUG / UGUGUG-like 预测可信度较高。",
            "reason": "纯 RBD kNN 近邻并不强支持 UGUGUG-like，说明不是简单 nearest-neighbor 平滑；但 UGUGUG-like family 在 pseudo-query 中 recoverability 高，CNN+JPLE 能输出一致 motif。",
            "display": "适合展示为 UGUGUG-like motif，建议说明不是 top-neighbor 直接复制。",
        },
        "w4": {
            "trust": "HIGH",
            "judgment": "UUUUUUU / U-rich 预测可信。",
            "reason": "U-rich family 在 pseudo-query 中最可恢复；w4 top50 近邻有明确 U-rich 支持，CNN+JPLE top motif 与 family 支持一致。",
            "display": "适合展示。",
        },
        "w5": {
            "trust": "LOW",
            "judgment": "当前 GA-rich 预测不作为可靠目标。",
            "reason": "RBD 近邻和 CNN+JPLE 输出均偏 GA-rich，但该 query 之前多模型不稳定；如果预期不是 GA-rich，则不可信。",
            "display": "不建议作为主要优化/展示目标。",
        },
        "w6": {
            "trust": "HIGH",
            "judgment": "UUUUUUU / U-rich 预测可信。",
            "reason": "CNN+JPLE top motif 是 U-rich；U-rich family 本身可恢复性高。虽然 top50 RBD 近邻没有 w4 那么干净，但 prediction 与可恢复 family 一致。",
            "display": "适合展示。",
        },
        "AtPTBP3": {
            "trust": "MODERATE_HIGH",
            "judgment": "CU/U-rich tendency 可信，但不要过度强调 exact top1。",
            "reason": "CNN+JPLE 输出 ACUUUCU/CUUUCAC 这类 CU/UCUC-like motif；CUUCU-like family 在训练集中支持少，因此 exact top1 不宜过度解读，但多个模型和诊断支持其 CU/U-rich 倾向。",
            "display": "适合展示为 CU/U-rich 或 CUUCU/UCUCUC-like tendency，避免说精确 top1 已确定。",
        },
    }
    rows = []
    for _, r in support.iterrows():
        sid = short(r.query_id)
        m = manual[sid]
        rows.append(
            {
                "query": sid,
                "cnn_jple_top1": r.current_cnn_jple_top1,
                "cnn_jple_top5": r.current_cnn_jple_top5,
                "cnn_jple_assigned_family_top50": r.current_cnn_jple_assigned_family_top50,
                "neighbor_majority_family": r.neighbor_majority_family,
                "neighbor_majority_fraction": r.neighbor_majority_fraction,
                "same_family_neighbor_support": r.neighbor_support_fraction_same_family,
                "trust_level": m["trust"],
                "judgment": m["judgment"],
                "failure_or_support_reason": m["reason"],
                "display_recommendation": m["display"],
            }
        )
    order = ["w1", "w2", "w3", "w4", "w5", "w6", "AtPTBP3"]
    df = pd.DataFrame(rows)
    df["_order"] = df["query"].map({q: i for i, q in enumerate(order)})
    df = df.sort_values("_order").drop(columns="_order")
    df.to_csv(out / "final_cnn_jple_query_judgment.tsv", sep="\t", index=False)

    lines = []
    lines.append("# CNN+JPLE Final Diagnostic Judgment\n\n")
    lines.append("## Overall\n\n")
    lines.append("- Use CNN+JPLE as the selected route, but interpret outputs by query-level diagnostic support.\n")
    lines.append("- JPLE latent decoder fidelity is good overall: Pearson mean 0.816, Spearman mean 0.771, top20 overlap mean 0.424, NDCG@20 mean 0.787.\n")
    lines.append("- U-rich and UGUGUG-like families are recoverable; CUUCU-like has weak RNAcompete support under this family definition.\n")
    lines.append("- Do not tune or select based on W1-W6/AtPTBP3 expected motifs.\n\n")
    lines.append("## Query Judgment\n\n")
    for _, r in df.iterrows():
        lines.append(f"### {r['query']}\n")
        lines.append(f"- CNN+JPLE top1: `{r['cnn_jple_top1']}`\n")
        lines.append(f"- top5: `{r['cnn_jple_top5']}`\n")
        lines.append(f"- assigned family: `{r['cnn_jple_assigned_family_top50']}`\n")
        lines.append(f"- neighbor majority: `{r['neighbor_majority_family']}` ({float(r['neighbor_majority_fraction']):.2f})\n")
        lines.append(f"- same-family neighbor support: {float(r['same_family_neighbor_support']):.2f}\n")
        lines.append(f"- trust: **{r['trust_level']}**\n")
        lines.append(f"- judgment: {r['judgment']}\n")
        lines.append(f"- reason: {r['failure_or_support_reason']}\n")
        lines.append(f"- display: {r['display_recommendation']}\n\n")
    lines.append("## Display Set\n\n")
    lines.append("- Suitable to display: w3, w4, w6, AtPTBP3.\n")
    lines.append("- Not suitable as reliable CNN+JPLE results: w1, w2, w5.\n")
    lines.append("- For AtPTBP3, display as CU/U-rich tendency rather than exact top1 certainty.\n")
    (out / "final_cnn_jple_diagnostic_judgment.md").write_text("".join(lines))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
