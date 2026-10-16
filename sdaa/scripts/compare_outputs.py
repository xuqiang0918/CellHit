#!/usr/bin/env python
"""两平台 external_inference 输出对照（预测分布 / 逐点偏差 / SHAP top-k）。

用法：
    python sdaa/scripts/compare_outputs.py <sdaa_csv> <cuda_csv> \
        [--shaps <sdaa_shap_csv> <cuda_shap_csv>]

输入：full_inference/<drugID>.csv（列含 Index/Predictions/Source）；
      可选 full_inference_shaps/<drugID>.csv（行=样本，列=基因）。
输出：逐项打印对照指标，用于 SDAA_ADAPTATION.md 精度表。
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("sdaa_csv")
ap.add_argument("cuda_csv")
ap.add_argument("--shaps", nargs=2, metavar=("SDAA_SHAP", "CUDA_SHAP"), default=None)
args = ap.parse_args()

sdaa = pd.read_csv(args.sdaa_csv)
cuda = pd.read_csv(args.cuda_csv)

print("== rows: sdaa %d | cuda %d" % (len(sdaa), len(cuda)))
print("\n== 预测分布（逐来源）==")
for src in sorted(set(sdaa["Source"]) & set(cuda["Source"])):
    for name, df in [("SDAA", sdaa), ("CUDA", cuda)]:
        a = df[df["Source"] == src]["Predictions"].values
        print("%s %-6s n=%-6d mean=%.4f std=%.4f med=%.4f" % (name, src, len(a), a.mean(), a.std(), np.median(a)))

m = sdaa.merge(cuda, on="Index", suffixes=("_s", "_c"))
d = (m["Predictions_s"] - m["Predictions_c"]).abs()
r = np.corrcoef(m["Predictions_s"], m["Predictions_c"])[0, 1]
print("\n== 逐点对照 ==")
print("merged=%d pearson r=%.5f absdiff mean=%.4f med=%.4f p95=%.4f max=%.4f"
      % (len(m), r, d.mean(), d.median(), d.quantile(0.95), d.max()))
print("<0.1: %.1f%% | <0.2: %.1f%%" % ((d < 0.1).mean() * 100, (d < 0.2).mean() * 100))
for src in sorted(set(m["Source_s"])):
    sub = d[m["Source_s"] == src]
    print("  %-6s mean_absdiff=%.4f max=%.4f" % (src, sub.mean(), sub.max()))

if args.shaps:
    sa = pd.read_csv(args.shaps[0], index_col=0)
    ca = pd.read_csv(args.shaps[1], index_col=0)
    ms, mc = sa.abs().mean(), ca.abs().mean()
    genes = ms.index.intersection(mc.index)
    rr = np.corrcoef(ms.loc[genes], mc.loc[genes])[0, 1]
    top50 = len(set(ms.nlargest(50).index) & set(mc.nlargest(50).index))
    top5_s = ms.nlargest(5).index.tolist()
    top5_c = mc.nlargest(5).index.tolist()
    print("\n== SHAP mean|shap| 对照 ==")
    print("genes=%d pearson r=%.4f top50 overlap=%d/50 top5 same order=%s"
          % (len(genes), rr, top50, top5_s == top5_c))
    print("top5 sdaa:", ",".join(top5_s))
    print("top5 cuda:", ",".join(top5_c))
