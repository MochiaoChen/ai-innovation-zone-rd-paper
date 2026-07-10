"""
09_summary_report.py — 汇总所有输出，生成 RESULTS_SUMMARY.md
"""
import os
import numpy as np
import pandas as pd

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN      = os.path.join(BASE, "data/clean")
OUT_TABLES = os.path.join(BASE, "output/tables")
OUT_FIGS   = os.path.join(BASE, "output/figures")

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)

lines = []
lines.append("# RESULTS_SUMMARY.md")
lines.append(f"\n生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("\n---\n")

# ── 1. 样本信息
lines.append("## 1. 样本统计\n")
lines.append(f"- 总观测数 N = **{len(df):,}**")
lines.append(f"- 公司数 = **{df['stkcd'].nunique():,}**")
lines.append(f"- 年份范围 = {df['year'].min()} — {df['year'].max()}")
if "ever_treated" in df.columns:
    n_treat = df[df["ever_treated"] == 1]["stkcd"].nunique()
    n_ctrl  = df[df["ever_treated"] == 0]["stkcd"].nunique()
    lines.append(f"- 处理组公司数 = **{n_treat:,}**")
    lines.append(f"- 对照组公司数 = **{n_ctrl:,}**")
if "treat_year" in df.columns:
    batch_dist = df[df["treat_year"].notna()].groupby("treat_year")["stkcd"].nunique()
    lines.append("\n批次分布（公司数）：")
    for yr, cnt in batch_dist.items():
        lines.append(f"  - {int(yr)} 年批次：{cnt} 家公司")

lines.append("")

# ── 2. 关键变量描述统计
lines.append("## 2. 关键变量摘要\n")
for var, label in [("RD", "RD Intensity"), ("CAPEX", "CAPEX Intensity"),
                   ("Treat", "Treatment"), ("TobinQ", "Tobin's Q")]:
    if var in df.columns:
        s = df[var].dropna()
        lines.append(f"- **{label}**：Mean={s.mean():.4f}, Std={s.std():.4f}, "
                     f"P50={s.median():.4f}")

lines.append("")

# ── 3. 主回归结果
lines.append("## 3. 主回归系数（Table 2）\n")
import json as _json
json_path = os.path.join(CLEAN, "main_results.json")
if os.path.exists(json_path):
    try:
        with open(json_path) as f:
            coef_summary = _json.load(f)
        for y, vals in coef_summary.items():
            coef = vals.get("coef", float("nan"))
            tstat = vals.get("tstat", float("nan"))
            pval = vals.get("pval", float("nan"))
            n = vals.get("N", 0)
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            lines.append(f"- **{y}**：Treat = {coef:.4f}{stars} (t={tstat:.3f}, N={n:,})")
    except Exception as e:
        lines.append(f"⚠️  无法加载 main_results.json：{e}")
else:
    lines.append("⚠️  main_results.json 不存在（请先跑 05_main_regression.py）")

lines.append("")

# ── 4. 已生成文件清单
lines.append("## 4. 输出文件\n")
for d, label in [(OUT_TABLES, "Tables"), (OUT_FIGS, "Figures")]:
    lines.append(f"### {label}\n")
    if os.path.exists(d):
        for fname in sorted(os.listdir(d)):
            fpath = os.path.join(d, fname)
            size_kb = os.path.getsize(fpath) / 1024
            lines.append(f"- `{fname}` ({size_kb:.1f} KB)")
    else:
        lines.append("（目录不存在）")
    lines.append("")

# ── 5. 用户确认清单
lines.append("## 5. 用户核实清单（来自 instruction.md §12）\n")
lines.append("- [ ] `data/treatment/ai_zones.csv` 城市名单已与科技部官网核实")
lines.append(f"- [ ] Table 1 样本量是否在 3 万—4 万 firm-year 范围内（当前 N={len(df):,}）")
lines.append("- [ ] Figure 1 冲击前各期系数（k = -3, -2）是否不显著（平行趋势成立）")
lines.append("- [ ] Table 2 中 Treat 系数符号是否正向（政策促进研发）")
lines.append("- [ ] TobinQ 缺失率过高时，考虑重跑 `01b_fetch_prices.py` 并更新分析数据")
lines.append("- [ ] SOE 哑变量当前为关键词估算，建议用官方控制人数据替换")
lines.append("- [ ] 是否需要补充行业×年份固定效应版本")

# ── 输出
out_path = os.path.join(BASE, "RESULTS_SUMMARY.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[STEP 9 COMPLETE] RESULTS_SUMMARY.md 已生成：{out_path}")

# 打印到终端
print("\n" + "=" * 60)
for line in lines[:40]:
    print(line)
print("... (详见 RESULTS_SUMMARY.md)")
print("=" * 60)
