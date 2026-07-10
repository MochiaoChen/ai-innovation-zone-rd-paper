"""
04_descriptive.py — 描述性统计 Table 1
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean")
OUT_TABLES = os.path.join(BASE, "output/tables")

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)
print(f"分析样本: {df.shape}")

# ── 1. 描述性统计主表 ─────────────────────────────────────────────────
VARS = ["RD", "CAPEX", "RD_staff", "Treat", "Size", "Lev", "ROA", "Age", "Cash", "TobinQ", "SOE", "HighAI"]
LABELS = {
    "RD": "R\\&D Intensity",
    "CAPEX": "CAPEX Intensity",
    "RD_staff": "R\\&D Personnel Ratio",
    "Treat": "Treatment ($Treat_{it}$)",
    "Size": "Firm Size (log assets)",
    "Lev": "Leverage",
    "ROA": "Return on Assets",
    "Age": "Firm Age (log)",
    "Cash": "Cash Holdings",
    "TobinQ": "Tobin's Q",
    "SOE": "State-owned Enterprise",
    "HighAI": "High AI Exposure Industry",
}

rows = []
for var in VARS:
    if var not in df.columns:
        continue
    s = df[var].dropna()
    rows.append({
        "Variable": LABELS.get(var, var),
        "N": int(s.count()),
        "Mean": s.mean(),
        "Std": s.std(),
        "P25": s.quantile(0.25),
        "Median": s.median(),
        "P75": s.quantile(0.75),
    })

desc = pd.DataFrame(rows)

# CSV 输出
desc.to_csv(os.path.join(OUT_TABLES, "table1_descriptive.csv"), index=False, encoding="utf-8-sig")

# LaTeX 输出
def fmt(x, decimals=3):
    if pd.isna(x):
        return ""
    return f"{x:,.{decimals}f}"

def fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(x):,}"

tex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Descriptive Statistics}",
    r"\label{tab:descriptive}",
    r"\begin{threeparttable}",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Variable & $N$ & Mean & Std.\ Dev. & P25 & Median & P75 \\",
    r"\midrule",
]
for _, row in desc.iterrows():
    line = (
        f"{row['Variable']} & {fmt_int(row['N'])} & {fmt(row['Mean'])} & "
        f"{fmt(row['Std'])} & {fmt(row['P25'])} & {fmt(row['Median'])} & {fmt(row['P75'])} \\\\"
    )
    tex_lines.append(line)

tex_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{tablenotes}",
    r"\small",
    r"\item \textit{Notes}: The sample covers A-share listed firms in China from 2013 to 2025, "
    r"excluding financial firms (banking and non-bank finance) and ST/\*ST firms. "
    r"All continuous variables are winsorized at the 1st and 99th percentiles.",
    r"\end{tablenotes}",
    r"\end{threeparttable}",
    r"\end{table}",
]

tex_path = os.path.join(OUT_TABLES, "table1_descriptive.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

# ── 2. 分组统计（处理组 vs 对照组）──────────────────────────────────
if "ever_treated" in df.columns:
    treat_df = df[df["ever_treated"] == 1]
    ctrl_df = df[df["ever_treated"] == 0]
    print(f"\n  处理组公司数: {treat_df['stkcd'].nunique()}")
    print(f"  对照组公司数: {ctrl_df['stkcd'].nunique()}")

# ── 3. 批次分布 ───────────────────────────────────────────────────────
if "treat_year" in df.columns:
    treated_firms = df[df["ever_treated"] == 1][["stkcd", "treat_year"]].drop_duplicates()
    batch_dist = treated_firms.groupby("treat_year")["stkcd"].count()
    print("\n  批次处理公司分布:")
    print(batch_dist)

# ── 4. 行业分布（Top 10）─────────────────────────────────────────────
if "industry_sw" in df.columns:
    ind_dist = df.groupby("industry_sw")["stkcd"].nunique().sort_values(ascending=False).head(10)
    print("\n  行业分布 (Top 10 by 公司数):")
    print(ind_dist)

# ── 5. 年份分布 ───────────────────────────────────────────────────────
year_dist = df.groupby("year")["stkcd"].nunique()
print("\n  年份分布 (公司数):")
print(year_dist)

print(f"\n[STEP 4 COMPLETE] 描述性统计完成 | N = {len(df)} | 变量数 = {len(rows)}")
