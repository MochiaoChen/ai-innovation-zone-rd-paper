"""
05b: Staggered DID — Sun-Abraham (SaturatedEventStudy) + Gardner DID2S
    用于解决 TWFE 在异质处理效应下的偏误问题
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.did import SaturatedEventStudy, did2s
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean")
OUT_T = os.path.join(BASE, "output/tables")
OUT_F = os.path.join(BASE, "output/figures")

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)
print(f"[05b] 样本: {len(df)} 观测, {df['stkcd'].nunique()} 公司")

# ── 准备 gname（从未处理=0，处理组=实际处理年份）──────────────────────
df["gname"] = df["treat_year"].fillna(0).astype(int)

# 只保留有 RD 的观测
df_rd = df.dropna(subset=["RD"]).copy()
print(f"[05b] RD 子样本: {len(df_rd)} 观测")

controls = "Size + Lev + ROA + Age + Cash + TobinQ + SOE"

# ══════════════════════════════════════════════════════════════════════
# 1. Sun-Abraham Saturated Event Study — RD
# ══════════════════════════════════════════════════════════════════════
print("\n[05b] Sun-Abraham 估计 (RD)...")
try:
    sa_rd = SaturatedEventStudy(
        data=df_rd,
        yname="RD",
        idname="stkcd",
        tname="year",
        gname="gname",
        cluster="stkcd",
        att=True,
    )
    sa_rd.fit()
    agg_rd = sa_rd.aggregate(agg="period")
    print(agg_rd[["estimate","std_error","t_stat","p_value"]].round(4))

    # ATT（aggregate average treatment effect on treated）
    att_row = agg_rd[agg_rd.index == "ATT"] if "ATT" in agg_rd.index else None
    if att_row is None:
        agg_all = sa_rd.aggregate(agg="att")
        print(f"  Sun-Abraham ATT: {agg_all['estimate'].values[0]:.4f} "
              f"(se={agg_all['std_error'].values[0]:.4f})")

    # Event study plot
    es_rd = sa_rd.aggregate(agg="event")
    fig, ax = plt.subplots(figsize=(8, 4))
    post = es_rd[es_rd.index >= 0]
    pre  = es_rd[es_rd.index < 0]
    for sub, color, label in [(pre, "#2196F3", "Pre-treatment"), (post, "#F44336", "Post-treatment")]:
        ax.errorbar(sub.index, sub["estimate"],
                    yerr=1.96 * sub["std_error"],
                    fmt="o", color=color, label=label, capsize=3, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Years Relative to AI Zone Designation")
    ax.set_ylabel("ATT on R&D Intensity")
    ax.set_title("Figure A1: Sun-Abraham Event Study — R&D Intensity")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_F, "figureA1_sa_event_rd.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_F, "figureA1_sa_event_rd.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [05b] Sun-Abraham event study 图已保存")

except Exception as e:
    print(f"  [05b] Sun-Abraham 失败: {e}")

# ══════════════════════════════════════════════════════════════════════
# 2. Gardner DID2S — 更稳健的聚合 ATT
# ══════════════════════════════════════════════════════════════════════
print("\n[05b] Gardner DID2S 估计...")
try:
    fit_did2s_rd = did2s(
        df_rd,
        yname="RD",
        first_stage="~ 0 | stkcd + year",
        second_stage="~ Treat",
        treatment="Treat",
        cluster="stkcd",
    )
    s = fit_did2s_rd.summary()
    coef = fit_did2s_rd.coef().get("Treat", np.nan)
    se   = fit_did2s_rd.se().get("Treat", np.nan)
    tval = coef / se if se else np.nan
    print(f"  DID2S RD: coef={coef:.4f}, se={se:.4f}, t={tval:.3f}")

    fit_did2s_cap = did2s(
        df.dropna(subset=["CAPEX"]),
        yname="CAPEX",
        first_stage="~ 0 | stkcd + year",
        second_stage="~ Treat",
        treatment="Treat",
        cluster="stkcd",
    )
    coef2 = fit_did2s_cap.coef().get("Treat", np.nan)
    se2   = fit_did2s_cap.se().get("Treat", np.nan)
    tval2 = coef2 / se2 if se2 else np.nan
    print(f"  DID2S CAPEX: coef={coef2:.4f}, se={se2:.4f}, t={tval2:.3f}")

except Exception as e:
    print(f"  [05b] DID2S 失败: {e}")

# ══════════════════════════════════════════════════════════════════════
# 3. Cohort-by-cohort ATT decomposition（手动 TWFE per cohort）
# ══════════════════════════════════════════════════════════════════════
print("\n[05b] Cohort-by-cohort ATT (清洁 DID)...")
cohorts = sorted([g for g in df_rd["gname"].unique() if g > 0])
cohort_results = []

for g in cohorts:
    # 干净对照：从未处理 + 尚未处理的公司
    sub = df_rd[(df_rd["gname"] == g) | (df_rd["gname"] == 0)].copy()
    sub["post"] = (sub["year"] >= g).astype(int)
    sub["treated_unit"] = (sub["gname"] == g).astype(int)
    sub["D"] = sub["post"] * sub["treated_unit"]
    n_treat = sub[sub["treated_unit"]==1]["stkcd"].nunique()
    try:
        fit = pf.feols(f"RD ~ D + {controls} | stkcd + year",
                       data=sub, vcov={"CRV1": "stkcd"})
        coef = fit.coef().get("D", np.nan)
        se   = fit.se().get("D", np.nan)
        pval = fit.pvalue().get("D", np.nan)
        cohort_results.append({"cohort": g, "N_firms": n_treat,
                                "coef": coef, "se": se,
                                "tstat": coef/se if se else np.nan,
                                "pval": pval})
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  Cohort {g}: β={coef:.4f} (t={coef/se:.3f}){stars}, N_firms={n_treat}")
    except Exception as e:
        print(f"  Cohort {g}: 失败 - {e}")

if cohort_results:
    df_cohort = pd.DataFrame(cohort_results)
    # 加权平均 ATT
    weights = df_cohort["N_firms"] / df_cohort["N_firms"].sum()
    w_att = (df_cohort["coef"] * weights).sum()
    print(f"\n  加权平均 ATT (by cohort size): {w_att:.4f}")
    df_cohort.to_csv(os.path.join(OUT_T, "tableA1_cohort_att.csv"), index=False, encoding="utf-8-sig")

    # Cohort ATT 图
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#F44336" if p < 0.1 else "#9E9E9E" for p in df_cohort["pval"]]
    ax.bar(df_cohort["cohort"].astype(str), df_cohort["coef"], color=colors,
           yerr=1.96 * df_cohort["se"], capsize=4, error_kw={"linewidth": 1.2})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Treatment Cohort")
    ax.set_ylabel("ATT on R&D Intensity")
    ax.set_title("Figure A2: Cohort-Specific ATT (Clean Comparison)")
    ax.annotate("Red = significant at 10%", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_F, "figureA2_cohort_att.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_F, "figureA2_cohort_att.png"), dpi=150, bbox_inches="tight")
    plt.close()

print("\n[STEP 05b COMPLETE] Staggered DID 估计完成")
