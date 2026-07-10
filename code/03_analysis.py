"""
Step 03: 全部实证分析
- Table 1: 描述性统计
- Table 2: 主回归（TWFE DID）
- Figure 1: 平行趋势检验（事件研究图）
- Table 3: 异质性分析（行业 AI 暴露度）
- Table 4: 稳健性检验
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN  = os.path.join(BASE, "data/clean")
TABLES = os.path.join(BASE, "output/tables")
FIGS   = os.path.join(BASE, "output/figures")

# ── 绘图风格
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ── 高 AI 暴露度行业（基于 Felten et al. 及文献惯例）
HIGH_AI_INDUSTRIES = [
    "计算机", "电子", "通信", "传媒", "软件",
    "医药生物", "机械设备", "汽车", "电气设备", "国防军工"
]

CONTROLS = ["Size", "Lev", "ROA", "Cash", "Age", "TobinQ"]
DEP_VARS = ["RD", "CAPEX"]  # RD 为主变量，CAPEX 为第二被解释变量

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def save_table(df, name, caption=""):
    """同时保存 CSV 和 LaTeX"""
    csv_path = os.path.join(TABLES, f"{name}.csv")
    tex_path = os.path.join(TABLES, f"{name}.tex")
    df.to_csv(csv_path, encoding="utf-8-sig")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        f.write(df.to_latex(escape=False))
        f.write("\\end{table}\n")
    print(f"     已保存：{name}.csv / .tex")


def run_twfe(data, dep_var, controls=CONTROLS, cluster_var="code",
             extra_fe=None, sample_mask=None):
    """
    双向固定效应回归（TWFE）
    使用 pyfixest；若安装失败则降级为 statsmodels demean 法
    """
    df = data.copy()
    if sample_mask is not None:
        df = df[sample_mask]
    # TobinQ 可能部分缺失：只用实际存在于 df 中的控制变量
    available_controls = [c for c in controls if c in df.columns]
    controls = available_controls
    df = df.dropna(subset=[dep_var] + controls + ["Treat", "code", "year"])

    fe_spec = "code + year" if extra_fe is None else f"code + {extra_fe}"

    try:
        import pyfixest as pf
        fml = f"{dep_var} ~ Treat + {' + '.join(controls)} | {fe_spec}"
        model = pf.feols(fml, data=df, vcov={f"CRV1": cluster_var})
        coef_df = model.coef()
        se_df   = model.se()
        pval_df = model.pvalue()
        n = len(df)
        r2 = model.r2()

        treat_coef = coef_df.get("Treat", np.nan)
        treat_se   = se_df.get("Treat", np.nan)
        treat_p    = pval_df.get("Treat", np.nan)

        return {
            "coef": treat_coef,
            "se": treat_se,
            "t": treat_coef / treat_se if treat_se else np.nan,
            "p": treat_p,
            "N": n,
            "r2": r2,
            "model": model,
        }

    except Exception as e:
        print(f"     pyfixest 报错（{e}），降级使用 within 估计...")
        return run_twfe_manual(df, dep_var, controls, cluster_var)


def run_twfe_manual(df, dep_var, controls, cluster_var):
    """手动 within 变换的 TWFE（pyfixest 不可用时的备选）"""
    from statsmodels.formula.api import ols
    from statsmodels.stats.sandwich_covariance import cov_cluster

    # 公司和年份 demeaning
    for col in [dep_var, "Treat"] + controls:
        df[f"_dm_{col}"] = (
            df[col]
            - df.groupby("code")[col].transform("mean")
            - df.groupby("year")[col].transform("mean")
            + df[col].mean()
        )

    dm_cols = [f"_dm_{c}" for c in ["Treat"] + controls]
    fml = f"_dm_{dep_var} ~ {' + '.join(dm_cols)} - 1"
    model = ols(fml, data=df.dropna()).fit()

    coef = model.params.get(f"_dm_Treat", np.nan)
    cov  = cov_cluster(model, df.dropna()[cluster_var])
    se   = np.sqrt(np.diag(cov))[list(model.params.index).index(f"_dm_Treat")] \
           if f"_dm_Treat" in model.params.index else np.nan
    p    = 2 * (1 - __import__("scipy").stats.t.cdf(abs(coef / se), df=model.df_resid)) \
           if se else np.nan

    return {
        "coef": coef, "se": se,
        "t": coef / se if se else np.nan,
        "p": p,
        "N": len(df.dropna()),
        "r2": model.rsquared,
        "model": model,
    }


def event_study_coefs(data, dep_var, controls=CONTROLS,
                      min_k=-3, max_k=4, cluster_var="code"):
    """
    事件研究法：生成各期系数和置信区间
    基准期为 k = -1（自动排除）
    """
    df = data.copy()
    # 只保留有 treat_year 的公司（处理组）+ never-treated 公司
    df_event = df[(df["rel_time"] != -999) | (df["treat_year"].isna())].copy()

    # 构建相对时间哑变量，bin 两端
    df_event["rel_time_bin"] = df_event["rel_time"].copy()
    df_event.loc[df_event["rel_time_bin"] < min_k, "rel_time_bin"] = min_k
    df_event.loc[df_event["rel_time_bin"] > max_k, "rel_time_bin"] = max_k
    df_event.loc[df_event["rel_time"] == -999, "rel_time_bin"] = -999  # never-treated

    periods = list(range(min_k, max_k + 1))
    periods.remove(-1)  # 基准期

    for k in periods:
        df_event[f"D_{k}"] = (df_event["rel_time_bin"] == k).astype(int)

    D_cols = [f"D_{k}" for k in periods]
    all_vars = D_cols + controls + ["code", "year", dep_var]
    df_event = df_event[all_vars].dropna()

    try:
        import pyfixest as pf
        fml = f"{dep_var} ~ {' + '.join(D_cols)} + {' + '.join(controls)} | code + year"
        model = pf.feols(fml, data=df_event, vcov={"CRV1": cluster_var})
        coef_df = model.coef()
        se_df   = model.se()

        results = []
        for k in periods:
            c = coef_df.get(f"D_{k}", np.nan)
            s = se_df.get(f"D_{k}", np.nan)
            results.append({"k": k, "coef": c, "se": s,
                            "ci_lo": c - 1.96 * s, "ci_hi": c + 1.96 * s})
        results.append({"k": -1, "coef": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
        return sorted(results, key=lambda x: x["k"])
    except Exception as e:
        print(f"     事件研究报错：{e}")
        return []


# ─────────────────────────────────────────────
# Table 1: 描述性统计
# ─────────────────────────────────────────────

def table1_descriptive(panel):
    print("[03] Table 1: 描述性统计...")
    vars_to_desc = ["RD", "CAPEX", "Treat", "Size", "Lev", "ROA", "Cash", "Age", "TobinQ"]
    rows = []
    for v in vars_to_desc:
        if v not in panel.columns:
            continue
        s = panel[v].dropna()
        rows.append({
            "变量": v,
            "N": int(len(s)),
            "均值": round(s.mean(), 4),
            "标准差": round(s.std(), 4),
            "P25": round(s.quantile(0.25), 4),
            "中位数": round(s.median(), 4),
            "P75": round(s.quantile(0.75), 4),
        })
    df_desc = pd.DataFrame(rows).set_index("变量")
    save_table(df_desc, "table1_descriptive", "描述性统计")

    # 附加：处理组 vs 对照组对比
    treat_firms = panel[panel["treat_year"].notna()]["code"].nunique()
    ctrl_firms  = panel[panel["treat_year"].isna()]["code"].nunique()
    print(f"     处理组：{treat_firms} 家 | 对照组：{ctrl_firms} 家")
    print(f"     总观测数：{len(panel)} | 公司数：{panel['code'].nunique()}")
    return df_desc


# ─────────────────────────────────────────────
# Table 2: 主回归
# ─────────────────────────────────────────────

DEP_VAR_LABELS = {
    "RD":    "R\\&D Intensity",
    "CAPEX": "Capital Expenditure",
}

def table2_main_regression(panel):
    print("[03] Table 2: 主回归...")
    rows = []
    res_map = {}

    for dep in DEP_VARS:
        if dep not in panel.columns or panel[dep].isna().all():
            print(f"     ⚠️  {dep} 全缺失，跳过")
            continue
        res = run_twfe(panel, dep)
        res_map[dep] = res
        row = {
            "被解释变量": DEP_VAR_LABELS.get(dep, dep),
            "Treat 系数": f"{res['coef']:.4f}{stars(res['p'])}",
            "t 统计量": f"({res['t']:.3f})",
            "N": res["N"],
            "R²": f"{res['r2']:.3f}",
            "Controls": "Yes",
            "公司 FE": "Yes",
            "年份 FE": "Yes",
            "聚类 SE": "公司层面",
        }
        rows.append(row)
        print(f"     {dep}: coef = {res['coef']:.4f}{stars(res['p'])}, "
              f"t = {res['t']:.3f}, N = {res['N']}")

    df_reg = pd.DataFrame(rows).set_index("被解释变量")
    save_table(df_reg, "table2_main_regression",
               "Main Results: AI Pilot Zone Policy and Corporate Investment")

    # 同时输出面向论文的横向格式 .tex
    _save_table2_wide_tex(res_map)
    return rows


def _save_table2_wide_tex(res_map):
    """输出横向多列格式的 Table 2（更接近经济学论文排版）"""
    cols = [d for d in DEP_VARS if d in res_map]
    if not cols:
        return

    col_headers = " & ".join(
        [f"({i+1}) {DEP_VAR_LABELS.get(d, d)}" for i, d in enumerate(cols)]
    )

    def fmt_row(label, key, res_map, cols, fmt_fn):
        vals = " & ".join(fmt_fn(res_map[d][key]) if d in res_map else "" for d in cols)
        return f"{label} & {vals} \\\\"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Main Regression Results: AI Pilot Zone Policy and Corporate Investment}",
        r"\label{tab:main}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        f" & {col_headers} \\\\",
        r"\midrule",
        # Treat 系数行
        "$Treat_{{it}}$ & " + " & ".join(
            f"{res_map[d]['coef']:.4f}{stars(res_map[d]['p'])}" if d in res_map else ""
            for d in cols
        ) + r" \\",
        # t 统计量行
        " & " + " & ".join(
            f"({res_map[d]['t']:.3f})" if d in res_map else ""
            for d in cols
        ) + r" \\",
        r"\midrule",
        "Controls & " + " & ".join(["Yes"] * len(cols)) + r" \\",
        "Firm FE & " + " & ".join(["Yes"] * len(cols)) + r" \\",
        "Year FE & " + " & ".join(["Yes"] * len(cols)) + r" \\",
        r"\midrule",
        "$N$ & " + " & ".join(
            f"{res_map[d]['N']:,}" if d in res_map else "" for d in cols
        ) + r" \\",
        "$R^2$ & " + " & ".join(
            f"{res_map[d]['r2']:.3f}" if d in res_map else "" for d in cols
        ) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item \textit{Notes}: $Treat_{it}=1$ if firm $i$'s city is an AI pilot zone in year $t$. "
        r"Controls: Size, Lev, ROA, Cash, Age, TobinQ. "
        r"Standard errors clustered at firm level. "
        r"$t$-statistics in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.10$.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    tex_path = os.path.join(TABLES, "table2_wide.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"     Table 2 宽格式已保存：{tex_path}")


# ─────────────────────────────────────────────
# Figure 1: 平行趋势（事件研究图）
# ─────────────────────────────────────────────

def figure1_event_study(panel):
    print("[03] Figure 1: 平行趋势检验...")

    dep = "RD"
    results = event_study_coefs(panel, dep)

    if not results:
        print("     跳过（无法获取事件研究系数）")
        return

    ks    = [r["k"] for r in results]
    coefs = [r["coef"] for r in results]
    ci_lo = [r["ci_lo"] for r in results]
    ci_hi = [r["ci_hi"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(ks, ci_lo, ci_hi, alpha=0.2, color="steelblue", label="95% CI")
    ax.plot(ks, coefs, "o-", color="steelblue", lw=1.8, ms=6, label="Coefficient")
    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.axvline(-0.5, color="red", lw=1.2, ls="--", alpha=0.7, label="Policy shock")

    # 标注 Pre/Post
    ax.text(-1.8, ax.get_ylim()[1] * 0.9, "Pre-treatment",
            ha="center", color="gray", fontsize=10)
    ax.text(2.0, ax.get_ylim()[1] * 0.9, "Post-treatment",
            ha="center", color="gray", fontsize=10)

    ax.set_xlabel("Years relative to AI pilot zone designation", fontsize=11)
    ax.set_ylabel("Effect on R&D Intensity (RD/Assets)", fontsize=11)
    ax.set_title("Figure 1: Event Study - Parallel Trends Test", fontsize=12, pad=10)
    ax.set_xticks(sorted(set(ks)))
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "figure1_event_study.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "figure1_event_study.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("     已保存：figure1_event_study.pdf / .png")


# ─────────────────────────────────────────────
# Table 3: 异质性分析
# ─────────────────────────────────────────────

def table3_heterogeneity(panel):
    print("[03] Table 3: 异质性分析...")

    # 标注高/低 AI 暴露度行业
    panel = panel.copy()
    panel["HighAI"] = panel["industry"].apply(
        lambda x: 1 if pd.notna(x) and any(kw in str(x) for kw in HIGH_AI_INDUSTRIES) else 0
    )

    rows = []
    for label, mask in [("高 AI 暴露度行业", panel["HighAI"] == 1),
                         ("低 AI 暴露度行业", panel["HighAI"] == 0),
                         ("全样本", pd.Series([True] * len(panel), index=panel.index))]:
        res = run_twfe(panel, "RD", sample_mask=mask)
        rows.append({
            "子样本": label,
            "Treat 系数": f"{res['coef']:.4f}{stars(res['p'])}",
            "t 统计量": f"({res['t']:.3f})",
            "N": res["N"],
            "R²": f"{res['r2']:.3f}",
        })
        print(f"     {label}: coef = {res['coef']:.4f}{stars(res['p'])}, N = {res['N']}")

    df_het = pd.DataFrame(rows).set_index("子样本")
    save_table(df_het, "table3_heterogeneity",
               "异质性分析：行业 AI 暴露度的调节效应")
    return df_het


# ─────────────────────────────────────────────
# Table 4: 稳健性检验
# ─────────────────────────────────────────────

def table4_robustness(panel):
    print("[03] Table 4: 稳健性检验...")
    rows = []

    # 列 (1): 基准回归（参照）
    res0 = run_twfe(panel, "RD")
    rows.append({
        "规格": "(1) 基准回归",
        "Treat 系数": f"{res0['coef']:.4f}{stars(res0['p'])}",
        "t 统计量": f"({res0['t']:.3f})",
        "N": res0["N"],
        "R²": f"{res0['r2']:.3f}",
        "备注": "公司 + 年份 FE",
    })

    # 列 (2): 省份 × 年份固定效应
    panel2 = panel.copy()
    panel2["province_year"] = panel2["city_clean"].str[:2].fillna("unknown") + \
                               "_" + panel2["year"].astype(str)
    res2 = run_twfe(panel2, "RD", extra_fe="province_year")
    rows.append({
        "规格": "(2) 省份×年份 FE",
        "Treat 系数": f"{res2['coef']:.4f}{stars(res2['p'])}",
        "t 统计量": f"({res2['t']:.3f})",
        "N": res2["N"],
        "R²": f"{res2['r2']:.3f}",
        "备注": "公司 + 省份×年份 FE",
    })

    # 列 (3): 排除直辖市
    muni = ["北京", "上海", "天津", "重庆"]
    mask3 = ~panel["city_clean"].isin(muni)
    res3 = run_twfe(panel, "RD", sample_mask=mask3)
    rows.append({
        "规格": "(3) 排除直辖市",
        "Treat 系数": f"{res3['coef']:.4f}{stars(res3['p'])}",
        "t 统计量": f"({res3['t']:.3f})",
        "N": res3["N"],
        "R²": f"{res3['r2']:.3f}",
        "备注": "剔除北京/上海/天津/重庆",
    })

    # 列 (4): 安慰剂检验（打乱处理时间 200 次，报告真实系数位置）
    print("[03]   运行安慰剂检验（200 次随机置换）...")
    placebo_coefs = []
    rng = np.random.default_rng(42)
    df_pl = panel.dropna(subset=["RD"] + CONTROLS + ["Treat", "code", "year"]).copy()
    city_list = df_pl["city_clean"].dropna().unique()

    for _ in range(200):
        df_pl["Treat_fake"] = df_pl.apply(
            lambda row: 1 if (row["city_clean"] in rng.choice(city_list, 10, replace=False)
                              and row["year"] >= rng.integers(2017, 2022)) else 0,
            axis=1
        )
        try:
            import pyfixest as pf
            fml = f"RD ~ Treat_fake + {' + '.join(CONTROLS)} | code + year"
            m = pf.feols(fml, data=df_pl, vcov={"CRV1": "code"})
            placebo_coefs.append(m.coef().get("Treat_fake", np.nan))
        except Exception:
            pass

    real_coef = res0["coef"]
    pct_rank = (np.array(placebo_coefs) < real_coef).mean()
    rows.append({
        "规格": "(4) 安慰剂检验",
        "Treat 系数": f"{real_coef:.4f}",
        "t 统计量": f"(真实系数超过 {pct_rank:.1%} 伪系数)",
        "N": res0["N"],
        "R²": "--",
        "备注": "200 次随机置换",
    })

    # 安慰剂分布图
    if placebo_coefs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(placebo_coefs, bins=40, color="lightgray", edgecolor="gray", alpha=0.8)
        ax.axvline(real_coef, color="red", lw=2, label=f"True coef = {real_coef:.4f}")
        ax.set_xlabel("Placebo coefficient", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Figure A1: Placebo Test Distribution", fontsize=12)
        ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGS, "figureA1_placebo.pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(FIGS, "figureA1_placebo.png"), dpi=300, bbox_inches="tight")
        plt.close()

    df_rob = pd.DataFrame(rows).set_index("规格")
    save_table(df_rob, "table4_robustness", "稳健性检验")
    return df_rob


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("[03] 开始实证分析")
    print("=" * 60)

    panel_path = os.path.join(CLEAN, "panel_data.csv")
    if not os.path.exists(panel_path):
        print("[03] ✗ 找不到 panel_data.csv，请先运行 02_clean_merge.py")
        return

    panel = pd.read_csv(panel_path, encoding="utf-8-sig", low_memory=False)
    print(f"[03] 读取面板数据：{len(panel)} 条，{panel['code'].nunique()} 家公司")

    # 过滤有 RD 数据的样本（RD 为主变量）
    panel_rd = panel.dropna(subset=["RD"])
    print(f"[03] 有 RD 数据的样本：{len(panel_rd)} 条，{panel_rd['code'].nunique()} 家")

    t1 = table1_descriptive(panel_rd)
    t2 = table2_main_regression(panel_rd)
    figure1_event_study(panel_rd)
    t3 = table3_heterogeneity(panel_rd)
    t4 = table4_robustness(panel_rd)

    # 结果汇总
    print("\n" + "=" * 60)
    print("[03] STEP 03 COMPLETE - 所有输出已保存至 output/")
    print("\n关键系数汇总：")
    for row in t2:
        print(f"  {row['被解释变量']}: Treat = {row['Treat 系数']}  {row['t 统计量']}")
    print("=" * 60)

    # 写 RESULTS_SUMMARY.md
    summary_path = os.path.join(BASE, "RESULTS_SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 实证结果汇总\n\n")
        f.write(f"样本：{len(panel_rd)} 条观测，{panel_rd['code'].nunique()} 家公司\n\n")
        f.write("## 主回归关键系数\n\n")
        for row in t2:
            f.write(f"- **{row['被解释变量']}**: Treat = {row['Treat 系数']} {row['t 统计量']}, N = {row['N']}\n")
        f.write("\n## 用户待核查事项\n\n")
        f.write("- [ ] Figure 1 中冲击前各期系数是否均不显著（平行趋势成立）\n")
        f.write("- [ ] Table 2 中 Treat 系数符号是否符合预期（正向）\n")
        f.write("- [ ] AI 试验区城市名单是否已与科技部官网核实\n")
        f.write("- [ ] RD 缺失率是否过高（若 > 50% 需重新考虑策略）\n")
    print(f"\n结果摘要已保存：{summary_path}")


if __name__ == "__main__":
    main()
