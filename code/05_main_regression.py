"""
05_main_regression.py — 主回归 Table 2 (TWFE)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean")
OUT_TABLES = os.path.join(BASE, "output/tables")

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)

BASE_CONTROLS = ["Size", "Lev", "ROA", "Age", "Cash"]
EXTRA_CONTROLS = ["TobinQ", "SOE"]
avail_extra = [v for v in EXTRA_CONTROLS if v in df.columns and df[v].notna().mean() > 0.4]
CONTROLS = " + ".join(BASE_CONTROLS + avail_extra)
print(f"  控制变量: {CONTROLS}")
FE = "firm_id + year"
OUTCOMES = ["RD", "CAPEX", "RD_staff"]
OUTCOME_LABELS = {
    "RD": "R\\&D Intensity",
    "CAPEX": "Capital Expenditure",
    "RD_staff": "R\\&D Personnel",
}

results = {}

try:
    import pyfixest as pf

    for y in OUTCOMES:
        if y not in df.columns or df[y].isna().all():
            print(f"  ⚠️  {y} 全为缺失，跳过")
            continue
        sub = df.dropna(subset=[y] + CONTROLS.replace(" ", "").split("+")).copy()
        try:
            m = pf.feols(
                fml=f"{y} ~ Treat + {CONTROLS} | {FE}",
                data=sub,
                vcov={"CRV1": "firm_id"},
            )
            results[y] = m
            print(f"  {y}: Treat coef = {m.coef()['Treat']:.4f}, "
                  f"t = {m.tstat()['Treat']:.3f}, N = {m._N}")
        except Exception as e:
            print(f"  ⚠️  {y} 回归失败: {e}")

except ImportError:
    print("  pyfixest 不可用，降级使用 linearmodels")
    try:
        from linearmodels.panel import PanelOLS
        import statsmodels.formula.api as smf

        df_idx = df.set_index(["firm_id", "year"])
        for y in OUTCOMES:
            if y not in df_idx.columns or df_idx[y].isna().all():
                continue
            sub = df_idx.dropna(subset=[y]).copy()
            ctrl_cols = [c.strip() for c in CONTROLS.split("+") if c.strip() in sub.columns]
            exog_cols = ["Treat"] + ctrl_cols
            try:
                m = PanelOLS(
                    dependent=sub[y],
                    exog=sub[exog_cols],
                    entity_effects=True,
                    time_effects=True,
                ).fit(cov_type="clustered", cluster_entity=True)
                results[y] = m
                print(f"  {y}: Treat coef = {m.params['Treat']:.4f}, "
                      f"t = {m.tstats['Treat']:.3f}")
            except Exception as e:
                print(f"  ⚠️  {y} linearmodels 失败: {e}")
    except ImportError:
        print("  linearmodels 也不可用，跳过回归")


# ── 生成 Table 2 ──────────────────────────────────────────────────────
def extract_pyfixest(m, var="Treat"):
    coef = m.coef().get(var, np.nan)
    tstat = m.tstat().get(var, np.nan)
    pval = m.pvalue().get(var, np.nan)
    n = getattr(m, "_N", np.nan)
    r2 = getattr(m, "_r2_within", getattr(m, "_r2", np.nan))
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    return coef, tstat, stars, n, r2


def make_table2_tex(results, outcome_labels, controls_str, fe_str):
    cols = [y for y in OUTCOMES if y in results]
    if not cols:
        print("  ⚠️  无回归结果，跳过 Table 2 生成")
        return

    header_cols = " & ".join([f"({i+1}) {outcome_labels[y]}" for i, y in enumerate(cols)])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Main Regression Results: AI Pilot Zone Policy and Corporate Investment}",
        r"\label{tab:main}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        f" & {header_cols} \\\\",
        r"\midrule",
    ]

    # Treat 行
    coef_row = "$Treat_{it}$"
    tstat_row = ""
    n_row = "$N$"
    r2_row = "$R^2$ (within)"
    for y in cols:
        m = results[y]
        try:
            coef, tstat, stars, n, r2 = extract_pyfixest(m)
            coef_row += f" & {coef:.4f}{stars}"
            tstat_row += f" & ({tstat:.3f})"
            n_row += f" & {int(n):,}" if not np.isnan(n) else " & "
            r2_row += f" & {r2:.4f}" if not np.isnan(r2) else " & "
        except Exception:
            coef_row += " & "
            tstat_row += " & "
            n_row += " & "
            r2_row += " & "

    lines += [
        coef_row + r" \\",
        tstat_row + r" \\",
        r"\midrule",
        "Controls & " + " & ".join(["Yes"] * len(cols)) + r" \\",
        "Firm FE & " + " & ".join(["Yes"] * len(cols)) + r" \\",
        "Year FE & " + " & ".join(["Yes"] * len(cols)) + r" \\",
        r"\midrule",
        n_row + r" \\",
        r2_row + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textit{Notes}: Dependent variables are defined as: (1) R\&D expense scaled by total assets; "
        r"(2) capital expenditure scaled by total assets; (3) R\&D personnel as a share of total employees. "
        r"$Treat_{it}$ equals one if firm $i$'s city is designated as an AI pilot zone in or before year $t$. "
        r"Controls include firm size (log total assets), leverage, ROA, firm age (log), cash holdings, "
        r"Tobin's Q, and an SOE dummy. All regressions include firm and year fixed effects. "
        r"Standard errors are clustered at the firm level. "
        r"$t$-statistics in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.10$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    tex_path = os.path.join(OUT_TABLES, "table2_main.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Table 2 已输出: {tex_path}")


make_table2_tex(results, OUTCOME_LABELS, CONTROLS, FE)

# 保存关键系数供后续步骤调用
coef_summary = {}
for y, m in results.items():
    try:
        coef_summary[y] = {
            "coef": float(m.coef().get("Treat", float("nan"))),
            "tstat": float(m.tstat().get("Treat", float("nan"))),
            "pval": float(m.pvalue().get("Treat", float("nan"))),
            "N": int(getattr(m, "_N", 0)),
        }
    except Exception:
        pass
import json
with open(os.path.join(CLEAN, "main_results.json"), "w") as f:
    json.dump(coef_summary, f, indent=2)

print(f"\n[STEP 5 COMPLETE] 主回归完成 | 模型数 = {len(results)}")
