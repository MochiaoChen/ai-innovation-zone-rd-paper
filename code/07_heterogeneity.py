"""
07_heterogeneity.py — 异质性分析 Table 3
- 交叉项回归：Treat × HighAI
- 分组回归：高AI暴露 vs 低AI暴露子样本
"""
import os
import numpy as np
import pandas as pd

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN      = os.path.join(BASE, "data/clean")
OUT_TABLES = os.path.join(BASE, "output/tables")

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)
print(f"[07] 样本: {len(df)} 观测, {df['stkcd'].nunique()} 公司")

# 确定控制变量（TobinQ 缺失率高时跳过）
BASE_CONTROLS = ["Size", "Lev", "ROA", "Age", "Cash"]
EXTRA = ["TobinQ", "SOE"]
avail_controls = BASE_CONTROLS + [v for v in EXTRA if v in df.columns and df[v].notna().mean() > 0.4]
CONTROLS = " + ".join(avail_controls)

results_hetero = {}

try:
    import pyfixest as pf

    # ── 1. 交叉项回归（RD 为主）
    for y in ["RD", "CAPEX"]:
        if y not in df.columns or df[y].isna().all():
            continue
        fml = f"{y} ~ Treat + HighAI + Treat_HighAI + {CONTROLS} | firm_id + year"
        sub = df.dropna(subset=[y]).copy()
        try:
            m = pf.feols(fml=fml, data=sub, vcov={"CRV1": "firm_id"})
            results_hetero[f"{y}_interact"] = m
            coef_interact = m.coef().get("Treat_HighAI", np.nan)
            tstat_interact = m.tstat().get("Treat_HighAI", np.nan)
            pval_interact = m.pvalue().get("Treat_HighAI", np.nan)
            stars = "***" if pval_interact < 0.01 else "**" if pval_interact < 0.05 else "*" if pval_interact < 0.10 else ""
            print(f"  [{y}] Treat_HighAI: {coef_interact:.4f}{stars} (t={tstat_interact:.3f})")
        except Exception as e:
            print(f"  ⚠️  {y} 交叉项回归失败: {e}")

    # ── 2. 分组回归
    for y in ["RD", "CAPEX"]:
        if y not in df.columns or df[y].isna().all():
            continue
        fml_base = f"{y} ~ Treat + {CONTROLS} | firm_id + year"
        for grp, label in [(1, "HighAI=1"), (0, "HighAI=0")]:
            sub = df[df["HighAI"] == grp].dropna(subset=[y]).copy()
            try:
                m = pf.feols(fml=fml_base, data=sub, vcov={"CRV1": "firm_id"})
                key = f"{y}_{label}"
                results_hetero[key] = m
                coef = m.coef().get("Treat", np.nan)
                tstat = m.tstat().get("Treat", np.nan)
                pval = m.pvalue().get("Treat", np.nan)
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                print(f"  [{y} {label}] Treat: {coef:.4f}{stars} (t={tstat:.3f}, N={m._N})")
            except Exception as e:
                print(f"  ⚠️  {y} {label} 回归失败: {e}")

except ImportError:
    print("  pyfixest 不可用，跳过异质性分析")


def make_table3_tex(results, out_dir):
    if not results:
        return

    # 列顺序: RD_interact, CAPEX_interact, RD_HighAI=1, RD_HighAI=0, CAPEX_HighAI=1, CAPEX_HighAI=0
    col_keys = ["RD_interact", "CAPEX_interact",
                "RD_HighAI=1", "RD_HighAI=0", "CAPEX_HighAI=1", "CAPEX_HighAI=0"]
    col_labels = [
        "(1) R\\&D\\\\ \\small(Interaction)",
        "(2) CAPEX\\\\ \\small(Interaction)",
        "(3) R\\&D\\\\ \\small(High AI)",
        "(4) R\\&D\\\\ \\small(Low AI)",
        "(5) CAPEX\\\\ \\small(High AI)",
        "(6) CAPEX\\\\ \\small(Low AI)",
    ]
    avail_keys = [k for k in col_keys if k in results]
    avail_labels = [col_labels[col_keys.index(k)] for k in avail_keys]

    if not avail_keys:
        return

    rows_data = []
    for var, display in [("Treat", "$Treat_{it}$"), ("HighAI", "$HighAI_i$"),
                          ("Treat_HighAI", "$Treat_{it} \\times HighAI_i$")]:
        coef_row = [display]
        tstat_row = [""]
        for k in avail_keys:
            m = results[k]
            try:
                coef = m.coef().get(var, np.nan)
                tstat = m.tstat().get(var, np.nan)
                pval = m.pvalue().get(var, np.nan)
                if np.isnan(coef):
                    coef_row.append("")
                    tstat_row.append("")
                else:
                    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    coef_row.append(f"{coef:.4f}{stars}")
                    tstat_row.append(f"({tstat:.3f})")
            except Exception:
                coef_row.append("")
                tstat_row.append("")
        rows_data.append((coef_row, tstat_row))

    n_col = len(avail_keys)
    hdr = " & ".join(avail_labels)
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Heterogeneity Analysis: AI Exposure and Policy Effects}",
        r"\label{tab:hetero}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{l" + "c" * n_col + "}",
        r"\toprule",
        f" & {hdr} \\\\",
        r"\midrule",
    ]
    for coef_row, tstat_row in rows_data:
        tex.append(" & ".join(coef_row) + r" \\")
        tex.append(" & ".join(tstat_row) + r" \\")

    tex += [r"\midrule"]
    n_row = ["$N$"]
    for k in avail_keys:
        m = results[k]
        n = getattr(m, "_N", None)
        n_row.append(f"{int(n):,}" if n else "")
    tex.append(" & ".join(n_row) + r" \\")
    tex += [
        "Controls & " + " & ".join(["Yes"] * n_col) + r" \\",
        "Firm FE & " + " & ".join(["Yes"] * n_col) + r" \\",
        "Year FE & " + " & ".join(["Yes"] * n_col) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textit{Notes}: Columns (1)-(2) include the interaction term $Treat_{it} \times HighAI_i$. "
        r"Columns (3)-(6) are subgroup regressions. $HighAI_i$ equals one if the firm belongs to a "
        r"high-AI-exposure industry (computing, electronics, telecoms, media, biomedical, machinery, automotive). "
        r"Standard errors clustered at the firm level. "
        r"$t$-statistics in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.10$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    tex_path = os.path.join(out_dir, "table3_heterogeneity.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"  Table 3 已输出: {tex_path}")

    # CSV 版本
    csv_rows = []
    for coef_row, tstat_row in rows_data:
        csv_rows.append(coef_row)
        csv_rows.append(tstat_row)
    pd.DataFrame(csv_rows, columns=["Variable"] + avail_keys).to_csv(
        os.path.join(out_dir, "table3_heterogeneity.csv"), index=False, encoding="utf-8-sig"
    )


make_table3_tex(results_hetero, OUT_TABLES)
print(f"\n[STEP 7 COMPLETE] 异质性分析完成 | 模型数 = {len(results_hetero)}")
