"""
06_event_study.py — 平行趋势检验（事件研究图 Figure 1）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean")
OUT_FIGS = os.path.join(BASE, "output/figures")

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)

BASE_CONTROLS = ["Size", "Lev", "ROA", "Age", "Cash"]
EXTRA_CONTROLS = ["TobinQ", "SOE"]
avail_extra = [v for v in EXTRA_CONTROLS if v in df.columns and df[v].notna().mean() > 0.4]
CONTROLS = " + ".join(BASE_CONTROLS + avail_extra)
EVENT_DUMMIES = ["D_m3", "D_m2", "D_0", "D_1", "D_2", "D_3", "D_4"]  # D_m1 为基准期（省略）
K_VALUES = [-3, -2, 0, 1, 2, 3, 4]

event_results = {}

try:
    import pyfixest as pf

    for y in ["RD", "CAPEX"]:
        if y not in df.columns or df[y].isna().all():
            continue
        available_dummies = [d for d in EVENT_DUMMIES if d in df.columns]
        if not available_dummies:
            print(f"  ⚠️  事件研究哑变量缺失，跳过 {y}")
            continue

        dummy_str = " + ".join(available_dummies)
        sub = df.dropna(subset=[y]).copy()
        try:
            m = pf.feols(
                fml=f"{y} ~ {dummy_str} + {CONTROLS} | firm_id + year",
                data=sub,
                vcov={"CRV1": "firm_id"},
            )
            coefs = m.coef()
            tstat = m.tstat()
            pvals = m.pvalue()

            event_results[y] = {
                "coefs": coefs,
                "tstat": tstat,
                "pvals": pvals,
            }
            print(f"  {y} 事件研究完成")
        except Exception as e:
            print(f"  ⚠️  {y} 事件研究失败: {e}")

except ImportError:
    print("  pyfixest 不可用，事件研究跳过")


def plot_event_study(event_res, outcome, label, out_dir):
    if outcome not in event_res:
        print(f"  ⚠️  {outcome} 无事件研究结果，跳过绘图")
        return

    res = event_res[outcome]
    coefs = res["coefs"]

    # 构建绘图数据（k=-1 基准期系数为 0）
    plot_data = []
    for k, dummy in zip(K_VALUES, EVENT_DUMMIES):
        if dummy in coefs.index:
            c = coefs[dummy]
            se = c / res["tstat"][dummy] if res["tstat"][dummy] != 0 else 0
            plot_data.append({"k": k, "coef": c, "se": se,
                               "ci_lo": c - 1.96 * se, "ci_hi": c + 1.96 * se})
        else:
            plot_data.append({"k": k, "coef": np.nan, "se": np.nan,
                               "ci_lo": np.nan, "ci_hi": np.nan})

    # 插入基准期
    plot_data.append({"k": -1, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
    plot_df = pd.DataFrame(plot_data).sort_values("k").reset_index(drop=True)

    # ── 绘图 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    # 样式
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })

    pre = plot_df[plot_df["k"] < 0]
    post = plot_df[plot_df["k"] >= 0]

    # 置信区间阴影
    ax.fill_between(pre["k"], pre["ci_lo"], pre["ci_hi"], alpha=0.15, color="steelblue")
    ax.fill_between(post["k"], post["ci_lo"], post["ci_hi"], alpha=0.15, color="firebrick")

    # 系数点线
    ax.plot(pre["k"], pre["coef"], "o-", color="steelblue", linewidth=1.5,
            markersize=5, label="Pre-treatment")
    ax.plot(post["k"], post["coef"], "s-", color="firebrick", linewidth=1.5,
            markersize=5, label="Post-treatment")

    # 基准期
    ax.scatter([-1], [0], color="black", zorder=5, s=40)

    # 零线
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    # 冲击时间垂直线
    ax.axvline(-0.5, color="black", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.text(-0.45, ax.get_ylim()[1] * 0.95 if not np.isnan(plot_df["ci_hi"]).all() else 0.01,
            "Policy\nOnset", fontsize=9, va="top")

    ax.set_xlabel("Years Relative to AI Pilot Zone Designation", fontsize=11)
    ax.set_ylabel(f"Coefficient on {label}", fontsize=11)
    ax.set_title(f"Event Study: Effect on {label}\n(Parallel Trends Test)", fontsize=12)
    ax.set_xticks(K_VALUES + [-1])
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")

    plt.tight_layout()

    png_path = os.path.join(out_dir, f"figure1_event_{outcome.lower()}.png")
    pdf_path = os.path.join(out_dir, f"figure1_event_{outcome.lower()}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Figure 1 ({outcome}) 已保存: {png_path}")


plot_event_study(event_results, "RD", "R\\&D Intensity", OUT_FIGS)
plot_event_study(event_results, "CAPEX", "Capital Expenditure", OUT_FIGS)

print(f"\n[STEP 6 COMPLETE] 事件研究完成 | 绘图数 = {len(event_results)}")
