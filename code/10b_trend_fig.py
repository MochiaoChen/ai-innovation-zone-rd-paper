"""
Figure 0 重制：Within-firm demeaned trends（消除composition bias）
同时画两个版本：
  (a) demeaned均值 + 95% CI ribbon
  (b) 以2018年标准化的累计变动（level change from 2018）
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
OUT   = os.path.join(BASE, "output/figures")

BLUE  = "#1565C0"
RED   = "#C62828"
LBLUE = "#BBDEFB"
LRED  = "#FFCDD2"
GRAY  = "#757575"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)

# ── within-firm demean ────────────────────────────────────────────────
for var in ["RD", "CAPEX"]:
    firm_mean = df.groupby("stkcd")[var].transform("mean")
    df[f"{var}_dm"] = df[var] - firm_mean

# ── 按 year × ever_treated 取均值 + bootstrap SE ─────────────────────
def group_stats(data, var, treat_val):
    sub = data[data["ever_treated"] == treat_val].dropna(subset=[var])
    stats = sub.groupby("year")[var].agg(["mean", "sem"]).reset_index()
    stats.columns = ["year", "mean", "se"]
    return stats

# ════════════════════════════════════════════════════════════════════
# Version A: Demeaned trends + CI ribbon
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

for ax, var, dm_var, ylabel, panel in [
    (axes[0], "RD",    "RD_dm",    "Within-Firm R&D Intensity (Demeaned)",      "Panel A: R&D Intensity"),
    (axes[1], "CAPEX", "CAPEX_dm", "Within-Firm Capital Expenditure (Demeaned)", "Panel B: Capital Expenditure"),
]:
    for treat_val, color, fc, label in [
        (1, RED,  LRED,  "Treated (AI Zone Cities)"),
        (0, BLUE, LBLUE, "Control (Non-AI Zone Cities)"),
    ]:
        st = group_stats(df, dm_var, treat_val)
        ax.fill_between(st["year"], st["mean"] - 1.96*st["se"],
                                    st["mean"] + 1.96*st["se"],
                        alpha=0.18, color=fc)
        ax.plot(st["year"], st["mean"], "o-", color=color,
                markersize=4.5, linewidth=1.9, label=label)

    # 政策窗口
    ax.axvspan(2018.6, 2022.4, alpha=0.07, color="orange", zorder=0)
    ax.axhline(0, color=GRAY, linewidth=0.7, linestyle="--", zorder=1)
    ax.text(2019.2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.001,
            "Policy\nrollout", fontsize=8, color=GRAY, va="top")

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(panel, fontsize=11, fontweight="bold")
    ax.set_xticks(range(2013, 2026, 2))
    ax.yaxis.grid(True, linestyle=":", alpha=0.35, zorder=0)
    ax.legend(fontsize=9, framealpha=0.6)

plt.suptitle(
    "Figure 1: Within-Firm Trends — Treated vs. Control Firms\n"
    r"$\it{Note}$: Each observation demeaned by firm mean to remove composition effects.",
    fontsize=11, fontweight="bold", y=1.03
)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "figure0a_trends_demeaned.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure0a_trends_demeaned.png"), bbox_inches="tight", dpi=180)
plt.close()
print("Figure 0a (demeaned trends) saved")

# ════════════════════════════════════════════════════════════════════
# Version B: 以2018为基准的累计变动（更直观）
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

for ax, var, ylabel, panel in [
    (axes[0], "RD",    "Change in R&D Intensity (pp, relative to 2018)",    "Panel A: R&D Intensity"),
    (axes[1], "CAPEX", "Change in CAPEX Intensity (pp, relative to 2018)",   "Panel B: Capital Expenditure"),
]:
    base_year = 2018
    for treat_val, color, fc, label in [
        (1, RED,  LRED,  "Treated (AI Zone Cities)"),
        (0, BLUE, LBLUE, "Control (Non-AI Zone Cities)"),
    ]:
        st = group_stats(df, var, treat_val)
        base_val = st.loc[st["year"] == base_year, "mean"].values
        if len(base_val) == 0:
            continue
        base_val = base_val[0]
        st["delta"] = (st["mean"] - base_val) * 100   # in percentage points

        ax.fill_between(st["year"],
                        st["delta"] - 1.96*st["se"]*100,
                        st["delta"] + 1.96*st["se"]*100,
                        alpha=0.18, color=fc)
        ax.plot(st["year"], st["delta"], "o-", color=color,
                markersize=4.5, linewidth=1.9, label=label)

    ax.axvspan(2018.6, 2022.4, alpha=0.07, color="orange", zorder=0)
    ax.axhline(0, color=GRAY, linewidth=0.7, linestyle="--", zorder=1)
    ax.axvline(base_year, color=GRAY, linewidth=0.9, linestyle=":", zorder=1)
    ax.text(base_year + 0.1, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0 else 0.1,
            "Base\nyear", fontsize=8, color=GRAY, va="top")
    ax.text(2019.2, ax.get_ylim()[1]*0.75 if ax.get_ylim()[1] > 0 else 0.05,
            "Policy\nrollout", fontsize=8, color=GRAY, va="top")

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(panel, fontsize=11, fontweight="bold")
    ax.set_xticks(range(2013, 2026, 2))
    ax.yaxis.grid(True, linestyle=":", alpha=0.35, zorder=0)
    ax.legend(fontsize=9, framealpha=0.6)

plt.suptitle(
    "Figure 1: Investment Trends Relative to 2018 Baseline — Treated vs. Control Firms\n"
    r"$\it{Note}$: Values expressed as percentage-point changes relative to 2018 group mean.",
    fontsize=11, fontweight="bold", y=1.03
)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "figure0b_trends_indexed.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure0b_trends_indexed.png"), bbox_inches="tight", dpi=180)
plt.close()
print("Figure 0b (indexed trends) saved")

print("[DONE]")
