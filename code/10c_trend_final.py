"""
趋势图最终版：quasi-balanced panel + 双版本
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "output/figures")
BLUE="#1565C0"; RED="#C62828"; LBLUE="#BBDEFB"; LRED="#FFCDD2"; GRAY="#757575"

plt.rcParams.update({"font.family":"serif","font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False})

df = pd.read_csv(os.path.join(BASE,"data/clean/analysis_data.csv"),
                 encoding="utf-8-sig", low_memory=False)

# quasi-balanced: >=8年观测
keep = df.groupby("stkcd")["year"].count()
keep = keep[keep >= 8].index
df = df[df["stkcd"].isin(keep)].copy()
print(f"Quasi-balanced panel: {df['stkcd'].nunique()} 公司, {len(df)} 观测")

# ── within-firm demean ───────────────────────────────────────────────
for v in ["RD","CAPEX"]:
    df[f"{v}_dm"] = df[v] - df.groupby("stkcd")[v].transform("mean")

def stats(data, var, treat_val):
    s = data[data["ever_treated"]==treat_val].dropna(subset=[var])
    g = s.groupby("year")[var].agg(["mean","sem","count"]).reset_index()
    g.columns = ["year","mean","se","n"]
    return g

# ══════════════════════════════════════════════════════
# Figure A: Demeaned within-firm trends（干净版）
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, var, dm, ylabel, panel in [
    (axes[0],"RD","RD_dm",
     "Within-Firm R&D Intensity\n(deviation from firm mean, ×100 pp)","Panel A: R&D Intensity"),
    (axes[1],"CAPEX","CAPEX_dm",
     "Within-Firm Capital Expenditure\n(deviation from firm mean, ×100 pp)","Panel B: Capital Expenditure"),
]:
    for tv, col, fc, lbl in [(1,RED,LRED,"Treated (AI Zone Cities)"),
                              (0,BLUE,LBLUE,"Control (Non-AI Zone Cities)")]:
        s = stats(df, dm, tv)
        y  = s["mean"]*100
        ci = s["se"]*100*1.96
        ax.fill_between(s["year"], y-ci, y+ci, alpha=0.18, color=fc)
        mk = "o" if tv==1 else "s"
        ax.plot(s["year"], y, f"{mk}-", color=col, markersize=4.5,
                linewidth=1.9, label=lbl, zorder=3)

    ax.axvspan(2018.6,2022.4,alpha=0.07,color="#FF9800",zorder=0,
               label="Policy rollout (2019–2022)")
    ax.axhline(0,color=GRAY,linewidth=0.7,linestyle="--",zorder=1)
    ax.set_xlim(2012.5,2025.5)
    ax.set_xticks(range(2013,2026,2))
    ax.set_xlabel("Year",fontsize=10)
    ax.set_ylabel(ylabel,fontsize=10)
    ax.set_title(panel,fontsize=12,fontweight="bold")
    ax.yaxis.grid(True,linestyle=":",alpha=0.35,zorder=0)
    if ax == axes[0]:
        ax.legend(fontsize=9,framealpha=0.6,loc="upper left")

plt.suptitle(
    "Figure 1: Within-Firm Investment Trends — Treated vs. Control Firms\n"
    r"$\it{Note}$: Quasi-balanced panel (firms observed ≥8 years). "
    "Each firm demeaned by its own sample mean. Shaded areas = 95% CI.",
    fontsize=10.5, fontweight="bold", y=1.04
)
plt.tight_layout()
fig.savefig(os.path.join(OUT,"figure1_trends_demeaned.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"figure1_trends_demeaned.png"),bbox_inches="tight",dpi=200)
plt.close(); print("Figure 1a saved")

# ══════════════════════════════════════════════════════
# Figure B: Raw levels，但只用balanced panel + 只显示2015-2025
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, var, ylabel, panel in [
    (axes[0],"RD","Mean R&D Intensity (R&D / Total Assets)","Panel A: R&D Intensity"),
    (axes[1],"CAPEX","Mean Capital Expenditure (CAPEX / Total Assets)","Panel B: Capital Expenditure"),
]:
    df15 = df[df["year"] >= 2015]
    for tv, col, fc, lbl in [(1,RED,LRED,"Treated (AI Zone Cities)"),
                              (0,BLUE,LBLUE,"Control (Non-AI Zone Cities)")]:
        s = stats(df15, var, tv)
        ci = s["se"]*1.96
        ax.fill_between(s["year"], s["mean"]-ci, s["mean"]+ci, alpha=0.18, color=fc)
        mk = "o" if tv==1 else "s"
        ax.plot(s["year"], s["mean"], f"{mk}-", color=col, markersize=4.5,
                linewidth=1.9, label=lbl, zorder=3)

    ax.axvspan(2018.6,2022.4,alpha=0.07,color="#FF9800",zorder=0)
    ax.set_xlim(2014.5,2025.5)
    ax.set_xticks(range(2015,2026,2))
    ax.set_xlabel("Year",fontsize=10)
    ax.set_ylabel(ylabel,fontsize=10)
    ax.set_title(panel,fontsize=12,fontweight="bold")
    ax.yaxis.grid(True,linestyle=":",alpha=0.35,zorder=0)
    ax.legend(fontsize=9,framealpha=0.6)

# 标注政策开始
for ax in axes:
    ax.annotate("Policy\nrollout\nbegins",
                xy=(2019,ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05),
                fontsize=8,color="#E65100",ha="center")

plt.suptitle(
    "Figure 1: Mean Investment Intensity — Treated vs. Control Firms (2015–2025)\n"
    r"$\it{Note}$: Quasi-balanced panel (firms observed ≥8 years). Shaded areas = 95% CI.",
    fontsize=10.5, fontweight="bold", y=1.04
)
plt.tight_layout()
fig.savefig(os.path.join(OUT,"figure1_trends_levels.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"figure1_trends_levels.png"),bbox_inches="tight",dpi=200)
plt.close(); print("Figure 1b saved")
print("[DONE]")
