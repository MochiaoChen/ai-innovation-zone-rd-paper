"""
Stacked event-time trend figure（消除一切composition bias）
对每个cohort，以treatment year为t=0对齐，
画treated firms 和 never-treated firms 的R&D均值随event-time的演变
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "output/figures")
BLUE="#1565C0"; RED="#C62828"; LBLUE="#BBDEFB"; LRED="#FFCDD2"; GRAY="#616161"

plt.rcParams.update({"font.family":"serif","font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False})

df = pd.read_csv(os.path.join(BASE,"data/clean/analysis_data.csv"),
                 encoding="utf-8-sig", low_memory=False)

cohorts = [2019, 2020, 2021, 2022]
never   = df[df["ever_treated"] == 0].copy()

WIN_PRE, WIN_POST = 4, 4   # event window

# ── 构建stacked dataset ─────────────────────────────────────────────
records = []
for g in cohorts:
    treated_g = df[df["treat_year"] == g].copy()
    treated_g["event_time"] = treated_g["year"] - g
    treated_g["group"] = "Treated"
    treated_g["cohort"] = g

    # never-treated：用同样的"假设处理年"g 对齐
    ctrl_g = never.copy()
    ctrl_g["event_time"] = ctrl_g["year"] - g
    ctrl_g["group"] = "Control"
    ctrl_g["cohort"] = g

    records.append(treated_g[["stkcd","year","event_time","cohort","group","RD","CAPEX"]])
    records.append(ctrl_g[["stkcd","year","event_time","cohort","group","RD","CAPEX"]])

stacked = pd.concat(records, ignore_index=True)
stacked = stacked[(stacked["event_time"] >= -WIN_PRE) & (stacked["event_time"] <= WIN_POST)]

# ── 汇总：按 event_time × group 取均值±SE ──────────────────────────
def agg_by_event(data, var):
    out = (data.dropna(subset=[var])
               .groupby(["event_time","group"])[var]
               .agg(["mean","sem","count"])
               .reset_index())
    out.columns = ["event_time","group","mean","se","n"]
    return out

# ══════════════════════════════════════════════════════════════════
# Figure A: Stacked event-time trends（主图）
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, var, ylabel, panel in [
    (axes[0],"RD","Mean R&D Intensity (R&D / Total Assets)","Panel A: R&D Intensity"),
    (axes[1],"CAPEX","Mean Capital Expenditure (CAPEX / Total Assets)","Panel B: Capital Expenditure"),
]:
    agg = agg_by_event(stacked, var)
    for grp, col, fc, mk, lbl in [
        ("Treated", RED,  LRED,  "o", "Treated Firms (AI Zone Cities)"),
        ("Control", BLUE, LBLUE, "s", "Control Firms (Never-Treated)"),
    ]:
        s = agg[agg["group"] == grp].sort_values("event_time")
        ci = s["se"] * 1.96
        ax.fill_between(s["event_time"], s["mean"]-ci, s["mean"]+ci, alpha=0.20, color=fc)
        ax.plot(s["event_time"], s["mean"], f"{mk}-", color=col,
                markersize=5, linewidth=2, label=lbl, zorder=3)

    # 标注
    ax.axvline(-0.5, color=GRAY, linewidth=0.9, linestyle=":", zorder=2)
    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle="--", zorder=1)
    ymin, ymax = ax.get_ylim()
    ax.text(-0.4, ymin + (ymax-ymin)*0.94, "Treatment\nyear",
            fontsize=8.5, color=GRAY, va="top")
    ax.fill_betweenx([ymin, ymax], -0.5, WIN_POST+0.5,
                     alpha=0.05, color="#FF9800", zorder=0)

    ax.set_xlabel("Event Time (Years Relative to AI Zone Designation)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(panel, fontsize=12, fontweight="bold")
    ax.set_xticks(range(-WIN_PRE, WIN_POST+1))
    ax.xaxis.set_ticklabels([f"t={t}" if t != 0 else "t=0\n(Designation)"
                             for t in range(-WIN_PRE, WIN_POST+1)], fontsize=8.5)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.legend(fontsize=9.5, framealpha=0.65, loc="upper left")

plt.suptitle(
    "Figure 1: Investment Trends Around AI Zone Designation — Stacked Event-Time Plot\n"
    r"$\it{Note}$: Each cohort (2019–2022) aligned at t=0 (designation year) and stacked. "
    "Never-treated firms serve as the control group. Shaded areas = 95% CI.",
    fontsize=10.5, fontweight="bold", y=1.04
)
plt.tight_layout()
fig.savefig(os.path.join(OUT,"figure1_stacked_trend.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT,"figure1_stacked_trend.png"), bbox_inches="tight", dpi=200)
plt.close(); print("Figure 1 (stacked) saved")

# ══════════════════════════════════════════════════════════════════
# Figure B: 四个cohort分开画（appendix用）
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
axes = axes.flatten()

for i, g in enumerate(cohorts):
    ax  = axes[i]
    sub = stacked[stacked["cohort"] == g]
    agg = agg_by_event(sub, "RD")
    for grp, col, fc, mk, lbl in [
        ("Treated", RED,  LRED,  "o", "Treated"),
        ("Control", BLUE, LBLUE, "s", "Control"),
    ]:
        s = agg[agg["group"] == grp].sort_values("event_time")
        ci = s["se"] * 1.96
        ax.fill_between(s["event_time"], s["mean"]-ci, s["mean"]+ci, alpha=0.20, color=fc)
        ax.plot(s["event_time"], s["mean"], f"{mk}-", color=col,
                markersize=4.5, linewidth=1.8, label=lbl, zorder=3)

    n_treat = df[df["treat_year"]==g]["stkcd"].nunique()
    ax.axvline(-0.5, color=GRAY, linewidth=0.9, linestyle=":")
    ax.set_title(f"Cohort {g}  (N={n_treat:,} treated firms)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Event Time", fontsize=9.5)
    ax.set_ylabel("Mean R&D Intensity", fontsize=9.5)
    ax.set_xticks(range(-WIN_PRE, WIN_POST+1))
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.legend(fontsize=9, framealpha=0.6)

plt.suptitle("Figure A3: R&D Trends by Treatment Cohort — Stacked Event-Time",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT,"figureA3_cohort_trends.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT,"figureA3_cohort_trends.png"), bbox_inches="tight", dpi=180)
plt.close(); print("Figure A3 (cohort trends) saved")
print("[DONE]")
