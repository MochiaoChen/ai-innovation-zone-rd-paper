"""
10_figures.py — 论文图表（Publication quality）
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyfixest as pf

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean")
OUT   = os.path.join(BASE, "output/figures")

# ── 配色 & 样式 ─────────────────────────────────────────────────────────
BLUE  = "#1565C0"
RED   = "#C62828"
GRAY  = "#616161"
LBLUE = "#BBDEFB"
LRED  = "#FFCDD2"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)
df["gname"] = df["treat_year"].fillna(0).astype(int)
print(f"样本: {len(df)} 观测, {df['stkcd'].nunique()} 公司")

# ════════════════════════════════════════════════════════════════════
# Figure 0: 处理组 vs 对照组 R&D 均值趋势（introduction figure）
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, var, label, title in [
    (axes[0], "RD",    "R&D Intensity",       "Panel A: R&D Intensity"),
    (axes[1], "CAPEX", "Capital Expenditure",  "Panel B: Capital Expenditure"),
]:
    sub = df.dropna(subset=[var])
    tr  = sub[sub["ever_treated"] == 1].groupby("year")[var].mean()
    ct  = sub[sub["ever_treated"] == 0].groupby("year")[var].mean()
    ax.plot(tr.index, tr.values, color=RED,  marker="o", markersize=4, linewidth=1.8,
            label="Treated (AI Zone Cities)")
    ax.plot(ct.index, ct.values, color=BLUE, marker="s", markersize=4, linewidth=1.8,
            label="Control (Non-AI Zone Cities)")
    ax.axvspan(2018.5, 2022.5, alpha=0.08, color="orange", label="Policy rollout period")
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.5)
    ax.set_xticks(range(2013, 2026, 2))

plt.suptitle("Figure 1: Mean R&D Intensity and CAPEX — Treated vs. Control Firms",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "figure0_trends.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure0_trends.png"), bbox_inches="tight")
plt.close()
print("Figure 0 (trends) saved")

# ════════════════════════════════════════════════════════════════════
# Figure 1: Event Study RD — 重画，更好看
# ════════════════════════════════════════════════════════════════════
def run_event_study(data, yvar, label, max_pre=4, max_post=4):
    d = data.dropna(subset=[yvar, "rel_year"]).copy()
    d["rel_year_c"] = d["rel_year"].clip(-max_pre, max_post)
    # 只保留 ever_treated 或 never_treated
    d = d[d["ever_treated"].isin([0,1])].copy()
    dummies = [f"I(rel_year_c == {t})" for t in range(-max_pre, max_post+1) if t != -1]
    fml = f"{yvar} ~ {' + '.join(dummies)} + Size + Lev + ROA + Age + Cash + TobinQ + SOE | stkcd + year"
    try:
        fit = pf.feols(fml, data=d, vcov={"CRV1": "stkcd"})
        coefs = fit.coef(); ses = fit.se(); pvals = fit.pvalue()
        rows = []
        for t in range(-max_pre, max_post+1):
            if t == -1:
                rows.append({"t": t, "coef": 0.0, "se": 0.0, "pval": 1.0})
                continue
            key = f"I(rel_year_c == {t})"
            if key in coefs.index:
                rows.append({"t": t, "coef": coefs[key], "se": ses[key], "pval": pvals[key]})
        return pd.DataFrame(rows).set_index("t")
    except Exception as e:
        print(f"  event study failed for {yvar}: {e}")
        return None

# rel_year already in data?
if "rel_year" not in df.columns:
    df["rel_year"] = df["year"] - df["treat_year"]
    df.loc[df["ever_treated"] == 0, "rel_year"] = np.nan

es_rd  = run_event_study(df, "RD",    "R&D Intensity")
es_cap = run_event_study(df, "CAPEX", "CAPEX Intensity")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, es, title, ylab in [
    (axes[0], es_rd,  "Panel A: Effect on R\\&D Intensity",    "Coefficient (R&D / Assets)"),
    (axes[1], es_cap, "Panel B: Effect on Capital Expenditure", "Coefficient (CAPEX / Assets)"),
]:
    if es is None:
        continue
    pre  = es[es.index < 0]
    post = es[es.index >= 0]
    for sub, col, fc in [(pre, BLUE, LBLUE), (post, RED, LRED)]:
        ax.fill_between(sub.index, sub["coef"] - 1.96*sub["se"],
                                   sub["coef"] + 1.96*sub["se"], alpha=0.2, color=fc)
        ax.plot(sub.index, sub["coef"], "o-", color=col, markersize=5, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(-0.5, color=GRAY, linewidth=0.8, linestyle=":")
    ax.text(-0.5, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0 else 0, " Policy\n onset",
            fontsize=8, color=GRAY, va="top")
    ax.set_xlabel("Years Relative to AI Zone Designation", fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(-4, 5))
    ax.xaxis.grid(True, linestyle=":", alpha=0.4)

pre_patch  = mpatches.Patch(color=BLUE, alpha=0.7, label="Pre-treatment")
post_patch = mpatches.Patch(color=RED,  alpha=0.7, label="Post-treatment")
fig.legend(handles=[pre_patch, post_patch], loc="lower center",
           ncol=2, framealpha=0.5, fontsize=10, bbox_to_anchor=(0.5, -0.04))
plt.suptitle("Figure 2: Event Study — Dynamic Treatment Effects",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "figure1_event_study.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure1_event_study.png"), bbox_inches="tight")
plt.close()
print("Figure 1 (event study) saved")

# ════════════════════════════════════════════════════════════════════
# Figure 2: 按 HighAI 分组的 Event Study（heterogeneity）
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, hi, col, fc, lbl in [
    (axes[0], 1, RED,  LRED,  "High AI-Exposure Industries"),
    (axes[1], 0, BLUE, LBLUE, "Low AI-Exposure Industries"),
]:
    sub_df = df[df["HighAI"] == hi].copy()
    es = run_event_study(sub_df, "RD", "RD")
    if es is None:
        continue
    ax.fill_between(es.index, es["coef"] - 1.96*es["se"],
                               es["coef"] + 1.96*es["se"], alpha=0.2, color=fc)
    ax.plot(es.index, es["coef"], "o-", color=col, markersize=5, linewidth=1.8, label=lbl)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(-0.5, color=GRAY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Years Relative to AI Zone Designation", fontsize=10)
    ax.set_ylabel("Coefficient on R&D Intensity", fontsize=10)
    ax.set_title(f"Panel {'A' if hi else 'B'}: {lbl}", fontsize=11, fontweight="bold")
    ax.set_xticks(range(-4, 5))
    ax.xaxis.grid(True, linestyle=":", alpha=0.4)

plt.suptitle("Figure 3: Heterogeneous Effects — High vs. Low AI-Exposure Industries",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "figure3_heterogeneity_event.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure3_heterogeneity_event.png"), bbox_inches="tight")
plt.close()
print("Figure 3 (heterogeneity event study) saved")

# ════════════════════════════════════════════════════════════════════
# Figure 3: Cohort ATT — 重绘（更好看的横版）
# ════════════════════════════════════════════════════════════════════
cohort_df = pd.read_csv(os.path.join(BASE, "output/tables/tableA1_cohort_att.csv"), encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 4))
colors = [RED if p < 0.05 else (GRAY if p < 0.1 else "#BDBDBD") for p in cohort_df["pval"]]
labels = [str(int(c)) for c in cohort_df["cohort"]]
n_firms = cohort_df["N_firms"].values

bars = ax.bar(labels, cohort_df["coef"], color=colors, width=0.5,
              zorder=3, edgecolor="white", linewidth=0.5)
ax.errorbar(labels, cohort_df["coef"], yerr=1.96 * cohort_df["se"],
            fmt="none", color="#212121", capsize=5, linewidth=1.4, zorder=4)
ax.axhline(0, color="black", linewidth=0.8, zorder=2)

# 标注公司数
for bar, n in zip(bars, n_firms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00025,
            f"N={n:,}", ha="center", va="bottom", fontsize=9, color="#424242")

# 加权平均 ATT 线
w = cohort_df["N_firms"] / cohort_df["N_firms"].sum()
w_att = (cohort_df["coef"] * w).sum()
ax.axhline(w_att, color="#E65100", linestyle="--", linewidth=1.3,
           label=f"Weighted avg ATT = {w_att:.4f}")

ax.set_xlabel("Treatment Cohort (Year of AI Zone Designation)", fontsize=11)
ax.set_ylabel("ATT on R&D Intensity", fontsize=11)
ax.set_title("Figure 4: Cohort-Specific Average Treatment Effects\n(Clean Comparison Groups)",
             fontsize=11, fontweight="bold")
ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)

red_p  = mpatches.Patch(color=RED,  label="Significant at 5%")
gray_p = mpatches.Patch(color=GRAY, label="Significant at 10%")
na_p   = mpatches.Patch(color="#BDBDBD", label="Not significant")
ax.legend(handles=[red_p, gray_p, na_p,
          plt.Line2D([0],[0], color="#E65100", linestyle="--", linewidth=1.3,
                     label=f"Weighted avg = {w_att:.4f}")],
          fontsize=9, framealpha=0.6, loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "figure4_cohort_att.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure4_cohort_att.png"), bbox_inches="tight")
plt.close()
print("Figure 4 (cohort ATT) saved")

# ════════════════════════════════════════════════════════════════════
# Figure 4: Placebo — 重绘（更漂亮）
# ════════════════════════════════════════════════════════════════════
# 重新跑500次安慰剂，用正确样本
print("安慰剂检验（重跑）...")
df_rd = df.dropna(subset=["RD"]).copy()
true_coef = 0.0004  # from main regression

np.random.seed(42)
placebo_coefs = []
treated_firms = df_rd[df_rd["ever_treated"] == 1]["stkcd"].unique()
n_treated = len(treated_firms)

for i in range(500):
    fake_treated = np.random.choice(df_rd["stkcd"].unique(), size=n_treated, replace=False)
    d2 = df_rd.copy()
    d2["Treat_fake"] = 0
    d2.loc[d2["stkcd"].isin(fake_treated), "Treat_fake"] = d2.loc[d2["stkcd"].isin(fake_treated), "Treat"]
    try:
        fit = pf.feols("RD ~ Treat_fake + Size + Lev + ROA + Age + Cash + TobinQ + SOE | stkcd + year",
                       data=d2, vcov={"CRV1": "stkcd"})
        placebo_coefs.append(fit.coef().get("Treat_fake", np.nan))
    except:
        pass
    if (i+1) % 100 == 0:
        print(f"  {i+1}/500", flush=True)

placebo_coefs = [c for c in placebo_coefs if not np.isnan(c)]
pct_rank = np.mean(np.array(placebo_coefs) < true_coef) * 100

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(placebo_coefs, bins=40, color=LBLUE, edgecolor="white", linewidth=0.5,
        alpha=0.9, label=f"Permuted coefficients (N={len(placebo_coefs)})")
ax.axvline(true_coef, color=RED, linewidth=2, linestyle="--",
           label=f"True coefficient = {true_coef:.4f}\n(Percentile rank: {pct_rank:.1f}%)")
ax.axvline(0, color="#212121", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"Coefficient on $Treat_{it}$ (Permuted Treatment)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Figure 5: Randomization Inference — Placebo Test\n(500 Random Treatment Assignments)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.7)
ax.xaxis.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "figure5_placebo.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure5_placebo.png"), bbox_inches="tight")
plt.close()
print(f"Figure 5 (placebo) saved | rank = {pct_rank:.1f}%")

# ════════════════════════════════════════════════════════════════════
# Figure 5: 累积效应（cumulative ATT by rel_year）
# ════════════════════════════════════════════════════════════════════
if es_rd is not None:
    post_es = es_rd[es_rd.index >= 0].copy()
    post_es["cum_coef"] = post_es["coef"].cumsum()
    # delta method近似SE（简化为根号n倍）
    post_es["cum_se"] = post_es["se"] * np.sqrt(np.arange(1, len(post_es)+1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(post_es.index,
                    post_es["cum_coef"] - 1.96*post_es["cum_se"],
                    post_es["cum_coef"] + 1.96*post_es["cum_se"],
                    alpha=0.2, color=LRED)
    ax.plot(post_es.index, post_es["cum_coef"], "o-", color=RED, markersize=6, linewidth=2,
            label="Cumulative ATT")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Years Since AI Zone Designation", fontsize=11)
    ax.set_ylabel("Cumulative Effect on R&D Intensity", fontsize=11)
    ax.set_title("Figure 6: Cumulative Treatment Effect on R&D Intensity",
                 fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "figure6_cumulative.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "figure6_cumulative.png"), bbox_inches="tight")
    plt.close()
    print("Figure 6 (cumulative) saved")

print("\n[STEP 10 COMPLETE] 所有图表生成完毕")
print("输出目录:", OUT)
