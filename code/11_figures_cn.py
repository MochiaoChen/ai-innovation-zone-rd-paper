"""
中文学术图表 — 楷体 + Times New Roman
无顶部图号标题、无底部注释（均由 LaTeX caption/正文承担）
"""
import os, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import pyfixest as pf

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean")
OUT   = os.path.join(BASE, "output/figures")

KAITI_PATH = "/mnt/c/Windows/Fonts/simkai.ttf"
TNR_PATH   = "/mnt/c/Windows/Fonts/times.ttf"

prop_kai       = fm.FontProperties(fname=KAITI_PATH, size=11)
prop_kai_sm    = fm.FontProperties(fname=KAITI_PATH, size=9.5)
prop_kai_title = fm.FontProperties(fname=KAITI_PATH, size=12)

fm.fontManager.addfont(KAITI_PATH)
if os.path.exists(TNR_PATH):
    fm.fontManager.addfont(TNR_PATH)

plt.rcParams.update({
    "font.family":       ["Times New Roman", "KaiTi", "SimKai"],
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi":        150,
})

BLUE="#1565C0"; RED="#C62828"; LBLUE="#BBDEFB"; LRED="#FFCDD2"; GRAY="#616161"

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)
df["gname"] = df["treat_year"].fillna(0).astype(int)
if "rel_year" not in df.columns:
    df["rel_year"] = df["year"] - df["treat_year"]
    df.loc[df["ever_treated"]==0, "rel_year"] = np.nan
print(f"样本：{len(df)} 观测，{df['stkcd'].nunique()} 公司")

# ════════════════════════════════════════════════════════════════════
# 图1：Stacked Event-Time 趋势图
# ════════════════════════════════════════════════════════════════════
cohorts = [2019, 2020, 2021, 2022]
never   = df[df["ever_treated"]==0].copy()
WIN = 4

records = []
for g in cohorts:
    tg = df[df["treat_year"]==g].copy()
    tg["event_time"] = tg["year"] - g
    tg["group"] = "处理组（AI试验区城市）"
    cg = never.copy()
    cg["event_time"] = cg["year"] - g
    cg["group"] = "对照组（非AI试验区城市）"
    for d in [tg, cg]:
        records.append(d[["stkcd","event_time","group","RD","CAPEX"]])

stacked = pd.concat(records, ignore_index=True)
stacked = stacked[(stacked["event_time"]>=-WIN) & (stacked["event_time"]<=WIN)]

def agg(data, var):
    return (data.dropna(subset=[var])
               .groupby(["event_time","group"])[var]
               .agg(["mean","sem"]).reset_index()
               .rename(columns={"mean":"mu","sem":"se"}))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, var, ylabel, panel in [
    (axes[0], "RD",    "研发强度（研发费用/总资产）", "（一）研发强度"),
    (axes[1], "CAPEX", "资本支出强度（资本支出/总资产）", "（二）资本支出强度"),
]:
    a = agg(stacked, var)
    for grp, col, fc, mk in [
        ("处理组（AI试验区城市）", RED,  LRED,  "o"),
        ("对照组（非AI试验区城市）", BLUE, LBLUE, "s"),
    ]:
        s = a[a["group"]==grp].sort_values("event_time")
        ax.fill_between(s["event_time"], s["mu"]-1.96*s["se"],
                        s["mu"]+1.96*s["se"], alpha=0.18, color=fc)
        ax.plot(s["event_time"], s["mu"], f"{mk}-", color=col,
                markersize=5, linewidth=1.8, label=grp)
    ax.axvline(-0.5, color=GRAY, linewidth=0.9, linestyle=":", zorder=2)
    ax.fill_betweenx(ax.get_ylim() if ax.get_ylim()[0]!=0 else [0,1],
                     -0.5, WIN+0.5, alpha=0.05, color="#FF9800", zorder=0)
    ax.set_xlabel("相对处理年份（年）", fontproperties=prop_kai, labelpad=6)
    ax.set_ylabel(ylabel, fontproperties=prop_kai, labelpad=6)
    ax.set_title(panel, fontproperties=prop_kai_title, fontweight="bold", pad=8)
    ax.set_xticks(range(-WIN, WIN+1))
    ax.xaxis.grid(True, linestyle=":", alpha=0.35)
    legend = ax.legend(fontsize=0, framealpha=0.6, loc="upper left")
    for text, grp in zip(legend.get_texts(),
                         ["处理组（AI试验区城市）","对照组（非AI试验区城市）"]):
        text.set_text(grp)
        text.set_fontproperties(prop_kai_sm)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_趋势图_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_趋势图_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图1 保存完成")

# ════════════════════════════════════════════════════════════════════
# 图2：事件研究图（动态处理效应）
# ════════════════════════════════════════════════════════════════════
ctrl = "Size + Lev + ROA + Age + Cash + TobinQ + SOE"

def event_study(data, yvar, max_pre=4, max_post=4):
    d = data.dropna(subset=[yvar]).copy()
    periods = [t for t in range(-max_pre, max_post+1) if t != -1]
    for t in periods:
        col = f"evt_{'m'+str(abs(t)) if t<0 else 'p'+str(t)}"
        d[col] = ((d["rel_year"] == t) & d["rel_year"].notna()).astype(int)
    dummy_cols = [f"evt_{'m'+str(abs(t)) if t<0 else 'p'+str(t)}" for t in periods]
    fml = f"{yvar} ~ {' + '.join(dummy_cols)} + {ctrl} | stkcd + year"
    try:
        fit = pf.feols(fml, data=d, vcov={"CRV1":"stkcd"})
        rows = []
        for t in range(-max_pre, max_post+1):
            if t == -1:
                rows.append({"t": t, "coef": 0.0, "se": 0.0}); continue
            col = f"evt_{'m'+str(abs(t)) if t<0 else 'p'+str(t)}"
            if col in fit.coef().index:
                rows.append({"t": t, "coef": fit.coef()[col], "se": fit.se()[col]})
        return pd.DataFrame(rows).set_index("t")
    except Exception as e:
        print(f"  event_study error ({yvar}): {e}")
        return None

es_rd  = event_study(df, "RD")
es_cap = event_study(df, "CAPEX")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, es, ylabel, panel in [
    (axes[0], es_rd,  "回归系数（研发强度）",   "（一）研发强度"),
    (axes[1], es_cap, "回归系数（资本支出强度）","（二）资本支出强度"),
]:
    if es is None: continue
    for sub, col, fc in [(es[es.index<0], BLUE, LBLUE),(es[es.index>=0], RED, LRED)]:
        ax.fill_between(sub.index, sub["coef"]-1.96*sub["se"],
                        sub["coef"]+1.96*sub["se"], alpha=0.20, color=fc)
        ax.plot(sub.index, sub["coef"], "o-", color=col, markersize=5, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(-0.5, color=GRAY, linewidth=0.9, linestyle=":")
    ax.set_xlabel("相对处理年份（年）", fontproperties=prop_kai, labelpad=6)
    ax.set_ylabel(ylabel, fontproperties=prop_kai, labelpad=6)
    ax.set_title(panel, fontproperties=prop_kai_title, fontweight="bold", pad=8)
    ax.set_xticks(range(-4, 5))
    ax.xaxis.grid(True, linestyle=":", alpha=0.35)

pre_p  = mpatches.Patch(color=BLUE, alpha=0.7, label="政策前")
post_p = mpatches.Patch(color=RED,  alpha=0.7, label="政策后")
leg = fig.legend(handles=[pre_p, post_p], loc="lower center", ncol=2,
                 framealpha=0.5, bbox_to_anchor=(0.5, -0.06))
for t in leg.get_texts():
    t.set_fontproperties(prop_kai_sm)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_事件研究_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_事件研究_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图2 保存完成")

# ════════════════════════════════════════════════════════════════════
# 图3：Cohort ATT 柱状图
# ════════════════════════════════════════════════════════════════════
cohort_df = pd.read_csv(os.path.join(BASE,"output/tables/tableA1_cohort_att.csv"),
                        encoding="utf-8-sig")
w = cohort_df["N_firms"] / cohort_df["N_firms"].sum()
w_att = (cohort_df["coef"] * w).sum()

fig, ax = plt.subplots(figsize=(8, 4.6))
colors = [RED if p<0.05 else (GRAY if p<0.1 else "#BDBDBD") for p in cohort_df["pval"]]
labels_cn = [f"{int(c)}年批次" for c in cohort_df["cohort"]]
bars = ax.bar(labels_cn, cohort_df["coef"]*100, color=colors, width=0.5,
              zorder=3, edgecolor="white", linewidth=0.5)
ax.errorbar(labels_cn, cohort_df["coef"]*100, yerr=1.96*cohort_df["se"]*100,
            fmt="none", color="#212121", capsize=6, linewidth=1.4, zorder=4)
ax.axhline(0, color="black", linewidth=0.8, zorder=2)
ax.axhline(w_att*100, color="#E65100", linestyle="--", linewidth=1.4)

for bar, row in zip(bars, cohort_df.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + 0.003 if bar.get_height()>=0 else bar.get_height()-0.006,
            f"N={row.N_firms:,}", ha="center", va="bottom",
            fontproperties=prop_kai_sm, color="#424242")

ax.set_ylabel("研发强度的平均处理效应（百分点）", fontproperties=prop_kai, labelpad=6)
ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
for t in ax.get_xticklabels():
    t.set_fontproperties(prop_kai_sm)

red_p  = mpatches.Patch(color=RED,       label="5%水平显著")
gray_p = mpatches.Patch(color=GRAY,      label="10%水平显著")
na_p   = mpatches.Patch(color="#BDBDBD", label="不显著")
line_p = plt.Line2D([0],[0], color="#E65100", linestyle="--", linewidth=1.4,
                    label=f"加权平均 = {w_att*100:.3f}pp")
leg = ax.legend(handles=[red_p, gray_p, na_p, line_p],
                fontsize=0, framealpha=0.6, loc="upper right")
for t, lbl in zip(leg.get_texts(),
                  ["5%水平显著","10%水平显著","不显著",
                   f"加权平均 = {w_att*100:.3f}pp"]):
    t.set_text(lbl); t.set_fontproperties(prop_kai_sm)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_cohort_ATT_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_cohort_ATT_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图3 保存完成")

# ════════════════════════════════════════════════════════════════════
# 图4：异质性——高/低AI行业事件研究
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, hi, col, fc, panel in [
    (axes[0], 1, RED,  LRED,  "（一）高AI暴露行业"),
    (axes[1], 0, BLUE, LBLUE, "（二）低AI暴露行业"),
]:
    sub_df = df[df["HighAI"]==hi].copy()
    es = event_study(sub_df, "RD")
    if es is None: continue
    ax.fill_between(es.index, es["coef"]-1.96*es["se"],
                    es["coef"]+1.96*es["se"], alpha=0.20, color=fc)
    ax.plot(es.index, es["coef"], "o-", color=col, markersize=5, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(-0.5, color=GRAY, linewidth=0.9, linestyle=":")
    ax.set_xlabel("相对处理年份（年）", fontproperties=prop_kai, labelpad=6)
    ax.set_ylabel("回归系数（研发强度）", fontproperties=prop_kai, labelpad=6)
    ax.set_title(panel, fontproperties=prop_kai_title, fontweight="bold", pad=8)
    ax.set_xticks(range(-4,5))
    ax.xaxis.grid(True, linestyle=":", alpha=0.35)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_异质性_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_异质性_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图4 保存完成")

# ════════════════════════════════════════════════════════════════════
# 图5：安慰剂检验
# ════════════════════════════════════════════════════════════════════
print("安慰剂检验（300次）...")
df_rd = df.dropna(subset=["RD"]).copy()
true_coef = 0.0004
np.random.seed(42)
placebo = []
all_firms = df_rd["stkcd"].unique()
n_treat = df_rd[df_rd["ever_treated"]==1]["stkcd"].nunique()
for i in range(300):
    fake = np.random.choice(all_firms, size=n_treat, replace=False)
    d2 = df_rd.copy()
    d2["Tf"] = d2["Treat"].where(d2["stkcd"].isin(fake), 0)
    try:
        fit = pf.feols(f"RD ~ Tf + {ctrl} | stkcd + year", data=d2, vcov={"CRV1":"stkcd"})
        placebo.append(fit.coef().get("Tf", np.nan))
    except: pass
    if (i+1)%100==0: print(f"  {i+1}/300")
placebo = [c for c in placebo if not np.isnan(c)]
pct = np.mean(np.array(placebo) < true_coef)*100

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(placebo, bins=35, color=LBLUE, edgecolor="white", linewidth=0.4, alpha=0.9,
        label=f"置换系数分布（N={len(placebo)}次）")
ax.axvline(true_coef, color=RED, linewidth=2, linestyle="--",
           label=f"真实系数 = {true_coef:.4f}（高于{pct:.1f}%置换系数）")
ax.axvline(0, color=GRAY, linewidth=0.8, linestyle=":")
ax.set_xlabel("处理变量系数（随机置换）", fontproperties=prop_kai, labelpad=6)
ax.set_ylabel("频次", fontproperties=prop_kai, labelpad=6)
ax.xaxis.grid(True, linestyle=":", alpha=0.3)
leg = ax.legend(fontsize=0, framealpha=0.7)
for t, lbl in zip(leg.get_texts(),
                  [f"置换系数分布（N={len(placebo)}次）",
                   f"真实系数 = {true_coef:.4f}（高于{pct:.1f}%置换系数）"]):
    t.set_text(lbl); t.set_fontproperties(prop_kai_sm)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_安慰剂_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_安慰剂_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图5 保存完成")

print("\n[全部完成] 输出目录:", OUT)
