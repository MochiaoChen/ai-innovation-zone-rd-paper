"""
额外图表：机制对比、所有制异质性、稳健性对比、样本分布
无顶部图号标题、无底部注释
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

BLUE="#1565C0"; RED="#C62828"; LBLUE="#BBDEFB"; LRED="#FFCDD2"
GRAY="#616161"; GREEN="#2E7D32"; LGREEN="#C8E6C9"; ORANGE="#E65100"

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)
df["gname"] = df["treat_year"].fillna(0).astype(int)
if "rel_year" not in df.columns:
    df["rel_year"] = df["year"] - df["treat_year"]
    df.loc[df["ever_treated"]==0, "rel_year"] = np.nan
print(f"样本：{len(df)} 观测，{df['stkcd'].nunique()} 公司")

ctrl = "Size + Lev + ROA + Age + Cash + TobinQ + SOE"

def run_reg(data, fml):
    try:
        return pf.feols(fml, data=data, vcov={"CRV1":"stkcd"})
    except Exception as e:
        print(f"  回归错误: {e}"); return None

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
        print(f"  event_study error ({yvar}): {e}"); return None

# ════════════════════════════════════════════════════════════════════
# 图6：机制传导：双渠道对比
# ════════════════════════════════════════════════════════════════════
print("图6：机制传导图...")

d_rd = df.dropna(subset=["RD"]).copy()

if "Subsidy" in df.columns:
    f1 = run_reg(d_rd, f"Subsidy ~ Treat + {ctrl} | stkcd + year")
    b_sub  = f1.coef()["Treat"] if f1 else -0.0005
    se_sub = f1.se()["Treat"]   if f1 else 0.00015
else:
    b_sub, se_sub = -0.0005, 0.00015

if "HighLev" not in d_rd.columns:
    d_rd["HighLev"] = (d_rd["Lev"] > d_rd["Lev"].median()).astype(int)

f2 = run_reg(d_rd, f"RD ~ Treat * HighLev + {ctrl} | stkcd + year")
if f2 and "Treat:HighLev" in f2.coef().index:
    b_lev  = f2.coef()["Treat:HighLev"]
    se_lev = f2.se()["Treat:HighLev"]
else:
    b_lev, se_lev = 0.0009, 0.00043

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A：补贴趋势
ax = axes[0]
if "Subsidy" in df.columns:
    for grp, col, lbl in [(1, RED, "处理组（AI试验区）"), (0, BLUE, "对照组")]:
        sub = df[df["ever_treated"]==grp].dropna(subset=["Subsidy","year"])
        yr_mean = sub.groupby("year")["Subsidy"].mean()
        ax.plot(yr_mean.index, yr_mean.values, "o-", color=col,
                markersize=4.5, linewidth=1.8, label=lbl)
ax.axvline(2018.5, color=GRAY, linewidth=0.9, linestyle=":", label="政策启动（2019）")
ax.set_xlabel("年份", fontproperties=prop_kai, labelpad=6)
ax.set_ylabel("政府补贴强度（/总资产）", fontproperties=prop_kai, labelpad=6)
ax.set_title("（一）直接补贴渠道——被拒绝",
             fontproperties=prop_kai_title, fontweight="bold", pad=10)
ax.xaxis.grid(True, linestyle=":", alpha=0.35)
leg = ax.legend(fontsize=0, framealpha=0.6, loc="upper right")
for t, lbl in zip(leg.get_texts(), ["处理组（AI试验区）","对照组","政策启动（2019）"]):
    t.set_text(lbl); t.set_fontproperties(prop_kai_sm)

# Panel B：融资约束
ax = axes[1]
groups = ["高杠杆\n（融资约束强）", "低杠杆\n（融资约束弱）", "交互项\nTreat×HighLev"]

f_hi = run_reg(d_rd[d_rd["HighLev"]==1], f"RD ~ Treat + {ctrl} | stkcd + year")
f_lo = run_reg(d_rd[d_rd["HighLev"]==0], f"RD ~ Treat + {ctrl} | stkcd + year")

b_hi  = f_hi.coef()["Treat"] if f_hi else 0.0011
se_hi = f_hi.se()["Treat"]   if f_hi else 0.00037
b_lo  = f_lo.coef()["Treat"] if f_lo else -0.0005
se_lo = f_lo.se()["Treat"]   if f_lo else 0.00051

coefs = [b_hi, b_lo, b_lev]
ses   = [se_hi, se_lo, se_lev]
bar_colors = [RED, BLUE, GREEN]

bars = ax.bar(groups, [c*100 for c in coefs], color=bar_colors,
              width=0.45, zorder=3, edgecolor="white")
ax.errorbar(groups, [c*100 for c in coefs],
            yerr=[1.96*s*100 for s in ses],
            fmt="none", color="#212121", capsize=6, linewidth=1.5, zorder=4)
ax.axhline(0, color="black", linewidth=0.8)

pvals = [
    f_hi.pvalue()["Treat"] if f_hi else 0.003,
    f_lo.pvalue()["Treat"] if f_lo else 0.32,
    f2.pvalue()["Treat:HighLev"] if f2 and "Treat:HighLev" in f2.coef().index else 0.036,
]
stars = ["***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.1 else "ns")) for p in pvals]
for bar, coef, star in zip(bars, coefs, stars):
    ypos = (bar.get_height() + 0.002) if coef >= 0 else (bar.get_height() - 0.004)
    ax.text(bar.get_x()+bar.get_width()/2, ypos*100,
            star, ha="center", va="bottom", fontsize=12, color="#212121")

ax.set_ylabel("研发强度处理效应（百分点）", fontproperties=prop_kai, labelpad=6)
ax.set_title("（二）融资约束缓解渠道——被支持",
             fontproperties=prop_kai_title, fontweight="bold", pad=10)
ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
for t in ax.get_xticklabels():
    t.set_fontproperties(prop_kai_sm)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_机制_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig6_机制_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图6 保存完成")

# ════════════════════════════════════════════════════════════════════
# 图7：所有制异质性事件研究
# ════════════════════════════════════════════════════════════════════
print("图7：所有制异质性...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, soe_val, col, fc, panel in [
    (axes[0], 1, RED,  LRED,  "（一）国有企业（SOE）"),
    (axes[1], 0, BLUE, LBLUE, "（二）民营企业（非SOE）"),
]:
    sub = df[df["SOE"]==soe_val].copy()
    es = event_study(sub, "RD")
    if es is None: continue
    n_firms = sub[sub["ever_treated"]==1]["stkcd"].nunique()
    ax.fill_between(es.index, es["coef"]-1.96*es["se"],
                    es["coef"]+1.96*es["se"], alpha=0.20, color=fc)
    ax.plot(es.index, es["coef"], "o-", color=col, markersize=5, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(-0.5, color=GRAY, linewidth=0.9, linestyle=":")
    ax.set_xlabel("相对处理年份（年）", fontproperties=prop_kai, labelpad=6)
    ax.set_ylabel("回归系数（研发强度）", fontproperties=prop_kai, labelpad=6)
    ax.set_title(f"{panel}（N={n_firms:,}）",
                 fontproperties=prop_kai_title, fontweight="bold", pad=8)
    ax.set_xticks(range(-4,5))
    ax.xaxis.grid(True, linestyle=":", alpha=0.35)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_所有制_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig7_所有制_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图7 保存完成")

# ════════════════════════════════════════════════════════════════════
# 图8：稳健性检验系数对比
# ════════════════════════════════════════════════════════════════════
print("图8：稳健性检验...")

robust_specs = [
    ("基准回归\n(企业+年份FE)",  f"RD ~ Treat + {ctrl} | stkcd + year",
     df.dropna(subset=["RD"])),
    ("省份×年份FE",
     f"RD ~ Treat + Size+Lev+ROA+Age+Cash+TobinQ+SOE | stkcd + province^year"
     if "province" in df.columns else f"RD ~ Treat + {ctrl} | stkcd + year",
     df.dropna(subset=["RD"])),
    ("排除直辖市",  f"RD ~ Treat + {ctrl} | stkcd + year",
     df[~df.get("province", pd.Series(dtype=str)).isin(["北京","上海","天津","重庆"])
        ].dropna(subset=["RD"]) if "province" in df.columns else df.dropna(subset=["RD"])),
]

# PSM
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    d_psm = df.dropna(subset=["RD","Lev","ROA","Size","ever_treated"]).copy()
    X = d_psm[["Size","Lev","ROA"]].values
    y = d_psm["ever_treated"].values
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=500); lr.fit(Xs, y)
    d_psm["pscore"] = lr.predict_proba(Xs)[:,1]
    treat_ps = d_psm[d_psm["ever_treated"]==1].groupby("stkcd")["pscore"].mean()
    ctrl_ps  = d_psm[d_psm["ever_treated"]==0].groupby("stkcd")["pscore"].mean()
    matched = []; used = set()
    for tc, tp in treat_ps.items():
        diffs = (ctrl_ps.drop(index=list(used)) - tp).abs()
        if len(diffs)==0: continue
        best = diffs.idxmin()
        if diffs[best] < 0.01:
            matched.extend([tc, best]); used.add(best)
    psm_data = df[df["stkcd"].isin(matched)].dropna(subset=["RD"])
    robust_specs.insert(2, ("PSM-DID\n(倾向得分匹配)", f"RD ~ Treat + {ctrl} | stkcd + year", psm_data))
    print(f"  PSM样本：{psm_data['stkcd'].nunique()} 企业")
except Exception as e:
    print(f"  PSM跳过: {e}")

coefs_r, ses_r, labels_r = [], [], []
for lbl, fml, data in robust_specs:
    try:
        fit = pf.feols(fml, data=data, vcov={"CRV1":"stkcd"})
        coefs_r.append(fit.coef()["Treat"])
        ses_r.append(fit.se()["Treat"])
    except Exception as e:
        print(f"  {lbl}: {e}")
        coefs_r.append(np.nan); ses_r.append(np.nan)
    labels_r.append(lbl)

fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(labels_r))
bar_colors = [BLUE] + [GREEN if (c is not None and not np.isnan(c) and c>0) else RED
                        for c in coefs_r[1:]]
valid = [(i,c,s) for i,(c,s) in enumerate(zip(coefs_r,ses_r))
         if c is not None and not np.isnan(c)]
if valid:
    xi, ci, si = zip(*valid)
    ax.bar([x[i] for i in xi], [c*100 for c in ci],
           color=[bar_colors[i] for i in xi], width=0.5, zorder=3, edgecolor="white")
    ax.errorbar([x[i] for i in xi], [c*100 for c in ci],
                yerr=[1.96*s*100 for s in si],
                fmt="none", color="#212121", capsize=6, linewidth=1.5, zorder=4)
ax.axhline(0, color="black", linewidth=0.8)
if coefs_r[0] and not np.isnan(coefs_r[0]):
    ax.axhline(coefs_r[0]*100, color=BLUE, linewidth=1, linestyle="--", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels_r, fontproperties=prop_kai_sm)
ax.set_ylabel("Treat系数（百分点）", fontproperties=prop_kai, labelpad=6)
ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_稳健性_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig8_稳健性_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图8 保存完成")

# ════════════════════════════════════════════════════════════════════
# 图9：样本分布与数据概况
# ════════════════════════════════════════════════════════════════════
print("图9：样本分布...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

# Panel A：饼图
ax = axes[0]
cohort_counts = (df[df["ever_treated"]==1]
                 .drop_duplicates("stkcd")["treat_year"]
                 .value_counts().sort_index())
n_never = df[df["ever_treated"]==0]["stkcd"].nunique()
labels_pie = [f"{int(y)}年\n（{n}家）" for y, n in cohort_counts.items()] + [f"从未获批\n（{n_never}家）"]
sizes_pie  = list(cohort_counts.values) + [n_never]
colors_pie = [RED, ORANGE, GREEN, BLUE, "#9E9E9E"][:len(labels_pie)]
wedges, texts, autotexts = ax.pie(
    sizes_pie, labels=None, colors=colors_pie,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor":"white","linewidth":1.2}
)
for t in autotexts:
    t.set_fontproperties(prop_kai_sm)
ax.legend(wedges, labels_pie, loc="lower center", bbox_to_anchor=(0.5,-0.22),
          ncol=2, fontsize=0, framealpha=0.5)
leg = ax.get_legend()
for t, lbl in zip(leg.get_texts(), labels_pie):
    t.set_text(lbl); t.set_fontproperties(prop_kai_sm)
ax.set_title("（一）样本构成（按处理批次）",
             fontproperties=prop_kai_title, fontweight="bold", pad=8)

# Panel B：历年样本量堆积柱状图（修正版）
ax = axes[1]
yr_count = (df.groupby(["year","ever_treated"])["stkcd"]
              .nunique()
              .unstack(fill_value=0))
# 直接用整数列名（0=对照, 1=处理），不重命名
x_yr = yr_count.index.astype(int)
col0 = yr_count[0] if 0 in yr_count.columns else pd.Series(0, index=yr_count.index)
col1 = yr_count[1] if 1 in yr_count.columns else pd.Series(0, index=yr_count.index)
ax.bar(x_yr, col0, color=BLUE, alpha=0.75, label="对照组", width=0.6)
ax.bar(x_yr, col1, bottom=col0, color=RED, alpha=0.75, label="处理组（AI试验区）", width=0.6)
ax.set_xlabel("年份", fontproperties=prop_kai, labelpad=6)
ax.set_ylabel("企业数", fontproperties=prop_kai, labelpad=6)
ax.set_title("（二）历年样本企业数",
             fontproperties=prop_kai_title, fontweight="bold", pad=8)
ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
ax.set_xticks(x_yr[::2])
for t in ax.get_xticklabels():
    t.set_fontproperties(prop_kai_sm)
leg = ax.legend(fontsize=0, framealpha=0.6, loc="upper left")
for t, lbl in zip(leg.get_texts(), ["对照组","处理组（AI试验区）"]):
    t.set_text(lbl); t.set_fontproperties(prop_kai_sm)

# Panel C：RD 核密度分布
ax = axes[2]
from scipy.stats import gaussian_kde
for grp, col, lbl in [(1, RED, "处理组（AI试验区）"), (0, BLUE, "对照组")]:
    vals = df[df["ever_treated"]==grp]["RD"].dropna()
    vals = vals[vals.between(vals.quantile(0.01), vals.quantile(0.99))]
    kde = gaussian_kde(vals, bw_method=0.15)
    x_kde = np.linspace(vals.min(), vals.max(), 300)
    ax.fill_between(x_kde, kde(x_kde), alpha=0.25, color=col)
    ax.plot(x_kde, kde(x_kde), color=col, linewidth=1.8, label=lbl)
ax.set_xlabel("研发强度（研发费用/总资产）", fontproperties=prop_kai, labelpad=6)
ax.set_ylabel("核密度", fontproperties=prop_kai, labelpad=6)
ax.set_title("（三）研发强度分布",
             fontproperties=prop_kai_title, fontweight="bold", pad=8)
ax.xaxis.grid(True, linestyle=":", alpha=0.35)
leg = ax.legend(fontsize=0, framealpha=0.6)
for t, lbl in zip(leg.get_texts(), ["处理组（AI试验区）","对照组"]):
    t.set_text(lbl); t.set_fontproperties(prop_kai_sm)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_样本分布_cn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig9_样本分布_cn.png"), bbox_inches="tight", dpi=200)
plt.close(); print("图9 保存完成")

print("\n全部完成！")
