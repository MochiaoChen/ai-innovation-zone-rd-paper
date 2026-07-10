"""
08_robustness.py — 稳健性检验 Table 4
列 (1): 省份×年份固定效应
列 (2): PSM-DID
列 (3): 排除直辖市
列 (4): 安慰剂检验（500次置换）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN      = os.path.join(BASE, "data/clean")
OUT_TABLES = os.path.join(BASE, "output/tables")
OUT_FIGS   = os.path.join(BASE, "output/figures")

df = pd.read_csv(os.path.join(CLEAN, "analysis_data.csv"), encoding="utf-8-sig", low_memory=False)
print(f"[08] 样本: {len(df)} 观测, {df['stkcd'].nunique()} 公司")

BASE_CONTROLS = ["Size", "Lev", "ROA", "Age", "Cash"]
EXTRA = ["TobinQ", "SOE"]
avail_controls = BASE_CONTROLS + [v for v in EXTRA if v in df.columns and df[v].notna().mean() > 0.4]
CONTROLS = " + ".join(avail_controls)
OUTCOME = "RD"

robust_results = {}
true_coef = np.nan

try:
    import pyfixest as pf

    # ── 列 (1): 省份×年份固定效应
    print("[08] 列(1): 省份×年份固定效应...")
    if "province_year" in df.columns and df["province_year"].notna().mean() > 0.5:
        sub1 = df.dropna(subset=[OUTCOME, "province_year"]).copy()
        try:
            m1 = pf.feols(
                fml=f"{OUTCOME} ~ Treat + {CONTROLS} | firm_id + province_year",
                data=sub1,
                vcov={"CRV1": "firm_id"},
            )
            robust_results["prov_year_fe"] = m1
            c = m1.coef().get("Treat", np.nan)
            t = m1.tstat().get("Treat", np.nan)
            print(f"  Treat: {c:.4f} (t={t:.3f}, N={m1._N})")
        except Exception as e:
            print(f"  ⚠️  失败: {e}")
    else:
        print("  ⚠️  province_year 缺失，跳过列(1)")

    # ── 列 (2): PSM-DID
    print("[08] 列(2): PSM-DID...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        # 选择处理前最后一年有足够样本的年份
        for pre_year in [2018, 2017, 2016]:
            pre_df = df[df["year"] == pre_year].copy()
            if len(pre_df) > 100:
                break
        match_vars = [v for v in ["Size", "Lev", "ROA", "Age"] if v in pre_df.columns and pre_df[v].notna().mean() > 0.5]
        pre_df = pre_df.dropna(subset=match_vars + ["ever_treated"])

        X = pre_df[match_vars].values
        y = pre_df["ever_treated"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_scaled, y)
        pre_df["pscore"] = lr.predict_proba(X_scaled)[:, 1]

        treat_idx = pre_df[pre_df["ever_treated"] == 1].index
        ctrl_idx  = pre_df[pre_df["ever_treated"] == 0].index

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(pre_df.loc[ctrl_idx, ["pscore"]].values)
        distances, indices = nn.kneighbors(pre_df.loc[treat_idx, ["pscore"]].values)

        # 卡尺 0.05
        caliper = 0.05
        matched_ctrl_idx = [ctrl_idx[i[0]] for d, i in zip(distances, indices) if d[0] <= caliper]
        matched_treat_idx = [t for t, d in zip(treat_idx, distances) if d[0] <= caliper]

        matched_codes = set(
            pre_df.loc[matched_ctrl_idx, "stkcd"].tolist() +
            pre_df.loc[matched_treat_idx, "stkcd"].tolist()
        )
        df_psm = df[df["stkcd"].isin(matched_codes)].copy()
        print(f"  PSM 匹配后样本公司数: {len(matched_codes)} (处理组={len(matched_treat_idx)}, 对照组={len(matched_ctrl_idx)})")

        m2 = pf.feols(
            fml=f"{OUTCOME} ~ Treat + {CONTROLS} | firm_id + year",
            data=df_psm.dropna(subset=[OUTCOME]),
            vcov={"CRV1": "firm_id"},
        )
        robust_results["psm_did"] = m2
        c = m2.coef().get("Treat", np.nan)
        t = m2.tstat().get("Treat", np.nan)
        print(f"  Treat: {c:.4f} (t={t:.3f}, N={m2._N})")

    except Exception as e:
        print(f"  ⚠️  PSM-DID 失败: {e}")

    # ── 列 (3): 排除直辖市
    print("[08] 列(3): 排除直辖市...")
    MUNIS = ["北京", "上海", "天津", "重庆"]
    city_col = "city_clean" if "city_clean" in df.columns else None
    if city_col:
        df_no_muni = df[~df[city_col].isin(MUNIS)].copy()
        sub3 = df_no_muni.dropna(subset=[OUTCOME]).copy()
        try:
            m3 = pf.feols(
                fml=f"{OUTCOME} ~ Treat + {CONTROLS} | firm_id + year",
                data=sub3,
                vcov={"CRV1": "firm_id"},
            )
            robust_results["no_muni"] = m3
            c = m3.coef().get("Treat", np.nan)
            t = m3.tstat().get("Treat", np.nan)
            print(f"  Treat: {c:.4f} (t={t:.3f}, N={m3._N})")
        except Exception as e:
            print(f"  ⚠️  失败: {e}")
    else:
        print("  ⚠️  city_clean 缺失，跳过列(3)")

    # ── 列 (4): 主回归系数（用于对比安慰剂）
    print("[08] 获取主回归真实系数（用于安慰剂对比）...")
    sub_main = df.dropna(subset=[OUTCOME]).copy()
    try:
        m_main = pf.feols(
            fml=f"{OUTCOME} ~ Treat + {CONTROLS} | firm_id + year",
            data=sub_main,
            vcov={"CRV1": "firm_id"},
        )
        true_coef = m_main.coef().get("Treat", np.nan)
        robust_results["main"] = m_main
        print(f"  真实系数: {true_coef:.4f}")
    except Exception as e:
        print(f"  ⚠️  主回归失败: {e}")

    # ── 安慰剂检验（500次置换 Treat）
    print("[08] 安慰剂检验（500次置换）...")
    N_PERM = 500
    placebo_coefs = []
    rng = np.random.default_rng(42)

    sub_placebo = df.dropna(subset=[OUTCOME]).copy()
    unique_codes = sub_placebo["stkcd"].unique()

    for i in range(N_PERM):
        # 随机打乱 treat_year（公司层面）
        codes_arr = unique_codes.copy()
        fake_treat_year_map = dict(zip(
            codes_arr,
            rng.permutation(sub_placebo.groupby("stkcd")["treat_year"].first().reindex(codes_arr).values)
        ))
        sub_perm = sub_placebo.copy()
        sub_perm["fake_treat_year"] = sub_perm["stkcd"].map(fake_treat_year_map)
        sub_perm["fake_Treat"] = (
            sub_perm["fake_treat_year"].notna() &
            (sub_perm["year"] >= sub_perm["fake_treat_year"])
        ).astype(int)

        try:
            m_p = pf.feols(
                fml=f"{OUTCOME} ~ fake_Treat + {CONTROLS} | firm_id + year",
                data=sub_perm,
                vcov={"CRV1": "firm_id"},
            )
            placebo_coefs.append(m_p.coef().get("fake_Treat", np.nan))
        except Exception:
            placebo_coefs.append(np.nan)

        if (i + 1) % 100 == 0:
            print(f"  置换进度: {i+1}/{N_PERM}")

    placebo_arr = np.array([x for x in placebo_coefs if not np.isnan(x)])
    pctile = (placebo_arr < true_coef).mean() if not np.isnan(true_coef) else np.nan
    print(f"  真实系数 {true_coef:.4f} 高于 {pctile:.1%} 的安慰剂系数")

    # 画安慰剂分布图
    if len(placebo_arr) > 10:
        plt.rcParams.update({"font.family": "serif", "font.size": 11,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(placebo_arr, bins=50, color="steelblue", alpha=0.7, edgecolor="none",
                label="Placebo coefficients")
        if not np.isnan(true_coef):
            ax.axvline(true_coef, color="firebrick", linewidth=2,
                       linestyle="--", label=f"True coef = {true_coef:.4f}")
        ax.set_xlabel("Coefficient on $Treat_{it}$ (placebo)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Placebo Test: Distribution of Permuted Coefficients\n(500 Replications)", fontsize=12)
        ax.legend(frameon=False)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(os.path.join(OUT_FIGS, f"figure2_placebo.{ext}"),
                        dpi=300 if ext == "png" else None, bbox_inches="tight")
        plt.close()
        print(f"  安慰剂图已保存: figure2_placebo.png/pdf")

except ImportError:
    print("  pyfixest 不可用，跳过稳健性检验")


def make_table4_tex(results, out_dir):
    col_map = [
        ("prov_year_fe", "(1) Province$\\times$Year FE"),
        ("psm_did",      "(2) PSM-DID"),
        ("no_muni",      "(3) Excl. Municipalities"),
        ("main",         "(4) Baseline (for reference)"),
    ]
    avail = [(k, l) for k, l in col_map if k in results]
    if not avail:
        return

    n_col = len(avail)
    hdr = " & ".join([l for _, l in avail])
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robustness Checks}",
        r"\label{tab:robust}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{l" + "c" * n_col + "}",
        r"\toprule",
        f" & {hdr} \\\\",
        r"\midrule",
    ]

    for var, display in [("Treat", "$Treat_{it}$")]:
        coef_cells = [display]
        tstat_cells = [""]
        for k, _ in avail:
            m = results[k]
            try:
                c = m.coef().get(var, np.nan)
                t = m.tstat().get(var, np.nan)
                p = m.pvalue().get(var, np.nan)
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                coef_cells.append(f"{c:.4f}{stars}")
                tstat_cells.append(f"({t:.3f})")
            except Exception:
                coef_cells.append("")
                tstat_cells.append("")
        tex.append(" & ".join(coef_cells) + r" \\")
        tex.append(" & ".join(tstat_cells) + r" \\")

    tex.append(r"\midrule")
    n_cells = ["$N$"]
    for k, _ in avail:
        m = results[k]
        n = getattr(m, "_N", None)
        n_cells.append(f"{int(n):,}" if n else "")
    tex.append(" & ".join(n_cells) + r" \\")

    # FE 说明
    fe_desc = {
        "prov_year_fe": "Firm, Province$\\times$Year",
        "psm_did":      "Firm, Year",
        "no_muni":      "Firm, Year",
        "main":         "Firm, Year",
    }
    fe_cells = ["Fixed Effects"]
    for k, _ in avail:
        fe_cells.append(fe_desc.get(k, "Firm, Year"))
    tex.append(" & ".join(fe_cells) + r" \\")

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textit{Notes}: Dependent variable is R\&D intensity. "
        r"Column (1) replaces year FE with province$\times$year FE. "
        r"Column (2) uses propensity-score matched sample (1:1 nearest neighbour, caliper 0.05, "
        r"matched on 2018 pre-treatment characteristics). "
        r"Column (3) excludes firms in the four directly-administered municipalities "
        r"(Beijing, Shanghai, Tianjin, Chongqing). "
        r"Column (4) replicates the baseline for comparison. "
        r"See Figure~\\ref{fig:placebo} for the placebo permutation test (500 replications). "
        r"$t$-statistics in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.10$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    tex_path = os.path.join(out_dir, "table4_robustness.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"  Table 4 已输出: {tex_path}")

    # CSV
    csv_rows = [[""] + [l for _, l in avail]]
    for var in ["Treat"]:
        row_c = [var]
        row_t = [""]
        for k, _ in avail:
            m = results[k]
            try:
                c = m.coef().get(var, np.nan)
                t = m.tstat().get(var, np.nan)
                p = m.pvalue().get(var, np.nan)
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                row_c.append(f"{c:.4f}{stars}")
                row_t.append(f"({t:.3f})")
            except Exception:
                row_c.append("")
                row_t.append("")
        csv_rows.append(row_c)
        csv_rows.append(row_t)
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(out_dir, "table4_robustness.csv"), index=False, header=False, encoding="utf-8-sig"
    )


make_table4_tex(robust_results, OUT_TABLES)
print(f"\n[STEP 8 COMPLETE] 稳健性检验完成 | 模型数 = {len(robust_results)}")
