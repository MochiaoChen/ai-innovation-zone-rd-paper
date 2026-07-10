"""
Step 02: 数据清洗 & 面板构建
- 合并利润表、资产负债表、公司信息
- 匹配 AI 试验区处理变量
- 清洗：剔除金融业、ST、缺失值；Winsorize
- 输出：data/clean/panel_data.csv
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, "data/raw")
CLEAN = os.path.join(BASE, "data/clean")
TREATMENT = os.path.join(BASE, "data/treatment")

SAMPLE_START = 2013
SAMPLE_END   = 2025

# ── 金融行业关键词（用于剔除）
FIN_KEYWORDS = ["银行", "保险", "证券", "信托", "基金", "期货",
                "租赁", "金融", "投资", "资产管理"]


def parse_number(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s in ("--", "None", "", "不适用", "nan"):
        return np.nan
    multiplier = 1.0
    if "亿" in s:
        multiplier = 1e8
        s = s.replace("亿", "")
    elif "万" in s:
        multiplier = 1e4
        s = s.replace("万", "")
    s = s.replace(",", "").replace("%", "")
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def normalize_city(city):
    """标准化城市名，去掉「市」「区」等后缀"""
    if not city or pd.isna(city):
        return None
    city = str(city).strip()
    city = re.sub(r'[市区县]$', '', city)
    return city


def parse_year(period_str):
    """从 '2022' 或 '2022-12-31' 中提取年份整数"""
    if pd.isna(period_str):
        return np.nan
    s = str(period_str).strip()
    m = re.match(r'^(\d{4})', s)
    if m:
        return int(m.group(1))
    return np.nan


def winsorize(series, lower=0.01, upper=0.99):
    """两端缩尾"""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def main():
    print("[02] 开始数据清洗与面板构建...")

    # ── 1. 读取原始数据
    print("[02] 读取原始数据...")
    inc_path = os.path.join(RAW, "income_partial.csv")
    bs_path  = os.path.join(RAW, "balance_partial.csv")
    prof_path = os.path.join(RAW, "profile_partial.csv")

    cf_path      = os.path.join(RAW, "cashflow_partial.csv")
    prices_path  = os.path.join(RAW, "prices_partial.csv")   # 01b 生成
    mktcap_path  = os.path.join(RAW, "mktcap_partial.csv")   # 01 生成（年末总市值）

    for p in [inc_path, bs_path, prof_path]:
        if not os.path.exists(p):
            print(f"[02] ✗ 找不到文件：{p}，请先运行 01_fetch_data.py")
            return

    df_inc  = pd.read_csv(inc_path,  encoding="utf-8-sig")
    df_bs   = pd.read_csv(bs_path,   encoding="utf-8-sig")
    df_prof = pd.read_csv(prof_path, encoding="utf-8-sig")

    df_cf     = pd.read_csv(cf_path,     encoding="utf-8-sig") if os.path.exists(cf_path)     else None
    df_prices = pd.read_csv(prices_path, encoding="utf-8-sig") if os.path.exists(prices_path) else None
    df_mktcap = pd.read_csv(mktcap_path, encoding="utf-8-sig") if os.path.exists(mktcap_path) else None

    print(f"[02] 利润表：{len(df_inc)} 行 | 资产负债表：{len(df_bs)} 行 | 公司信息：{len(df_prof)} 行")
    print(f"[02] 现金流量表：{len(df_cf) if df_cf is not None else '缺失'} 行 | "
          f"年末价格：{len(df_prices) if df_prices is not None else '缺失（请先跑 01b）'} 行 | "
          f"年末市值（01抓）：{len(df_mktcap) if df_mktcap is not None else '缺失'} 行")

    # ── 2. 利润表清洗
    print("[02] 处理利润表...")
    # 找报告期列（可能是「报告期」或包含「报告」的列）
    period_col = next((c for c in df_inc.columns if "报告期" in c), None)
    rd_col     = next((c for c in df_inc.columns if "研发费用" in c), None)
    np_col     = next((c for c in df_inc.columns if "净利润" in c and "同比" not in c), None)
    rev_col    = next((c for c in df_inc.columns if "营业总收入" in c and "同比" not in c), None)

    if period_col:
        df_inc["year"] = df_inc[period_col].apply(parse_year)
    if rd_col:
        df_inc["rd_expense"] = df_inc[rd_col].apply(parse_number)
    if np_col:
        df_inc["net_income"] = df_inc[np_col].apply(parse_number)
    if rev_col:
        df_inc["revenue"] = df_inc[rev_col].apply(parse_number)

    df_inc_clean = df_inc[["code", "year", "rd_expense", "net_income", "revenue"]].copy()
    df_inc_clean = df_inc_clean.dropna(subset=["year"])
    df_inc_clean["year"] = df_inc_clean["year"].astype(int)

    # ── 3. 资产负债表清洗
    print("[02] 处理资产负债表...")
    bs_period = next((c for c in df_bs.columns if "报告期" in c), None)

    def find_col(df, keyword):
        for c in df.columns:
            if keyword in c:
                return c
        return None

    asset_col  = find_col(df_bs, "资产合计")
    liab_col   = find_col(df_bs, "负债合计")
    equity_col = find_col(df_bs, "所有者权益")
    cash_col   = find_col(df_bs, "货币资金")
    share_col  = find_col(df_bs, "实收资本")   # 总股本（面值1元/股）

    if bs_period:
        df_bs["year"] = df_bs[bs_period].apply(parse_year)
    for col, name in [(asset_col, "total_assets"), (liab_col, "total_liabilities"),
                      (equity_col, "total_equity"), (cash_col, "cash"),
                      (share_col, "share_capital")]:
        if col:
            df_bs[name] = df_bs[col].apply(parse_number)
        else:
            df_bs[name] = np.nan

    df_bs_clean = df_bs[["code", "year", "total_assets", "total_liabilities",
                          "total_equity", "cash", "share_capital"]].copy()
    df_bs_clean = df_bs_clean.dropna(subset=["year"])
    df_bs_clean["year"] = df_bs_clean["year"].astype(int)

    # ── 4. 公司信息清洗
    print("[02] 处理公司信息...")
    df_prof["city_clean"] = df_prof["city"].apply(normalize_city)
    df_prof["ipo_year"] = df_prof["ipo_date"].apply(
        lambda x: int(str(x)[:4]) if pd.notna(x) and len(str(x)) >= 4 else np.nan
    )

    # ── 4b. 现金流量表清洗（CAPEX）
    df_cf_clean = None
    if df_cf is not None:
        print("[02] 处理现金流量表...")
        cf_period = next((c for c in df_cf.columns if "报告期" in c), None)
        capex_col = next((c for c in df_cf.columns if "capex_raw" in c or "购建固定资产" in c
                          or "购置固定资产" in c), None)
        if cf_period:
            df_cf["year"] = df_cf[cf_period].apply(parse_year)
        elif "year" in df_cf.columns:
            pass
        if capex_col:
            df_cf["capex"] = df_cf[capex_col].apply(parse_number)
        elif "capex_raw" in df_cf.columns:
            df_cf["capex"] = df_cf["capex_raw"].apply(parse_number)
        else:
            df_cf["capex"] = np.nan
        df_cf_clean = df_cf[["code", "year", "capex"]].dropna(subset=["year"]).copy()
        df_cf_clean["year"] = df_cf_clean["year"].astype(int)
        # CAPEX 取绝对值（现金流量表中为负数支出）
        df_cf_clean["capex"] = df_cf_clean["capex"].abs()
        print(f"[02] 现金流量表清洗后：{len(df_cf_clean)} 行")

    # ── 4c. 年末价格清洗 → 合并后计算 total_mktcap = close_price × share_capital
    df_prices_clean = None
    if df_prices is not None:
        print("[02] 处理年末收盘价数据...")
        df_prices["code"] = df_prices["code"].astype(str).str.zfill(6)
        df_prices["year"] = pd.to_numeric(df_prices["year"], errors="coerce").astype("Int64")
        df_prices["close_price"] = pd.to_numeric(df_prices["close_price"], errors="coerce")
        df_prices_clean = df_prices[["code", "year", "close_price"]].dropna().copy()
        df_prices_clean["year"] = df_prices_clean["year"].astype(int)
        print(f"[02] 年末价格数据：{len(df_prices_clean)} 行")

    # ── 5. 合并
    print("[02] 合并数据...")
    # 统一 code 为字符串6位
    for _df in [df_inc_clean, df_bs_clean, df_prof, df_cf_clean]:
        if _df is not None and "code" in _df.columns:
            _df["code"] = _df["code"].astype(str).str.zfill(6)
    panel = df_inc_clean.merge(df_bs_clean, on=["code", "year"], how="inner")
    panel = panel.merge(df_prof[["code", "city_clean", "ipo_year", "industry",
                                  "company_name"]], on="code", how="left")
    if df_cf_clean is not None:
        panel = panel.merge(df_cf_clean, on=["code", "year"], how="left")
        print(f"[02] 合并现金流量表后：CAPEX 覆盖率 {panel['capex'].notna().mean():.1%}")
    else:
        panel["capex"] = np.nan

    # 合并年末市值（优先用 mktcap_partial.csv，其次用 prices_partial.csv × share_capital）
    if df_mktcap is not None:
        # mktcap_partial 格式：code, year, total_mktcap
        df_mc_clean = df_mktcap[["code", "year", "total_mktcap"]].dropna().copy()
        df_mc_clean["code"] = df_mc_clean["code"].astype(str).str.zfill(6)
        df_mc_clean["year"] = pd.to_numeric(df_mc_clean["year"], errors="coerce").astype("Int64")
        df_mc_clean = df_mc_clean.dropna(subset=["year"])
        df_mc_clean["year"] = df_mc_clean["year"].astype(int)
        panel = panel.merge(df_mc_clean, on=["code", "year"], how="left")
        coverage = panel["total_mktcap"].notna().mean()
        print(f"[02] 年末市值（来自 mktcap_partial）：覆盖率 {coverage:.1%}")
    elif df_prices_clean is not None:
        panel = panel.merge(df_prices_clean, on=["code", "year"], how="left")
        panel["total_mktcap"] = panel["close_price"] * panel["share_capital"]
        coverage = panel["total_mktcap"].notna().mean()
        print(f"[02] 市值计算完成（收盘价×股本）：覆盖率 {coverage:.1%}")
    else:
        panel["close_price"]  = np.nan
        panel["total_mktcap"] = np.nan
        print("[02] ⚠️  市值数据缺失，TobinQ 将全为 NaN（01b 跑完后重跑此步骤）")

    print(f"[02] 合并后：{len(panel)} 条观测，{panel['code'].nunique()} 家公司")

    # ── 6. 过滤样本期
    panel = panel[(panel["year"] >= SAMPLE_START) & (panel["year"] <= SAMPLE_END)]
    print(f"[02] 样本期 {SAMPLE_START}-{SAMPLE_END} 后：{len(panel)} 条观测")

    # ── 7. 剔除金融行业
    if "industry" in panel.columns:
        fin_mask = panel["industry"].apply(
            lambda x: any(kw in str(x) for kw in FIN_KEYWORDS) if pd.notna(x) else False
        )
        panel = panel[~fin_mask]
        print(f"[02] 剔除金融行业后：{len(panel)} 条观测")

    # ── 8. 剔除总资产为空或为负的异常值
    panel = panel[panel["total_assets"] > 0]

    # ── 9. 构建变量
    print("[02] 构建基础变量...")
    panel["RD"]    = panel["rd_expense"] / panel["total_assets"]
    panel["CAPEX"] = panel["capex"] / panel["total_assets"]
    panel["Lev"]   = panel["total_liabilities"] / panel["total_assets"]
    panel["ROA"]   = panel["net_income"] / panel["total_assets"]
    panel["Cash"]  = panel["cash"] / panel["total_assets"]
    panel["Size"]  = np.log(panel["total_assets"])
    panel["Age"]   = panel["year"] - panel["ipo_year"]
    panel["Age"]   = panel["Age"].apply(lambda x: np.log(x + 1) if pd.notna(x) and x >= 0 else np.nan)
    # TobinQ = (总市值 + 总负债) / 总资产（简化版）
    panel["TobinQ"] = (panel["total_mktcap"] + panel["total_liabilities"]) / panel["total_assets"]

    # ── 10. 匹配 AI 试验区处理变量
    print("[02] 匹配 AI 试验区...")
    zones = pd.read_csv(os.path.join(TREATMENT, "ai_zones.csv"), encoding="utf-8-sig")
    zones["city"] = zones["city"].apply(normalize_city)
    city_to_year = dict(zip(zones["city"], zones["treat_year"]))

    panel["treat_year"] = panel["city_clean"].map(city_to_year)
    panel["Treat"] = (panel["treat_year"].notna() &
                      (panel["year"] >= panel["treat_year"])).astype(int)

    n_treated = panel[panel["treat_year"].notna()]["code"].nunique()
    print(f"[02] 处理组公司数：{n_treated}")

    # ── 11. Winsorize 连续变量
    print("[02] 缩尾处理（1%-99%）...")
    for col in ["RD", "CAPEX", "TobinQ", "Lev", "ROA", "Cash", "Size"]:
        if col in panel.columns and panel[col].notna().sum() > 0:
            panel[col] = winsorize(panel[col].dropna()).reindex(panel.index)

    # RD / CAPEX 下界为 0
    panel["RD"]    = panel["RD"].clip(lower=0)
    panel["CAPEX"] = panel["CAPEX"].clip(lower=0)

    miss_capex  = panel["CAPEX"].isna().mean()
    miss_tobinq = panel["TobinQ"].isna().mean()
    print(f"[02] CAPEX 缺失率：{miss_capex:.1%} | TobinQ 缺失率：{miss_tobinq:.1%}")

    # ── 12. 剔除连续观测不足 3 年的公司
    obs_count = panel.groupby("code")["year"].count()
    valid_codes = obs_count[obs_count >= 3].index
    panel = panel[panel["code"].isin(valid_codes)]
    print(f"[02] 剔除观测不足后：{len(panel)} 条观测，{panel['code'].nunique()} 家公司")

    # ── 13. 构建相对时间变量（用于事件研究）
    panel["rel_time"] = panel["year"] - panel["treat_year"]
    # 从未处理的公司 rel_time 设为 -999（标记用）
    panel["rel_time"] = panel["rel_time"].fillna(-999)

    # ── 14. 保存
    out_path = os.path.join(CLEAN, "panel_data.csv")
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[02] 面板数据已保存：{out_path}")

    # ── 汇总报告
    print("\n" + "=" * 50)
    print(f"[02] STEP 02 COMPLETE")
    print(f"     总观测数 N = {len(panel)}")
    print(f"     公司数 = {panel['code'].nunique()}")
    print(f"     年份范围 = {panel['year'].min()} - {panel['year'].max()}")
    print(f"     处理组公司数 = {panel[panel['treat_year'].notna()]['code'].nunique()}")
    print(f"     对照组公司数 = {panel[panel['treat_year'].isna()]['code'].nunique()}")
    print(f"     RD 缺失率 = {panel['RD'].isna().mean():.1%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
