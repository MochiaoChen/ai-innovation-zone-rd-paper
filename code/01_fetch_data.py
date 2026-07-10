"""
Step 01: 数据抓取（主循环，预计运行 8-12 小时）
- 从同花顺获取利润表（含研发费用）
- 从同花顺获取资产负债表
- 从同花顺获取现金流量表（用于 CAPEX）
- 从东方财富获取年末总市值历史（用于 TobinQ）
- 从巨潮资讯获取公司基本信息（含注册城市）
- 每抓取 50 家公司保存一次检查点，支持断点续跑
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import json
import re
from tqdm import tqdm
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE, "data/raw")
LOG_PATH = os.path.join(BASE, "data/raw/fetch_log.txt")

SLEEP_NORMAL = 0.8    # 正常请求间隔（秒）
SLEEP_RETRY  = 5.0    # 报错重试等待
MAX_RETRY    = 3      # 最大重试次数
CHECKPOINT_EVERY = 50 # 每 N 家公司保存一次


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_number(s):
    """把 '221.47亿'、'35.69万' 等字符串转为浮点数（单位：元）"""
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


def extract_city(address):
    """从注册地址中提取城市名"""
    if not address or pd.isna(address):
        return None
    address = str(address)
    # 直辖市：北京/上海/天津/重庆 直接出现在开头
    for muni in ["北京", "上海", "天津", "重庆"]:
        if address.startswith(muni):
            return muni
    # 其他省份：提取省份后的第一个「市」
    match = re.search(r'省(.{2,6}?)市', address)
    if match:
        return match.group(1)
    # 自治区
    match2 = re.search(r'(?:自治区|壮族自治区|回族自治区)(.{2,6}?)市', address)
    if match2:
        return match2.group(1)
    return None


def fetch_with_retry(func, *args, **kwargs):
    """带重试的 API 调用"""
    for attempt in range(MAX_RETRY):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(SLEEP_RETRY * (attempt + 1))
            else:
                raise e
    return None


def fetch_one_company(code):
    """抓取单个公司的全部所需数据，返回 dict"""
    result = {"code": code, "status": "ok"}

    # ── 利润表（研发费用、净利润、营业收入）
    try:
        inc = fetch_with_retry(
            ak.stock_financial_benefit_ths, symbol=code, indicator="按年度"
        )
        time.sleep(SLEEP_NORMAL)
        if inc is not None and not inc.empty:
            result["income"] = inc[
                [c for c in ["报告期", "研发费用", "*净利润", "*营业总收入"]
                 if c in inc.columns]
            ].copy()
    except Exception as e:
        result["income_err"] = str(e)

    # ── 资产负债表（总资产、总负债、货币资金）
    try:
        bs = fetch_with_retry(
            ak.stock_financial_debt_ths, symbol=code, indicator="按年度"
        )
        time.sleep(SLEEP_NORMAL)
        if bs is not None and not bs.empty:
            want = ["报告期", "*资产合计", "*负债合计", "*所有者权益（或股东权益）合计", "货币资金"]
            # 字段名可能略有差异，做模糊匹配
            col_map = {}
            for want_col in want:
                for actual_col in bs.columns:
                    if want_col.replace("*", "") in actual_col:
                        col_map[want_col] = actual_col
                        break
            keep = [col_map.get(c, c) for c in want if col_map.get(c, c) in bs.columns]
            result["balance"] = bs[keep].copy()
    except Exception as e:
        result["balance_err"] = str(e)

    # ── 现金流量表（CAPEX：购建固定资产等支付的现金）
    try:
        cf = fetch_with_retry(
            ak.stock_financial_cash_ths, symbol=code, indicator="按年度"
        )
        time.sleep(SLEEP_NORMAL)
        if cf is not None and not cf.empty:
            # 找报告期列
            period_col = next((c for c in cf.columns if "报告期" in c), None)
            # CAPEX 候选列名（同花顺版本不同，列名可能略有差异）
            capex_candidates = [
                "购建固定资产、无形资产和其他长期资产支付的现金",
                "购建固定资产无形资产和其他长期资产支付的现金",
                "购置固定资产、无形资产和其他长期资产支付的现金",
            ]
            capex_col = next(
                (c for c in cf.columns
                 if any(kw in c for kw in ["购建固定资产", "购置固定资产"])),
                None,
            )
            keep = [c for c in [period_col, capex_col] if c is not None]
            if keep:
                result["cashflow"] = cf[keep].copy()
                if capex_col:
                    result["_capex_col"] = capex_col
    except Exception as e:
        result["cashflow_err"] = str(e)

    # ── 年末总市值历史（东方财富 stock_a_lg_indicator）
    try:
        mktcap = fetch_with_retry(
            ak.stock_a_lg_indicator, symbol=code, indicator="总市值"
        )
        time.sleep(SLEEP_NORMAL)
        if mktcap is not None and not mktcap.empty:
            # 返回 DataFrame，列通常为 ["日期", "总市值"] 或类似
            date_col = next((c for c in mktcap.columns if "日期" in c or "date" in c.lower()), None)
            val_col  = next((c for c in mktcap.columns if "总市值" in c or "市值" in c), None)
            if date_col and val_col:
                mktcap_clean = mktcap[[date_col, val_col]].copy()
                mktcap_clean.columns = ["date", "total_mktcap"]
                mktcap_clean["date"] = pd.to_datetime(mktcap_clean["date"], errors="coerce")
                mktcap_clean["year"] = mktcap_clean["date"].dt.year
                # 取每年最后一个交易日的市值（即年末市值）
                mktcap_year = (
                    mktcap_clean.sort_values("date")
                    .groupby("year")["total_mktcap"]
                    .last()
                    .reset_index()
                )
                result["mktcap"] = mktcap_year
    except Exception as e:
        result["mktcap_err"] = str(e)

    # ── 公司基本信息（城市、上市日期、行业）
    try:
        profile = fetch_with_retry(ak.stock_profile_cninfo, symbol=code)
        time.sleep(SLEEP_NORMAL)
        if profile is not None and not profile.empty:
            row = profile.iloc[0]
            result["city_raw"] = row.get("注册地址", None)
            result["city"] = extract_city(row.get("注册地址", None))
            result["ipo_date"] = row.get("上市日期", None)
            result["industry"] = row.get("所属行业", None)
            result["company_name"] = row.get("A股简称", None)
    except Exception as e:
        result["profile_err"] = str(e)

    return result


def load_checkpoint():
    path = os.path.join(CHECKPOINT_DIR, "checkpoint.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": [], "failed": []}


def save_checkpoint(checkpoint):
    path = os.path.join(CHECKPOINT_DIR, "checkpoint.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)


def save_partial_results(all_income, all_balance, all_profile,
                         all_cashflow, all_mktcap):
    """保存当前已抓取的中间结果"""
    if all_income:
        pd.concat(all_income, ignore_index=True).to_csv(
            os.path.join(CHECKPOINT_DIR, "income_partial.csv"),
            index=False, encoding="utf-8-sig"
        )
    if all_balance:
        pd.concat(all_balance, ignore_index=True).to_csv(
            os.path.join(CHECKPOINT_DIR, "balance_partial.csv"),
            index=False, encoding="utf-8-sig"
        )
    if all_profile:
        pd.DataFrame(all_profile).to_csv(
            os.path.join(CHECKPOINT_DIR, "profile_partial.csv"),
            index=False, encoding="utf-8-sig"
        )
    if all_cashflow:
        pd.concat(all_cashflow, ignore_index=True).to_csv(
            os.path.join(CHECKPOINT_DIR, "cashflow_partial.csv"),
            index=False, encoding="utf-8-sig"
        )
    if all_mktcap:
        pd.concat(all_mktcap, ignore_index=True).to_csv(
            os.path.join(CHECKPOINT_DIR, "mktcap_partial.csv"),
            index=False, encoding="utf-8-sig"
        )


def main():
    log("=" * 60)
    log("STEP 01: 数据抓取开始")
    log("=" * 60)

    # ── 获取全部 A 股列表
    log("拉取 A 股公司列表...")
    stock_df = ak.stock_info_a_code_name()
    all_codes = stock_df["code"].tolist()
    log(f"A 股共 {len(all_codes)} 家公司")

    # ── 过滤 ST/*ST（减少无效请求）
    non_st = stock_df[~stock_df["name"].str.contains("ST|退", na=False)]["code"].tolist()
    log(f"剔除 ST/*ST 后剩余 {len(non_st)} 家")

    # ── 过滤金融行业（银行、保险、券商、信托）
    # 注意：行业信息需要抓取后才能剔除，这里先用股票代码前缀做粗筛
    # 金融股一般在 60xxxx/00xxxx 中，无法仅凭代码区分，先全部抓取后再剔除
    target_codes = non_st
    log(f"目标公司数：{len(target_codes)}")

    # ── 加载检查点
    checkpoint = load_checkpoint()
    done_set = set(checkpoint["done"])
    remaining = [c for c in target_codes if c not in done_set]
    log(f"已完成：{len(done_set)} | 剩余：{len(remaining)}")

    all_income   = []
    all_balance  = []
    all_profile  = []
    all_cashflow = []
    all_mktcap   = []

    # 加载已有的部分结果
    for fname, storage in [
        ("income_partial.csv",   all_income),
        ("balance_partial.csv",  all_balance),
        ("profile_partial.csv",  all_profile),
        ("cashflow_partial.csv", all_cashflow),
        ("mktcap_partial.csv",   all_mktcap),
    ]:
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, encoding="utf-8-sig")
                storage.append(df)
                log(f"加载已有数据：{fname}（{len(df)} 行）")
            except Exception:
                pass

    # ── 主循环
    failed_codes = []
    for i, code in enumerate(tqdm(remaining, desc="抓取进度")):
        try:
            data = fetch_one_company(code)

            # 存利润表
            if "income" in data:
                df_inc = data["income"].copy()
                df_inc["code"] = code
                all_income.append(df_inc)

            # 存资产负债表
            if "balance" in data:
                df_bs = data["balance"].copy()
                df_bs["code"] = code
                all_balance.append(df_bs)

            # 存现金流量表
            if "cashflow" in data:
                df_cf = data["cashflow"].copy()
                df_cf["code"] = code
                # 统一把 CAPEX 列重命名
                capex_col = data.get("_capex_col")
                if capex_col and capex_col in df_cf.columns:
                    df_cf = df_cf.rename(columns={capex_col: "capex_raw"})
                all_cashflow.append(df_cf)

            # 存年末市值
            if "mktcap" in data:
                df_mc = data["mktcap"].copy()
                df_mc["code"] = code
                all_mktcap.append(df_mc)

            # 存公司信息
            profile_row = {
                "code": code,
                "company_name": data.get("company_name"),
                "city": data.get("city"),
                "city_raw": data.get("city_raw"),
                "ipo_date": data.get("ipo_date"),
                "industry": data.get("industry"),
            }
            all_profile.append(profile_row)

            checkpoint["done"].append(code)

        except Exception as e:
            failed_codes.append(code)
            checkpoint["failed"].append(code)
            log(f"✗ {code} 最终失败：{e}")

        # 每 N 家保存一次检查点
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint)
            save_partial_results(all_income, all_balance, all_profile,
                                 all_cashflow, all_mktcap)
            log(f"检查点保存：已完成 {len(checkpoint['done'])} / {len(target_codes)}")

    # ── 最终保存
    save_checkpoint(checkpoint)
    save_partial_results(all_income, all_balance, all_profile,
                         all_cashflow, all_mktcap)

    log("=" * 60)
    log(f"STEP 01 COMPLETE | 成功 {len(checkpoint['done'])} | 失败 {len(failed_codes)}")
    if failed_codes:
        log(f"失败列表：{failed_codes[:20]}...")
    log("=" * 60)


if __name__ == "__main__":
    main()
