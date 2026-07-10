"""
Step 01b: 抓取年末收盘价 + 流通市值
- baostock 月线，取12月末数据
- 8进程并行，预计15分钟
"""

import baostock as bs
import pandas as pd
import numpy as np
import os, json, time
from datetime import datetime
from multiprocessing import Pool, Manager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, "data/raw")
CKPT_PATH = os.path.join(BASE, "data/raw/checkpoint_prices.json")
YEARS = list(range(2013, 2026))
N_WORKERS = 8


def code_to_bs(code):
    c = str(code).zfill(6)
    if c.startswith("6") or c.startswith("5"):
        return f"sh.{c}"
    elif c.startswith("8") or c.startswith("9"):
        return f"bj.{c}"
    else:
        return f"sz.{c}"


def fetch_batch(codes):
    """每个进程独立登录 baostock，抓一批公司"""
    lg = bs.login()
    if lg.error_code != "0":
        return []

    results = []
    for code in codes:
        bs_code = code_to_bs(code)
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, "date,close,volume,turn",
                start_date="2013-01-01", end_date="2025-12-31",
                frequency="m", adjustflag="3",
            )
            df = rs.get_data()
            if df is None or df.empty:
                continue
            df["date"]  = pd.to_datetime(df["date"])
            df["year"]  = df["date"].dt.year
            df["month"] = df["date"].dt.month
            for col in ["close", "volume", "turn"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[(df["month"] == 12) & (df["turn"] > 0)].dropna(subset=["close","volume","turn"])
            for yr in YEARS:
                sub = df[df["year"] == yr]
                if sub.empty:
                    continue
                last = sub.iloc[-1]
                close = float(last["close"])
                vol   = float(last["volume"])
                turn  = float(last["turn"])
                mktcap = close * vol / (turn / 100)
                results.append({
                    "code": code, "year": yr,
                    "close_price": close, "total_mktcap": mktcap,
                })
        except Exception:
            pass

    bs.logout()
    return results


def main():
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] STEP 01b 开始（{N_WORKERS}进程并行）", flush=True)

    codes = (
        pd.read_csv(os.path.join(RAW, "profile_partial.csv"), encoding="utf-8-sig")["code"]
        .dropna().astype(str).apply(lambda x: x.strip().zfill(6)).tolist()
    )
    print(f"[{ts}] 共 {len(codes)} 家公司", flush=True)

    # 分成 N_WORKERS 份
    chunk = len(codes) // N_WORKERS + 1
    batches = [codes[i:i+chunk] for i in range(0, len(codes), chunk)]

    with Pool(processes=N_WORKERS) as pool:
        all_results = pool.map(fetch_batch, batches)

    rows = [r for batch in all_results for r in batch]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 共获取 {len(rows)} 条记录", flush=True)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RAW, "prices_partial.csv"), index=False, encoding="utf-8-sig",
                  columns=["code","year","close_price"])
        df.to_csv(os.path.join(RAW, "mktcap_partial.csv"), index=False, encoding="utf-8-sig",
                  columns=["code","year","total_mktcap"])
        success = df["code"].nunique()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 01b COMPLETE | 成功 {success} 家公司", flush=True)
    else:
        print("✗ 无数据返回", flush=True)


if __name__ == "__main__":
    main()
