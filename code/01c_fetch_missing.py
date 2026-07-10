"""
Step 01c: 补全缺失市值数据（只抓01b未成功的公司）
结果追加到 mktcap_partial.csv / prices_partial.csv
"""

import baostock as bs
import pandas as pd
import os, time
from datetime import datetime
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, "data/raw")
YEARS = list(range(2013, 2026))
N_WORKERS = 4  # 减少并发，降低连接错误率


def code_to_bs(code):
    c = str(code).zfill(6)
    if c.startswith("6") or c.startswith("5"):
        return f"sh.{c}"
    elif c.startswith("8") or c.startswith("9"):
        return f"bj.{c}"
    else:
        return f"sz.{c}"


def fetch_batch(codes):
    for attempt in range(3):
        lg = bs.login()
        if lg.error_code == "0":
            break
        time.sleep(2)
    else:
        return []

    results = []
    for code in codes:
        bs_code = code_to_bs(code)
        for retry in range(2):
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,close,volume,turn",
                    start_date="2013-01-01", end_date="2025-12-31",
                    frequency="m", adjustflag="3",
                )
                df = rs.get_data()
                if df is None or df.empty:
                    break
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
                    close  = float(last["close"])
                    vol    = float(last["volume"])
                    turn   = float(last["turn"])
                    mktcap = close * vol / (turn / 100)
                    results.append({"code": code, "year": yr,
                                    "close_price": close, "total_mktcap": mktcap})
                break
            except Exception:
                time.sleep(1)

    bs.logout()
    return results


def main():
    missing_path = os.path.join(RAW, "missing_codes.csv")
    codes = pd.read_csv(missing_path)["code"].astype(str).apply(lambda x: x.zfill(6)).tolist()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 补全 {len(codes)} 家公司（{N_WORKERS}进程）", flush=True)

    chunk = len(codes) // N_WORKERS + 1
    batches = [codes[i:i+chunk] for i in range(0, len(codes), chunk)]

    with Pool(processes=N_WORKERS) as pool:
        all_results = pool.map(fetch_batch, batches)

    rows = [r for batch in all_results for r in batch]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 本次获取 {len(rows)} 条，{pd.DataFrame(rows)['code'].nunique() if rows else 0} 家", flush=True)

    if not rows:
        print("✗ 无数据返回", flush=True)
        return

    new_df = pd.DataFrame(rows)

    # 追加到已有CSV
    mktcap_path  = os.path.join(RAW, "mktcap_partial.csv")
    prices_path  = os.path.join(RAW, "prices_partial.csv")

    old_mktcap = pd.read_csv(mktcap_path, encoding="utf-8-sig")
    old_prices = pd.read_csv(prices_path, encoding="utf-8-sig")

    merged_mktcap = pd.concat([old_mktcap, new_df[["code","year","total_mktcap"]]], ignore_index=True)
    merged_prices = pd.concat([old_prices, new_df[["code","year","close_price"]]], ignore_index=True)

    # 去重（保留最新）
    merged_mktcap = merged_mktcap.drop_duplicates(subset=["code","year"], keep="last")
    merged_prices = merged_prices.drop_duplicates(subset=["code","year"], keep="last")

    merged_mktcap.to_csv(mktcap_path, index=False, encoding="utf-8-sig")
    merged_prices.to_csv(prices_path, index=False, encoding="utf-8-sig")

    total_cos = merged_mktcap["code"].nunique()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 合并后总计 {total_cos} 家公司", flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 01c COMPLETE", flush=True)


if __name__ == "__main__":
    main()
