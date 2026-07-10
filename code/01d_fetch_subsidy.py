"""
01d: 抓营业外收入（政府补贴代理变量）- THS接口
多进程 + checkpoint，约30-40分钟
"""
import akshare as ak
import pandas as pd, numpy as np, re, os, json, time
from datetime import datetime
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, "data/raw")
CKPT = os.path.join(RAW, "checkpoint_subsidy.json")
YEARS = set(range(2013, 2026))
N_WORKERS = 4


def parse_val(val_str):
    if pd.isna(val_str) or str(val_str).strip() in ("", "-", "--", "—"):
        return np.nan
    s = str(val_str).strip()
    mult = 1e8 if "亿" in s else (1e4 if "万" in s else 1)
    num = re.sub(r"[^\d\.\-]", "", s)
    return float(num) * mult if num else np.nan


def fetch_one(code_6):
    time.sleep(0.3)
    try:
        df = ak.stock_financial_benefit_ths(symbol=code_6, indicator="按年度")
        if df is None or df.empty:
            return []
        rows = []
        for col in ["加：营业外收入", "其中：政府补贴", "其他收益"]:
            if col in df.columns:
                for yr, val in zip(df["报告期"], df[col]):
                    try:
                        yr_int = int(yr)
                    except:
                        continue
                    if yr_int not in YEARS:
                        continue
                    v = parse_val(val)
                    if not np.isnan(v):
                        rows.append({"code": code_6, "year": yr_int,
                                     "field": col, "non_op_income": v})
        return rows
    except Exception:
        return []


def fetch_batch(codes):
    all_rows = []
    for code in codes:
        rows = fetch_one(code)
        all_rows.extend(rows)
    return all_rows


def main():
    # 读取所有代码
    profile = pd.read_csv(os.path.join(RAW, "profile_partial.csv"), encoding="utf-8-sig")
    all_codes = profile["code"].dropna().astype(str).apply(
        lambda x: x.strip().zfill(6)).tolist()

    # checkpoint：已完成的
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            done = set(json.load(f).get("done", []))
    codes = [c for c in all_codes if c not in done]
    print(f"[{datetime.now():%H:%M}] 待抓: {len(codes)} / {len(all_codes)} 家", flush=True)

    # 分批（每批50家，4进程并行，批次间sleep避免封IP）
    batch_size = len(codes) // N_WORKERS + 1
    batches = [codes[i:i+batch_size] for i in range(0, len(codes), batch_size)]

    all_rows = []
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(fetch_batch, batches)
    for r in results:
        all_rows.extend(r)

    # 追加/合并
    out_path = os.path.join(RAW, "subsidy_partial.csv")
    new_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["code","year","field","non_op_income"])

    if os.path.exists(out_path):
        old = pd.read_csv(out_path, encoding="utf-8-sig")
        new_df = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
            subset=["code","year","field"], keep="last")

    # 每家公司取最优字段（优先级：政府补贴 > 其他收益 > 营业外收入）
    priority = {"其中：政府补贴": 0, "其他收益": 1, "加：营业外收入": 2}
    new_df["priority"] = new_df["field"].map(priority).fillna(9)
    new_df = (new_df.sort_values("priority")
                     .drop_duplicates(subset=["code","year"], keep="first")
                     .drop(columns=["priority","field"]))

    new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_cos = new_df["code"].nunique()
    print(f"[{datetime.now():%H:%M}] 完成！{n_cos} 家公司有营业外收入/补贴数据", flush=True)
    print(f"共 {len(new_df)} 条记录，已保存 → {out_path}", flush=True)

    # 更新checkpoint
    with open(CKPT, "w") as f:
        json.dump({"done": all_codes}, f)


if __name__ == "__main__":
    main()
