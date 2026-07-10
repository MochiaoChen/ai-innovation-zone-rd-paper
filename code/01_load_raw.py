"""
01_load_raw.py — 读取所有原始 CSV，统一编码，初步检查字段完整性
"""
import os
import sys
import chardet
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data/raw")

# 预期文件及其核心字段（中英文字段名映射在此定义，后续步骤统一使用英文列名）
EXPECTED_FILES = {
    "basic_info.csv": {
        "desc": "上市公司基本情况",
        "key_cols_zh": ["股票代码", "公司名称", "注册城市", "省份", "上市日期", "实际控制人类型"],
        "rename": {
            "股票代码": "stkcd",
            "公司名称": "company_name",
            "注册城市": "reg_city",
            "省份": "province",
            "上市日期": "ipo_date",
            "实际控制人类型": "controller_type",
        },
    },
    "balance_sheet.csv": {
        "desc": "资产负债表",
        "key_cols_zh": ["股票代码", "会计期间", "总资产", "总负债", "货币资金", "所有者权益"],
        "rename": {
            "股票代码": "stkcd",
            "会计期间": "period",
            "总资产": "total_assets",
            "总负债": "total_liabilities",
            "货币资金": "monetary_fund",
            "所有者权益": "equity",
        },
    },
    "income_stmt.csv": {
        "desc": "利润表",
        "key_cols_zh": ["股票代码", "会计期间", "净利润", "营业收入"],
        "rename": {
            "股票代码": "stkcd",
            "会计期间": "period",
            "净利润": "net_income",
            "营业收入": "revenue",
        },
    },
    "cashflow.csv": {
        "desc": "现金流量表",
        "key_cols_zh": ["股票代码", "会计期间"],
        "rename": {
            "股票代码": "stkcd",
            "会计期间": "period",
        },
        "capex_candidates": [
            "购建固定资产无形资产和其他长期资产支付的现金",
            "购建固定资产、无形资产和其他长期资产支付的现金",
            "购建固定资产无形资产支付现金",
            "资本支出",
        ],
    },
    "rd_expense.csv": {
        "desc": "研发投入",
        "key_cols_zh": ["股票代码", "会计期间", "研发费用"],
        "rename": {
            "股票代码": "stkcd",
            "会计期间": "period",
            "研发费用": "rd_expense",
            "研发人员数量": "rd_employee",
            "研发人员": "rd_employee",
            "员工总数": "total_employee",
            "员工人数": "total_employee",
        },
    },
    "stock_price.csv": {
        "desc": "股票市场",
        "key_cols_zh": ["股票代码", "年度"],
        "rename": {
            "股票代码": "stkcd",
            "年度": "year",
            "总市值": "market_cap",
            "流通市值": "float_mktcap",
            "年末收盘价": "close_price",
        },
    },
}


def detect_encoding(filepath):
    with open(filepath, "rb") as f:
        raw = f.read(50000)
    result = chardet.detect(raw)
    enc = result.get("encoding", "gbk") or "gbk"
    # 常见别名修正
    enc_map = {"GB2312": "gbk", "ascii": "utf-8"}
    return enc_map.get(enc, enc)


def load_csv(filepath):
    enc = detect_encoding(filepath)
    for encoding in [enc, "gbk", "gb2312", "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            print(f"  读取成功 [{encoding}]: {os.path.basename(filepath)} — {df.shape}")
            return df
        except Exception:
            continue
    print(f"  ⚠️  无法读取: {filepath}")
    return None


def fuzzy_rename(df, rename_map):
    """列名模糊匹配：忽略空格、全半角差异"""
    col_map = {}
    for col in df.columns:
        col_clean = col.strip().replace(" ", "").replace("　", "")
        for zh, en in rename_map.items():
            if zh.replace(" ", "") in col_clean or col_clean in zh.replace(" ", ""):
                col_map[col] = en
                break
    return df.rename(columns=col_map)


frames = {}
field_report = []

for fname, meta in EXPECTED_FILES.items():
    fpath = os.path.join(RAW, fname)
    if not os.path.exists(fpath):
        print(f"[MISSING] {fname} — 请将文件放入 data/raw/")
        field_report.append({"file": fname, "status": "MISSING", "shape": None, "columns": []})
        continue

    df = load_csv(fpath)
    if df is None:
        field_report.append({"file": fname, "status": "READ_ERROR", "shape": None, "columns": []})
        continue

    # 特殊处理：cashflow 的资本支出列名多样
    if fname == "cashflow.csv":
        for candidate in meta.get("capex_candidates", []):
            matched = [c for c in df.columns if candidate in c]
            if matched:
                meta["rename"][matched[0]] = "capex"
                print(f"  cashflow capex 列识别为: {matched[0]}")
                break
        else:
            print("  ⚠️  cashflow.csv 未找到资本支出列，请检查字段名")

    df = fuzzy_rename(df, meta["rename"])

    # 年份标准化
    if "period" in df.columns:
        df["year"] = pd.to_datetime(df["period"].astype(str), errors="coerce").dt.year
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 股票代码标准化（去掉后缀、统一6位）
    if "stkcd" in df.columns:
        df["stkcd"] = df["stkcd"].astype(str).str.extract(r"(\d{6})")[0]

    frames[fname] = df
    field_report.append({
        "file": fname,
        "status": "OK",
        "shape": df.shape,
        "columns": list(df.columns),
    })

# 输出字段报告
print("\n" + "=" * 60)
print("字段完整性报告")
print("=" * 60)
for r in field_report:
    status = r["status"]
    shape = r["shape"]
    print(f"[{status}] {r['file']}  shape={shape}")
    if status == "OK":
        print(f"  列名: {r['columns']}")

# 保存中间帧供后续步骤使用
import pickle
with open(os.path.join(BASE, "data/clean/raw_frames.pkl"), "wb") as f:
    pickle.dump(frames, f)

n_ok = sum(1 for r in field_report if r["status"] == "OK")
print(f"\n[STEP 1 COMPLETE] 原始数据读取 | 成功 {n_ok}/{len(EXPECTED_FILES)} 个文件")
