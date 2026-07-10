"""
03_variables.py — 在 panel_data.csv 基础上补充分析变量
输出: data/clean/analysis_data.csv
"""
import os
import numpy as np
import pandas as pd

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean")
TREAT = os.path.join(BASE, "data/treatment")

df = pd.read_csv(os.path.join(CLEAN, "panel_data.csv"), encoding="utf-8-sig", low_memory=False)
print(f"[03] 读入面板: {df.shape}")

# ── 1. 被解释变量（02 已算好，这里确保字段名正确）
# RD = rd_expense / total_assets（02 已生成）
# CAPEX = capex / total_assets（02 已生成）
# RD_staff：研发人员比例（02 未抓取，全 NaN）
if "RD_staff" not in df.columns:
    df["RD_staff"] = np.nan

# ── 2. 控制变量（02 已算好：Size, Lev, ROA, Age, Cash, TobinQ）
# 02 中 cash 列即货币资金，Cash = cash / total_assets
# TobinQ = (total_mktcap + total_liabilities) / total_assets

# ── 3. SOE 哑变量（来自巨潮资讯实际控制人数据）
soe_path = os.path.join(TREAT, "soe_status.csv")
if os.path.exists(soe_path):
    soe_df = pd.read_csv(soe_path, encoding="utf-8-sig", dtype={"stkcd": str})
    soe_df["stkcd"] = soe_df["stkcd"].str.zfill(6)
    df["stkcd_tmp"] = df["code"].astype(str).str.zfill(6)
    df = df.merge(soe_df, left_on="stkcd_tmp", right_on="stkcd", how="left", suffixes=("", "_soe"))
    df["SOE"] = df["SOE"].fillna(0).astype(int)
    df = df.drop(columns=["stkcd_tmp"], errors="ignore")
    soe_rate = df.groupby("code")["SOE"].first().mean()
    print(f"[03] SOE 比例（巨潮实际控制人数据）: {soe_rate:.1%}")
else:
    df["SOE"] = 0
    print("[03] ⚠️  soe_status.csv 不存在，SOE 设为 0")

# ── 4. 行业 AI 暴露度（HighAI）
HIGH_AI_INDUSTRIES = [
    "计算机", "电子", "通信", "传媒", "医药生物", "机械设备", "汽车",
]
industry_col = "industry" if "industry" in df.columns else None
if industry_col:
    df["HighAI"] = df[industry_col].astype(str).apply(
        lambda x: 1 if any(ind in x for ind in HIGH_AI_INDUSTRIES) else 0
    )
    df["industry_sw"] = df[industry_col]   # 统一别名，方便后续脚本使用
else:
    df["HighAI"] = 0
    df["industry_sw"] = ""
    print("[03] ⚠️  industry 缺失，HighAI 全部设为 0")

# ── 5. ever_treated（公司是否属于处理组）
df["ever_treated"] = df["treat_year"].notna().astype(int)
print(f"[03] ever_treated=1 公司数: {df[df['ever_treated']==1]['code'].nunique()}")
print(f"[03] ever_treated=0 公司数: {df[df['ever_treated']==0]['code'].nunique()}")

# ── 6. 省份变量（用于稳健性检验的省份×年份FE）
zones = pd.read_csv(os.path.join(TREAT, "ai_zones.csv"), encoding="utf-8-sig")
city2prov = dict(zip(zones["city"], zones["province"]))

# 补充主要省份映射（非治理城市）
CITY_PROV_EXTRA = {
    "广州": "广东", "东莞": "广东", "佛山": "广东", "珠海": "广东",
    "惠州": "广东", "中山": "广东", "汕头": "广东", "湛江": "广东",
    "温州": "浙江", "嘉兴": "浙江", "湖州": "浙江", "绍兴": "浙江",
    "金华": "浙江", "台州": "浙江", "丽水": "浙江",
    "无锡": "江苏", "常州": "江苏", "镇江": "江苏", "扬州": "江苏",
    "徐州": "江苏", "南通": "江苏", "盐城": "江苏", "泰州": "江苏",
    "烟台": "山东", "威海": "山东", "潍坊": "山东", "淄博": "山东",
    "泰安": "山东", "临沂": "山东", "东营": "山东",
    "绵阳": "四川", "德阳": "四川", "南充": "四川", "宜宾": "四川",
    "洛阳": "河南", "开封": "河南", "新乡": "河南", "焦作": "河南",
    "宜昌": "湖北", "襄阳": "湖北", "荆州": "湖北",
    "株洲": "湖南", "湘潭": "湖南", "衡阳": "湖南", "岳阳": "湖南",
    "芜湖": "安徽", "蚌埠": "安徽", "马鞍山": "安徽",
    "泉州": "福建", "漳州": "福建", "厦门": "福建", "福州": "福建",
    "南昌": "江西", "赣州": "江西", "九江": "江西",
    "太原": "山西", "大同": "山西", "运城": "山西",
    "石家庄": "河北", "唐山": "河北", "保定": "河北", "邯郸": "河北",
    "大连": "辽宁", "鞍山": "辽宁", "抚顺": "辽宁",
    "哈尔滨": "黑龙江", "齐齐哈尔": "黑龙江", "大庆": "黑龙江",
    "长春": "吉林", "通化": "吉林", "四平": "吉林",
    "昆明": "云南", "曲靖": "云南",
    "贵阳": "贵州", "遵义": "贵州",
    "南宁": "广西", "柳州": "广西", "桂林": "广西",
    "海口": "海南", "三亚": "海南",
    "兰州": "甘肃",
    "西宁": "青海",
    "银川": "宁夏",
    "乌鲁木齐": "新疆",
    "呼和浩特": "内蒙古",
    "拉萨": "西藏",
}
city2prov.update(CITY_PROV_EXTRA)

# 直辖市=省份
for city in ["北京", "上海", "天津", "重庆"]:
    city2prov[city] = city

city_col = "city_clean" if "city_clean" in df.columns else None
if city_col:
    df["province"] = df[city_col].map(city2prov)
    coverage = df["province"].notna().mean()
    print(f"[03] 省份识别覆盖率: {coverage:.1%}")
    df["province_year"] = df["province"].astype(str) + "_" + df["year"].astype(str)
else:
    df["province"] = np.nan
    df["province_year"] = np.nan
    print("[03] ⚠️  city_clean 缺失，province 全为 NaN")

# ── 7. 相对时间变量（事件研究）
if "treat_year" in df.columns:
    df["rel_time"] = df["year"] - df["treat_year"]
    df.loc[df["treat_year"].isna(), "rel_time"] = -999
    df["rel_time_clipped"] = df["rel_time"].apply(
        lambda x: max(-3, min(4, x)) if x != -999 else -999
    )
    for k in [-3, -2, 0, 1, 2, 3, 4]:
        col = f"D_m{abs(k)}" if k < 0 else f"D_{k}"
        df[col] = ((df["rel_time_clipped"] == k) & (df["rel_time"] != -999)).astype(int)

# ── 7b. 政府补贴变量（营业外收入代理）
subsidy_path = os.path.join(BASE, "data/raw/subsidy_partial.csv")
if os.path.exists(subsidy_path):
    sub_df = pd.read_csv(subsidy_path, encoding="utf-8-sig")
    sub_df["code"] = sub_df["code"].astype(str).str.zfill(6)
    sub_df = sub_df.rename(columns={"non_op_income": "gov_subsidy_raw"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.merge(sub_df[["code","year","gov_subsidy_raw"]], on=["code","year"], how="left")
    # 标准化：补贴 / 总资产（Subsidy intensity）
    df["Subsidy"] = df["gov_subsidy_raw"] / df["total_assets"]
    df["Subsidy"] = df["Subsidy"].clip(df["Subsidy"].quantile(0.01),
                                        df["Subsidy"].quantile(0.99))
    cov = df["Subsidy"].notna().mean()
    print(f"[03] 政府补贴覆盖率: {cov:.1%} | 均值: {df['Subsidy'].mean():.4f}")
else:
    df["Subsidy"] = np.nan
    print("[03] ⚠️  subsidy_partial.csv 不存在，Subsidy 设为 NaN")

# ── 8. 交叉项（异质性分析）
df["Treat_HighAI"] = df["Treat"] * df["HighAI"]

# ── 9. 统一公司 ID（firm_id）
df["stkcd"] = df["code"].astype(str).str.zfill(6)
df["firm_id"] = pd.factorize(df["stkcd"])[0] + 1

# ── 10. 再次 Winsorize（02 已做过，这里保留以防重新计算）
CONT_VARS = ["RD", "CAPEX", "RD_staff", "Size", "Lev", "ROA", "Age", "Cash", "TobinQ"]
for var in CONT_VARS:
    if var not in df.columns or df[var].isna().all():
        continue
    lo = df[var].quantile(0.01)
    hi = df[var].quantile(0.99)
    df[var] = df[var].clip(lo, hi)

# ── 11. 删除核心控制变量缺失的观测
CORE_CONTROLS = ["Size", "Lev", "ROA", "Age", "Cash"]  # TobinQ/SOE 允许缺失时降级
n_before = len(df)
df = df.dropna(subset=[v for v in CORE_CONTROLS if v in df.columns])
print(f"[03] 删除核心控制变量缺失: -{n_before - len(df)} 条 → {len(df)}")

# TobinQ 缺失时补 0 并加缺失标记（避免样本过少）
if "TobinQ" in df.columns:
    miss_tq = df["TobinQ"].isna().mean()
    print(f"[03] TobinQ 缺失率: {miss_tq:.1%}")
    if miss_tq > 0.5:
        print("      ⚠️  TobinQ 缺失率>50%，将从主回归控制变量中剔除")
        df["TobinQ_missing"] = 1  # 标记
    else:
        df["TobinQ_missing"] = 0
        df = df.dropna(subset=["TobinQ"])

# ── 12. 输出
out_path = os.path.join(CLEAN, "analysis_data.csv")
os.makedirs(CLEAN, exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8-sig")

n_firms = df["stkcd"].nunique()
n_vars = len(df.columns)
print(f"\n[STEP 3 COMPLETE] 变量构建完成 | N = {len(df)} | 公司数 = {n_firms} | 变量数 K = {n_vars}")
print(f"  输出: {out_path}")
print(f"  处理组公司数: {df[df['ever_treated']==1]['stkcd'].nunique()}")
print(f"  对照组公司数: {df[df['ever_treated']==0]['stkcd'].nunique()}")
tq_miss = df['TobinQ'].isna().mean() if 'TobinQ' in df.columns else 1.0
print(f"  TobinQ 缺失率: {tq_miss:.1%}（等待 01b 完成后可重跑补全）")
