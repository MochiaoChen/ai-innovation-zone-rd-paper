"""
Step 00: 环境初始化
- 创建目录结构
- 写入 AI 试验区城市名单
- 安装依赖
"""

import subprocess
import sys
import os
import pandas as pd

# ── 安装依赖 ────────────────────────────────────────────────
print("[00] 安装依赖包...")
packages = ["akshare", "pandas", "numpy", "pyfixest",
            "statsmodels", "scikit-learn", "matplotlib",
            "seaborn", "scipy", "openpyxl", "tqdm", "chardet"]
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "--break-system-packages", "-q"] + packages)
print("[00] 依赖安装完成")

# ── 创建目录 ─────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ["data/raw", "data/treatment", "data/clean",
          "output/tables", "output/figures", "code"]:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)
print("[00] 目录结构就绪")

# ── AI 试验区城市名单 ─────────────────────────────────────────
# 来源：科技部官网，请用户自行核实第 3、4 批次日期
zones = [
    # 城市, 省份, 批次, 年份
    ("北京",   "北京",  1, 2019),
    ("上海",   "上海",  2, 2020),
    ("天津",   "天津",  2, 2020),
    ("深圳",   "广东",  2, 2020),
    ("杭州",   "浙江",  2, 2020),
    ("合肥",   "安徽",  2, 2020),
    ("德清",   "浙江",  2, 2020),
    ("重庆",   "重庆",  3, 2021),
    ("成都",   "四川",  3, 2021),
    ("西安",   "陕西",  3, 2021),
    ("济南",   "山东",  3, 2021),
    ("青岛",   "山东",  3, 2021),
    ("武汉",   "湖北",  3, 2021),
    ("长沙",   "湖南",  3, 2021),
    ("郑州",   "河南",  3, 2021),
    ("苏州",   "江苏",  3, 2021),
    ("沈阳",   "辽宁",  3, 2021),
    ("南京",   "江苏",  4, 2022),
    ("宁波",   "浙江",  4, 2022),
]

df_zones = pd.DataFrame(zones, columns=["city", "province", "batch", "treat_year"])
out_path = os.path.join(BASE, "data/treatment/ai_zones.csv")
df_zones.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[00] AI 试验区名单已写入：{out_path}")
print(f"     ⚠ 请核实科技部官网，尤其是第 3、4 批次年份")
print(df_zones.to_string(index=False))
print("\n[00] STEP 00 COMPLETE")
