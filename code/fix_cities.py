"""
fix_cities.py — 修正 profile_partial.csv 中的城市识别问题
- 改进正则：处理开发区/自贸区/自治州等特殊地址
- 手动覆盖：香港注册但实际总部在大陆的公司
- 输出修正后的 profile_partial.csv（覆盖原文件）
"""

import pandas as pd
import numpy as np
import re
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, "data/raw")

PROF_PATH = os.path.join(RAW, "profile_partial.csv")

# ── 手动覆盖：已知香港注册但主业在大陆的公司（总部城市）
MANUAL_CITY_OVERRIDE = {
    "600938": "北京",   # 中国海油（注册香港，总部北京）
    "600941": "北京",   # 中国移动（注册香港，总部北京）
}

# 中国主要城市名（用于在地址中做关键词匹配）
MAJOR_CITIES = [
    "北京", "上海", "天津", "重庆",
    "广州", "深圳", "杭州", "苏州", "南京", "宁波", "合肥", "德清",
    "成都", "重庆", "西安", "武汉", "长沙", "郑州", "济南", "青岛",
    "沈阳", "哈尔滨", "长春", "石家庄", "太原", "西宁", "兰州",
    "昆明", "贵阳", "南宁", "海口", "拉萨", "乌鲁木齐", "呼和浩特",
    "银川", "西宁", "福州", "厦门", "南昌", "武汉", "长沙",
    "南京", "无锡", "常州", "镇江", "扬州", "徐州", "南通",
    "温州", "嘉兴", "湖州", "绍兴", "金华", "台州", "丽水",
    "烟台", "威海", "潍坊", "淄博", "泰安", "临沂", "东营",
    "佛山", "东莞", "珠海", "惠州", "汕头", "湛江", "中山",
    "无锡", "苏州", "南通", "盐城", "泰州",
    "绵阳", "德阳", "南充", "宜宾", "泸州",
    "洛阳", "开封", "新乡", "焦作",
    "宜昌", "襄阳", "荆州",
    "株洲", "湘潭", "衡阳",
    "芜湖", "蚌埠", "马鞍山",
    "泉州", "漳州", "莆田",
    "南昌", "赣州", "九江",
    "太原", "大同", "运城",
    "鞍山", "大连", "抚顺", "本溪", "营口",
    "唐山", "保定", "邯郸", "邢台",
    "梅州", "张家港", "东阿", "马尔康",
    "岳阳", "常德", "益阳", "娄底",
    "遵义", "毕节",
    "大理", "玉溪", "曲靖",
    "防城港", "柳州", "桂林", "北海",
    "三亚", "儋州",
    "西宁", "海东",
    "嘉峪关", "金昌", "白银",
    "张掖", "酒泉",
    "延安", "铜川", "宝鸡", "渭南", "咸阳", "榆林", "商洛",
    "齐齐哈尔", "牡丹江", "佳木斯", "大庆",
    "通化", "延边", "四平",
]


def extract_city_v2(address):
    """改进版城市提取：处理开发区/自贸区/自治州等特殊地址格式"""
    if not address or pd.isna(address):
        return None
    address = str(address).strip()

    # ── 1. 直辖市（以直辖市名开头）
    for muni in ["北京", "上海", "天津", "重庆"]:
        if address.startswith(muni):
            return muni

    # ── 2. 香港/澳门（境外注册，返回 None，由手动覆盖处理）
    if address.startswith("香港") or address.startswith("澳门"):
        return None

    # ── 3. 自贸区（格式：中国（城市）自由贸易试验区...）
    m = re.search(r'中国[（(]([^）)]+)[）)]自由贸易试验区', address)
    if m:
        inner = m.group(1)
        for muni in ["北京", "上海", "天津", "重庆"]:
            if muni in inner:
                return muni
        city_m = re.search(r'(.{2,4})(市|区)', inner)
        if city_m:
            return city_m.group(1)
        if inner:
            return inner.strip()

    # ── 4. 标准省份地址：省...市
    m = re.search(r'省(.{2,8}?)市', address)
    if m:
        city = m.group(1)
        # 去掉省名本身出现在城市名里的情况
        city = re.sub(r'(自治州|地区|盟|开发区|高新区|工业园|经济区)', '', city)
        if 2 <= len(city) <= 6:
            return city.strip()

    # ── 5. 省份后直接跟开发区/工业园/高新区（没有「市」字）
    m = re.search(r'省(.{2,6}?)(?:开发区|高新|工业园|技术区|经济区|新区|保税区)', address)
    if m:
        # 尝试从中提取城市名
        prefix = m.group(1)
        # 在 prefix 里找主要城市名
        for city in MAJOR_CITIES:
            if city in prefix or city in address[:30]:
                return city

    # ── 6. 自治区
    m = re.search(r'(?:自治区|壮族自治区|回族自治区|维吾尔自治区)(.{2,6}?)(?:市|地区|州|盟)', address)
    if m:
        return m.group(1).strip()

    # ── 7. 兜底：在全地址里找主要城市关键词
    for city in MAJOR_CITIES:
        if city in address:
            return city

    return None


def main():
    df = pd.read_csv(PROF_PATH, encoding="utf-8-sig")
    print(f"读入 {len(df)} 行")
    print(f"原始城市空值：{df['city'].isna().sum()}")

    # 重新运行改进版提取
    df["city_v2"] = df["city_raw"].apply(extract_city_v2)

    # 用手动覆盖
    df["code_str"] = df["code"].astype(str).str.zfill(6)
    for code, city in MANUAL_CITY_OVERRIDE.items():
        mask = df["code_str"] == code
        if mask.any():
            df.loc[mask, "city_v2"] = city
            print(f"  手动覆盖: {code} → {city}")

    # 比较改进前后
    was_null = df["city"].isna()
    now_fixed = was_null & df["city_v2"].notna()
    print(f"\n改进后新增识别: {now_fixed.sum()} 家")
    print(f"改进后仍为空: {(was_null & df['city_v2'].isna()).sum()} 家")

    # 更新 city 列：优先用 v2，若 v2 为空且原来有值则保留原值
    df["city"] = df["city_v2"].where(df["city_v2"].notna(), df["city"])
    df = df.drop(columns=["city_v2", "code_str"])

    print(f"\n最终城市空值：{df['city'].isna().sum()}")

    # 看治理城市的修复情况
    zones = pd.read_csv(os.path.join(BASE, "data/treatment/ai_zones.csv"), encoding="utf-8-sig")
    zone_cities = set(zones["city"].tolist())
    print("\n治理城市在 profile 中的公司数（修复后）:")
    for city in sorted(zone_cities):
        cnt = (df["city"] == city).sum()
        if cnt > 0:
            print(f"  {city}: {cnt}")

    df.to_csv(PROF_PATH, index=False, encoding="utf-8-sig")
    print(f"\n已保存修正后的 profile_partial.csv")


if __name__ == "__main__":
    main()
