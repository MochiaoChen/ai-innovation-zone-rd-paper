# 国家新一代人工智能创新发展试验区政策与企业研发投入

**National AI Innovation Zones and Corporate R&D Investment: Evidence from Staggered Difference-in-Differences**

本项目利用 2013–2025 年 4,875 家 A 股上市公司面板数据（44,594 条公司-年度观测），采用多期双重差分（Staggered DID）方法，评估中国"国家新一代人工智能创新发展试验区"政策对企业研发投入的微观影响。

论文完整正文见 [`output/paper/main.pdf`](output/paper/main.pdf) / [`main.tex`](output/paper/main.tex)。

## 核心发现

- **总体平均效应不显著**：双向固定效应估计显示政策对全样本研发强度无显著平均效应（系数 = 0.04pp，t = 1.38）。
- **但批次间异质性极强**：2019 年首批（精准试点）与 2021 年第三批的 ATT 分别为 0.23pp（t=2.38）和 0.13pp（t=2.46），显著为正；而占处理组 51% 的 2020 年大规模铺开批次效果为零（t=0.18），拉低了整体均值——揭示"精准有效、铺开稀释"的规律。
- **行业异质性**：效应集中于高 AI 暴露行业（交互项系数=0.0028，t=3.74），低 AI 暴露行业无反应，印证吸收能力假说。
- **机制**：拒绝直接补贴渠道（Treat 对政府补贴系数显著为负），支持融资约束缓解渠道（政策对高杠杆企业的促进效应显著更强），表明政策通过信用背书而非直接发钱起作用。
- 通过平行趋势检验、PSM-DID、省份×年份固定效应、剔除直辖市、安慰剂检验（500 次置换）等多组稳健性检验。

详见 [`FINDINGS.md`](FINDINGS.md)（核心发现总结）与 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)。

## 目录结构

```
.
├── code/                  # 数据抓取、清洗、变量构建、回归、绘图脚本（按编号顺序执行）
├── output/
│   ├── paper/             # 论文 LaTeX 源码与编译后 PDF
│   ├── figures/           # 所有图表（含中英文双版本）
│   └── tables/            # 回归结果表格（.tex）
├── run_all.sh             # 一键运行 Step 02–09（不含数据抓取）
├── FINDINGS.md            # 核心发现总结
└── RESULTS_SUMMARY.md     # 结果摘要
```

## 数据说明

本仓库**不包含**原始数据（`data/` 目录）。原始数据来自同花顺（iFinD）、巨潮资讯与 baostock 等商业/公开数据源，其分发受相应服务条款限制，因此未随代码开源。

如需复现，请：
1. 自行申请同花顺 iFinD / 巨潮资讯数据接口权限，baostock 无需申请。
2. 依次运行 `code/00_setup.py`、`code/01_fetch_data.py`、`code/01b_fetch_prices.py` 等抓取脚本获取原始数据并存至 `data/raw/`。
3. 运行 `./run_all.sh` 完成清洗、变量构建、回归与出图全流程。

## 方法论

- 基准模型：企业固定效应 + 年份固定效应的双向固定效应（TWFE）DID
- 动态效应：事件研究法（event-study），验证平行趋势假设
- 针对 TWFE 在交错处理下的负权重偏误，采用分批次"干净比较"估计量（clean cohort ATT，参考 Callaway & Sant'Anna 2021、Sun & Abraham 2021 的队列分解思想）

## 作者

陈沫樵（Mochiao Chen），对外经济贸易大学中国金融学院鸿儒 2401 班本科生
