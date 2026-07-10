# RESULTS_SUMMARY.md

生成时间：2026-06-25 17:12

---

## 1. 样本统计

- 总观测数 N = **44,594**
- 公司数 = **4,875**
- 年份范围 = 2013 — 2025
- 处理组公司数 = **2,424**
- 对照组公司数 = **2,451**

批次分布（公司数）：
  - 2019 年批次：425 家公司
  - 2020 年批次：1167 家公司
  - 2021 年批次：630 家公司
  - 2022 年批次：202 家公司

## 2. 关键变量摘要

- **RD Intensity**：Mean=0.0254, Std=0.0240, P50=0.0207
- **CAPEX Intensity**：Mean=0.0465, Std=0.0446, P50=0.0334
- **Treatment**：Mean=0.2840, Std=0.4509, P50=0.0000
- **Tobin's Q**：Mean=1.8665, Std=1.3135, P50=1.4513

## 3. 主回归系数（Table 2）

- **RD**：Treat = 0.0004 (t=1.377, N=32,910)
- **CAPEX**：Treat = -0.0007 (t=-0.781, N=44,474)

## 4. 输出文件

### Tables

- `table1_descriptive.csv` (1.2 KB)
- `table1_descriptive.tex` (1.3 KB)
- `table2_main.tex` (1.1 KB)
- `table3_heterogeneity.csv` (0.3 KB)
- `table3_heterogeneity.tex` (1.4 KB)
- `table4_robustness.csv` (0.2 KB)
- `table4_robustness.tex` (1.1 KB)

### Figures

- `figure1_event_capex.pdf` (18.0 KB)
- `figure1_event_capex.png` (231.7 KB)
- `figure1_event_rd.pdf` (23.0 KB)
- `figure1_event_rd.png` (206.5 KB)
- `figure2_placebo.pdf` (19.4 KB)
- `figure2_placebo.png` (108.6 KB)

## 5. 用户核实清单（来自 instruction.md §12）

- [ ] `data/treatment/ai_zones.csv` 城市名单已与科技部官网核实
- [ ] Table 1 样本量是否在 3 万—4 万 firm-year 范围内（当前 N=44,594）
- [ ] Figure 1 冲击前各期系数（k = -3, -2）是否不显著（平行趋势成立）
- [ ] Table 2 中 Treat 系数符号是否正向（政策促进研发）
- [ ] TobinQ 缺失率过高时，考虑重跑 `01b_fetch_prices.py` 并更新分析数据
- [ ] SOE 哑变量当前为关键词估算，建议用官方控制人数据替换
- [ ] 是否需要补充行业×年份固定效应版本