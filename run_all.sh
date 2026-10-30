#!/bin/bash
# =====================================================
# 一键运行全部流程（从清洗到输出）
# 前置：01_fetch_data.py 和 01b_fetch_prices.py 已经跑完
# 用法：chmod +x run_all.sh && ./run_all.sh
# =====================================================

set -e

CODE_DIR="$(cd "$(dirname "$0")" && pwd)/code"
LOG_FILE="$(cd "$(dirname "$0")" && pwd)/run_all.log"

echo "======================================"
echo "开始运行 | $(date)"
echo "日志输出至：$LOG_FILE"
echo "======================================"

run_step() {
    local step=$1
    local script=$2
    echo ""
    echo "------ $step: $(date) ------"
    python3 "$CODE_DIR/$script" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "⚠️  $step 报错，但继续执行后续步骤"
    else
        echo "$step 完成: $(date)"
    fi
}

# Step 00: 初始化（约 2 分钟）
run_step "STEP 00 - 环境初始化" "00_setup.py"

# Step 01: 数据抓取（支持断点续跑，已经在跑则可跳过）
# run_step "STEP 01 - 数据抓取" "01_fetch_data.py"
# run_step "STEP 01b - 年末收盘价抓取" "01b_fetch_prices.py"
echo ">>> [提示] Step 01/01b 在后台运行中，完成后再执行 Step 02+ <<<"

# Step 02: 清洗（约 5 分钟）
run_step "STEP 02 - 数据清洗与面板构建" "02_clean_merge.py"

# Step 03: 变量构建（约 1 分钟）
run_step "STEP 03 - 变量构建" "03_variables.py"

# Step 04: 描述性统计 Table 1
run_step "STEP 04 - 描述性统计" "04_descriptive.py"

# Step 05: 主回归 Table 2
run_step "STEP 05 - 主回归" "05_main_regression.py"

# Step 05c: 插补估计与扩展双向固定效应。没有 analysis_data.csv 时只打印说明并退出。
run_step "STEP 05c - 异质性稳健估计" "05c_modern_did.py"

# Step 06: 平行趋势检验 Figure 1
run_step "STEP 06 - 事件研究图" "06_event_study.py"

# Step 07: 异质性分析 Table 3
run_step "STEP 07 - 异质性分析" "07_heterogeneity.py"

# Step 08: 稳健性检验 Table 4
run_step "STEP 08 - 稳健性检验" "08_robustness.py"

# Step 09: 汇总报告
run_step "STEP 09 - 汇总报告" "09_summary_report.py"

echo ""
echo "======================================"
echo "全部完成！$(date)"
echo "输出文件在 output/ 目录"
echo "======================================"
