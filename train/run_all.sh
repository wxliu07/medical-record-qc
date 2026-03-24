#!/bin/bash
# =============================================================================
# CCKS 2019 Medical NER Pipeline - 一键运行脚本
# =============================================================================
#
# 使用方法:
#   bash run_all.sh
#
# 执行流程:
#   1. 数据预处理
#   2. 模型训练
#   3. 模型评估
#   4. 示例推理
#
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python() {
    log_info "Checking Python environment..."
    if command -v python &> /dev/null; then
        python_version=$(python --version 2>&1)
        log_info "Python version: $python_version"
    else
        log_error "Python not found. Please install Python 3.11+"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    log_info "Checking dependencies..."
    python -c "import torch; import transformers; import datasets; import seqeval" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_info "All dependencies installed"
    else
        log_warn "Missing dependencies. Run: pip install -r requirements.txt"
    fi
}

# 主流程
main() {
    log_info "=============================================="
    log_info "CCKS 2019 Medical NER Pipeline"
    log_info "=============================================="

    # 环境检查
    check_python
    check_dependencies

    # 切换到脚本目录
    cd "$(dirname "$0")"

    # Step 1: 数据预处理
    log_info "=============================================="
    log_info "Step 1: Data Preprocessing"
    log_info "=============================================="
    python data_preprocess.py
    if [ $? -ne 0 ]; then
        log_error "Data preprocessing failed"
        exit 1
    fi
    log_info "Data preprocessing completed"

    # Step 2: 模型训练
    log_info "=============================================="
    log_info "Step 2: Model Training"
    log_info "=============================================="
    python train_ner.py
    if [ $? -ne 0 ]; then
        log_error "Model training failed"
        exit 1
    fi
    log_info "Model training completed"

    # Step 3: 模型评估
    log_info "=============================================="
    log_info "Step 3: Model Evaluation"
    log_info "=============================================="
    python evaluate.py
    if [ $? -ne 0 ]; then
        log_error "Model evaluation failed"
        exit 1
    fi
    log_info "Model evaluation completed"

    # Step 4: 示例推理
    log_info "=============================================="
    log_info "Step 4: Example Inference"
    log_info "=============================================="
    python inference.py --example
    if [ $? -ne 0 ]; then
        log_error "Example inference failed"
        exit 1
    fi
    log_info "Example inference completed"

    log_info "=============================================="
    log_info "All steps completed successfully!"
    log_info "=============================================="
    log_info "Results saved to:"
    log_info "  - train/models/ner/best_model/ (trained model)"
    log_info "  - train/results/evaluation_report.txt (evaluation report)"
    log_info "  - train/results/ner_f1_score.png (F1 score chart)"
}

# 运行
main "$@"
