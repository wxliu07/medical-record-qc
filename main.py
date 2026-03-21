import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent  # 获取当前脚本所在目录的绝对路径
CONFIG_DIR = BASE_DIR / "config"  # 配置文件存放目录
DATA_FILE = BASE_DIR / "data" / "simulate_data.json"  # 模拟数据文件路径
OUTPUT_DIR = BASE_DIR / "outputs"  # 处理结果输出根目录

# 将项目根目录添加到系统路径，确保能够导入自定义模块
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.case_generation import generate_physical_summary, generate_standard_medical_record
from modules.data_process.data_loader import load_config, load_simulated_reports
from modules.dataset import aggregate_grade_dataset, build_graded_record
from modules.ner_re import extract_by_report_type
from modules.quality_control import run_llm_reasoning_qc, run_rule_based_qc


def _load_env_file(env_path: Path) -> None:
    """
    加载环境变量配置文件(.env)
    
    功能：从指定路径读取.env文件，解析键值对并设置到环境变量中
    格式支持：忽略空行和注释行(#)，支持去除引号，避免覆盖已存在的环境变量
    
    参数:
        env_path: .env文件的路径对象
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # 跳过空行、注释行或格式不正确的行
        if not line or line.startswith("#") or "=" not in line:
            continue

        # 解析键值对，处理可能的引号包裹
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 仅在环境变量未设置时赋值
        if key and key not in os.environ:
            os.environ[key] = value


def _apply_env_model_overrides(model_config: Dict[str, Any]) -> None:
    """
    应用环境变量对模型配置的覆盖
    
    功能：从环境变量读取DEEPSEEK_MODEL和DEEPSEEK_BASE_URL，
          如果存在则更新模型配置，实现运行时配置动态调整
    
    参数:
        model_config: 模型配置字典，将被直接修改
    """
    llm_cfg = model_config.setdefault("llm", {})  # 获取或创建llm配置子字典
    env_model = os.getenv("DEEPSEEK_MODEL", "").strip()
    env_base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()

    # 环境变量优先级高于配置文件
    if env_model:
        llm_cfg["model"] = env_model
    if env_base_url:
        llm_cfg["base_url"] = env_base_url


def _save_json(data: Any, file_path: Path) -> None:
    """
    保存数据为JSON文件
    
    功能：将Python对象序列化为JSON格式并写入指定路径，
          自动创建不存在的父目录，使用UTF-8编码确保中文兼容
    
    参数:
        data: 要序列化的Python对象
        file_path: 输出文件路径
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_pipeline() -> Dict[str, Any]:
    """
    主处理流程函数
    
    功能：执行完整的报告处理流水线
    步骤：
        1. 初始化配置和环境
        2. 加载模拟报告数据
        3. 对每份报告进行实体提取和质量控制
        4. 生成分级数据集和摘要文档
        5. 保存所有处理结果
    
    返回:
        包含处理结果摘要的字典
    """

    _load_env_file(BASE_DIR / ".env")  # 加载环境变量

    # 加载配置文件并应用环境变量覆盖
    cfg = load_config(CONFIG_DIR)
    qc_rules = cfg["qc_rules"]  # 质量控制规则配置
    model_config = cfg["model_config"]  # 模型参数配置
    _apply_env_model_overrides(model_config)


    reports = load_simulated_reports(DATA_FILE)  # 加载模拟报告数据


    graded_records: List[Dict[str, Any]] = []  # 存储分级后的记录
    medical_records: List[Dict[str, Any]] = []  # 存储标准病历
    physical_summaries: List[Dict[str, Any]] = []  # 存储体检摘要

    for report in reports:
        # 4.1 实体识别与提取：根据报告类型使用NLP提取关键信息
        extraction = extract_by_report_type(report, qc_rules, model_config)
        
        # 4.2 基于规则的质量控制：检查提取结果是否符合预设规则
        rule_issues = run_rule_based_qc(report, extraction, qc_rules)
        
        # 4.3 基于LLM推理的质量控制：利用大模型进行深度质量评估
        reasoning = run_llm_reasoning_qc(report, extraction, rule_issues, model_config)

        # 4.4 构建分级记录：整合原始报告、提取结果和质量问题
        graded = build_graded_record(report, extraction, rule_issues, reasoning)
        graded_records.append(graded)

        # 4.5 生成文档摘要：基于分级记录生成标准化的体检摘要和病历
        is_corrected = False  # 标记是否为修正后的版本
        summary = generate_physical_summary(graded, corrected=is_corrected)
        med_record = generate_standard_medical_record(graded, corrected=is_corrected)

        # 4.6 保存生成的文档，关联原始报告ID
        physical_summaries.append({"report_id": report["report_id"], **summary})
        medical_records.append({"report_id": report["report_id"], **med_record})


    graded_dataset = aggregate_grade_dataset(graded_records)  # 聚合所有分级记录


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳用于目录命名
    out_dir = OUTPUT_DIR / timestamp  # 创建本次运行的输出目录

    # 保存各类处理结果
    _save_json(graded_dataset, out_dir / "graded_dataset.json")          # 分级数据集
    _save_json(physical_summaries, out_dir / "physical_summaries.json")  # 体检摘要
    _save_json(medical_records, out_dir / "medical_records.json")        # 标准病历


    result = {
        "output_dir": str(out_dir),  # 输出目录路径
        "summary": graded_dataset["summary"],  # 处理统计摘要
        "total_reports": len(reports),  # 总报告数
        "degraded_count": sum(1 for r in graded_records if r.get("extraction", {}).get("degraded")),  # 降级处理报告数
    }
    _save_json(result, out_dir / "run_summary.json")  # 保存运行摘要
    return result


if __name__ == "__main__":
    run_result = run_pipeline()
    print(json.dumps(run_result, ensure_ascii=False, indent=2))