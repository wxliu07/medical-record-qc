# Medical QC Framework

## 项目目标
基于 Python 搭建检查报告语言类质控到病历生成的工程化框架，覆盖：
- 数据模拟
- 分类型 NER/RE（指标类与影像类）
- 规则质控 + LLM 推理质控
- 分级数据集生成
- 体检总结与标准化病历生成

## 目录结构

```text
electronic-record/
├── config/
│   ├── qc_rules.json
│   └── model_config.json
├── data/
│   └── simulate_data.json
├── modules/
│   ├── ner_re/
│   │   ├── __init__.py
│   │   ├── indicator_ner_re.py
│   │   └── imaging_ner_re.py
│   ├── quality_control/
│   │   ├── __init__.py
│   │   ├── rule_based_qc.py
│   │   └── llm_reasoning_qc.py
│   ├── data_process/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── grade_dataset.py
│   └── case_generation/
│       ├── __init__.py
│       ├── physical_summary.py
│       └── medical_record.py
├── outputs/
├── .env
├── main.py
└── README.md
```

## 环境要求
- conda 环境: py311
- Python 3.11

## 安装依赖

```bash
pip install llm-ie
https://github.com/daviden1013/llm-ie
https://github.com/quqxui/Awesome-LLM4IE-Papers
```

可选影像解析：
```bash
pip install radgraph
https://github.com/Stanford-AIMI/radgraph
```

## 环境变量
将 DeepSeek API Key 设置到环境变量：

```bash
set DEEPSEEK_API_KEY=your_key
```

## 运行方式

```bash
python main.py
```

运行后会在 outputs 时间戳目录下生成：
- graded_dataset.json
- physical_summaries.json
- medical_records.json
- run_summary.json

## 分级逻辑
- 正常级：无缺失且无逻辑矛盾
- 缺失级：存在关键字段缺失
- 矛盾级：存在逻辑矛盾或指标异常

优先级：矛盾级 > 缺失级 > 正常级

## 降级策略
- 指标类：llm-ie 不可用或 API Key 缺失时，自动使用规则/正则抽取
- 影像类：radgraph 不可用时，自动使用占位解析器抽取 nodes/edges
- LLM 推理质控失败时，返回 result=unknown 并附失败原因