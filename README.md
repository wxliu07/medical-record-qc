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

## outputs 文件说明

### 目录层级
- outputs 下每个子目录（例如 20260324_213358）代表一次完整流水线运行。
- 子目录名是运行时间戳，格式为 YYYYMMDD_HHMMSS。
- 同一时间戳目录内保存该次运行的全部结果，方便回溯和对比。

### run_summary.json（运行摘要）
作用：用于快速查看本次运行是否成功、处理了多少报告、分级分布和降级情况。

主要字段：
- output_dir：本次输出目录绝对路径。
- summary：分级统计，通常包含 正常级、缺失级、矛盾级 的数量。
- total_reports：本次处理的报告总数。
- degraded_count：抽取阶段发生降级的报告数（例如模型不可用、超时后走 fallback）。

### graded_dataset.json（核心分级数据集）
作用：保存每条报告从原始内容到抽取、质控、推理、最终分级的完整中间信息，是最重要的审计文件。

顶层字段：
- summary：本次分级汇总。
- records：逐条报告记录数组。

records 中每条通常包含：
- report_id、report_type、report_subtype：报告标识与类型。
- original_label：输入数据原始标签。
- grade_label：系统最终分级标签。
- content：原始三段内容（描述、检查所见、检查提示）。
- extraction：NER/RE 抽取结果。
	- entities：抽取实体（指标类常见 指标 列表，影像类常见 nodes/edges 对应实体信息）。
	- relations：抽取关系。
	- source：抽取来源（例如 llm_ie、train_ner_imaging、regex_fallback、fallback_imaging_parser）。
	- degraded：是否降级。
	- error：降级或失败原因（若有）。
- qc_issues：规则质控发现的问题列表（缺失/矛盾等）。
- reasoning：LLM 推理质控结果（result、reason、source）。

### physical_summaries.json（体检总结结果）
作用：面向展示的简要体检总结输出。

每条通常包含：
- report_id：报告 ID。
- generated：是否生成成功。
- summary：生成的体检总结文本（generated 为 true 时非空）。
- note：未生成原因说明（例如存在质控问题，需先修正）。

### medical_records.json（标准化病历结果）
作用：面向业务落地的结构化病历输出。

每条通常包含：
- report_id：报告 ID。
- generated：是否生成正式病历。
- record：标准化病历主体（患者信息、检查结果、异常提示、建议等）。
- pending_issues：未通过质控时的待处理问题清单。
- note：未生成正式病历的原因说明。

### 四个文件之间的关系
- run_summary.json：看总体运行结果。
- graded_dataset.json：看全量明细与可追溯证据。
- physical_summaries.json：看面向体检摘要的输出。
- medical_records.json：看面向标准病历的输出。

建议排查顺序：
1. 先看 run_summary.json 判断是否大面积降级。
2. 再看 graded_dataset.json 定位具体报告的 extraction、qc_issues、reasoning。
3. 最后看 physical_summaries.json 和 medical_records.json 的 generated 与 note，判断是否可直接用于展示/下游。

## 分级逻辑
- 正常级：无缺失且无逻辑矛盾
- 缺失级：存在关键字段缺失
- 矛盾级：存在逻辑矛盾或指标异常

优先级：矛盾级 > 缺失级 > 正常级

## 降级策略
- 指标类：llm-ie 不可用或 API Key 缺失时，自动使用规则/正则抽取
- 影像类：radgraph 不可用时，自动使用占位解析器抽取 nodes/edges
- LLM 推理质控失败时，返回 result=unknown 并附失败原因