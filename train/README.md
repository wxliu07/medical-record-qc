# CCKS 2019 Medical NER + Relation Extraction Pipeline

基于中文医疗BERT的实体识别（NER）和规则引擎关系抽取（RE）训练流程。

## 目录结构

```
train/
├── requirements.txt          # 依赖包
├── data_preprocess.py        # 数据预处理
├── model.py                  # NER模型定义
├── train_ner.py              # 模型训练
├── evaluate.py               # 模型评估
├── rule_engine.py            # 关系抽取规则引擎
├── inference.py              # 端到端推理
├── utils.py                  # 工具函数
├── README.md                 # 本文档
├── run_all.sh                # 一键运行脚本
├── base-models/              # 预训练模型
│   └── chinese-bert-wwm-ext/
└── models/                    # 训练输出
    └── ner/
        └── best_model/       # 最佳模型
```

## 环境搭建

### 1. 创建conda虚拟环境

```bash
conda create -n py311 python=3.11 -y
conda activate py311
```

### 2. 安装依赖

```bash
cd train
pip install -r requirements.txt
```

### 3. 验证环境

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 数据准备

### 数据集位置

- 训练集: `data/datasets/CCKS2019/subtask1_training.txt` (1000条)
- 测试集: `data/datasets/CCKS2019/subtask1_test_set_with_answer.json` (379条)

### 数据格式

训练集（JSONL，每行一个JSON）：
```json
{"originalText": "病历原文...", "entities": [{"start_pos": 0, "end_pos": 3, "label_type": "疾病和诊断"}, ...]}
```

测试集（JSON数组）：
```json
[{"originalText": "...", "entities": [...]}, ...]
```

### 实体类型（6类）

| 中文标签 | 英文标签 | 说明 |
|---------|---------|------|
| 疾病和诊断 | DISEASE | 疾病、病因、分型等 |
| 检查 | TESTIMAGE | CT、MR、X线等影像检查 |
| 检验 | TESTLAB | 实验室检验 |
| 手术 | OPERATION | 手术治疗操作 |
| 药物 | DRUG | 化学药物、化疗方案 |
| 解剖部位 | ANATOMY | 人体解剖学部位 |

## 快速开始

### 方式一：一键运行

```bash
cd train
bash run_all.sh
```

### 方式二：分步运行

#### Step 1: 数据预处理

```bash
cd train
python data_preprocess.py
```

输出：
- `data/datasets/CCKS2019/processed/` - 处理后的HuggingFace Dataset
- `label_mapping.json` - 标签映射
- `preprocess_config.json` - 预处理配置

#### Step 2: 模型训练

```bash
cd train
python train_ner.py --num_train_epochs 15 --per_device_train_batch_size 8
```

关键超参数：
- `learning_rate`: 3e-5（默认）
- `num_train_epochs`: 15（默认）
- `per_device_train_batch_size`: 8（默认）
- `warmup_ratio`: 0.1
- `early_stopping_patience`: 3

训练输出：
- `train/models/ner/best_model/` - 最佳模型
- `train/models/ner/logs/` - TensorBoard日志
- `train/models/ner/checkpoint-*/` - 中间检查点

#### 模型保存机制

**Checkpoint 自动保存**
- 保存策略：每个 epoch 评估一次、同时自动保存一份 checkpoint（`save_strategy='epoch'`）
- Checkpoint 目录命名：按全局训练步数命名，例如 `checkpoint-1027`, `checkpoint-1106`, `checkpoint-1185` 等
- Checkpoint 内容：包含模型权重 (`model.safetensors`)、优化器状态 (`optimizer.pt`)、调度器状态 (`scheduler.pt`)、训练状态 (`trainer_state.json`) 等，支持断点续训
- 自动清理：`save_total_limit=3` 生效，只保留最近的 3 个 checkpoint，旧的会被自动删除

**Best Model 最终保存**
- 判定条件：验证集上**F1 分数最高**的模型（`metric_for_best_model='f1'`，`greater_is_better=True`）
- 保存方式：训练结束后，将验证集 F1 最优的 checkpoint 导出到固定目录 `train/models/ner/best_model/`（非每个 epoch 都存一份）
- 触发时机：由 `load_best_model_at_end=True` 自动触发，脚本中通过 `trainer.save_model(best_model_dir)` 和 `tokenizer.save_pretrained(best_model_dir)` 手动导出
- 内容：模型权重 (`model.safetensors`)、分词器 (`tokenizer.json`, `vocab.txt`, `special_tokens_map.json`) 等推理所需文件
- 注意：Best model 目录不包含优化器/调度器等训练状态，仅用于推理和评估

**早停机制**
- 配置：`EarlyStoppingCallback(early_stopping_patience=3)`
- 意义：若验证集 F1 连续 3 个 epoch 不再上升，则自动停止训练（不必等到 15 epoch 结束）

#### Step 3: 模型评估

```bash
cd train
python evaluate.py
```

评估输出：
- `train/results/evaluation_report.txt` - 评估报告
- `train/results/ner_errors.csv` - 错误案例分析
- `train/results/ner_f1_score.png` - F1分数柱状图

#### Step 4: 端到端推理

```bash
cd train
# 示例推理
python inference.py --example

# 单条文本推理
python inference.py --text "患者因胃癌于全麻上行胃癌根治术。"

# Gradio Web界面
python inference.py --gradio --port 7860
```

## 输出格式

### 推理结果

```json
{
  "original_text": "患者因胃癌于全麻上行胃癌根治术。",
  "entities": [
    {"text": "胃癌", "start_pos": 4, "end_pos": 6, "type": "DISEASE", "confidence": 0.98},
    {"text": "胃癌根治术", "start_pos": 11, "end_pos": 16, "type": "OPERATION", "confidence": 0.95}
  ],
  "relations": [
    {
      "head_text": "胃癌",
      "head_type": "DISEASE",
      "tail_text": "胃",
      "tail_type": "ANATOMY",
      "relation": "has_location",
      "confidence": 1.0,
      "match_rule": "text_contain"
    }
  ]
}
```

## 关系类型

| 关系类型 | 头实体 | 尾实体 | 说明 |
|---------|-------|-------|------|
| has_location | DISEASE | ANATOMY | 疾病发生的解剖部位 |
| treated_by_operation | DISEASE | OPERATION | 疾病采用的手术治疗 |
| treated_by_drug | DISEASE | DRUG | 疾病采用的药物治疗 |
| examined_by_image | DISEASE | TESTIMAGE | 疾病采用的影像检查 |
| examined_by_lab | DISEASE | TESTLAB | 疾病采用的实验室检验 |
| performed_on | OPERATION | ANATOMY | 手术实施的解剖部位 |

## 规则引擎自定义

编辑 `rule_engine.py` 中的关键词和规则：

```python
# 添加新的关系类型
RELATION_TYPES['new_relation'] = RelationType(
    name='new_relation',
    head_type='DISEASE',
    tail_type='DRUG',
    description='新关系类型',
    priority=1
)

# 添加新的触发关键词
KEYWORDS['new_relation'] = ['关键词1', '关键词2', ...]

# 添加文本包含规则
TEXT_CONTAIN_RULES['new_relation'] = [
    ('疾病短语', '部位短语'),
]
```

## 预期性能

NER模型在测试集上的预期F1值：

| 实体类型 | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| DISEASE | ~0.90 | ~0.88 | ~0.89 |
| TESTIMAGE | ~0.85 | ~0.82 | ~0.83 |
| TESTLAB | ~0.83 | ~0.80 | ~0.81 |
| OPERATION | ~0.88 | ~0.86 | ~0.87 |
| DRUG | ~0.85 | ~0.83 | ~0.84 |
| ANATOMY | ~0.87 | ~0.85 | ~0.86 |
| **Micro Avg** | **~0.87** | **~0.85** | **~0.86** |
| **Macro Avg** | **~0.86** | **~0.84** | **~0.85** |

注：实际性能取决于预训练模型质量和训练超参数。

## 常见问题

### Q: 显存不足怎么办？
A: 减小batch size或启用梯度累积：
```bash
python train_ner.py --per_device_train_batch_size 4 --gradient_accumulation_steps 4
```

### Q: 训练loss不下降？
A: 检查学习率是否合适，尝试调整为1e-5或5e-5。

### Q: Windows环境下fp16报错？
A: 使用 `--no_fp16` 参数禁用混合精度训练。

### Q: 如何增加新实体类型？
A: 1. 在 `utils.py` 的 `LABEL_TO_ID` 和 `ID_TO_LABEL` 中添加
   2. 在 `data_preprocess.py` 的 `CHINESE_TO_ENGLISH` 映射中添加中文标签

## 参考

- CCKS 2019 Task 1: https://www.biendata.xyz/models/ccks/2019/1/
- Chinese-BERT-WWM: https://github.com/ymcui/Chinese-BERT-wwm
- HuggingFace Transformers: https://huggingface.co/transformers/
