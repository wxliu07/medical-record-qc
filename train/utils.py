# -*- coding: utf-8 -*-
"""
工具函数模块 - 全流程复用
CCKS 2019 Medical NER Training Pipeline
"""

import random
import numpy as np
import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# =============================================================================
# 随机种子设置 - 保证结果完全可复现
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """
    固定全局所有随机种子，保证实验可复现性

    Args:
        seed: 随机种子值，默认42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


# =============================================================================
# BIO标签与实体列表转换
# =============================================================================
def bio_to_entities(
    text: str,
    bio_labels: List[str],
    confidences: Optional[List[float]] = None
) -> List[Dict]:
    """
    将BIO标签序列转换为结构化实体列表

    Args:
        text: 原始病历文本
        bio_labels: BIO标签序列，与文本字符对齐
        confidences: 每个标签的置信度（可选）

    Returns:
        实体列表，每项包含text, start_pos, end_pos, type, confidence
    """
    entities = []
    i = 0
    n = len(bio_labels)

    while i < n:
        label = bio_labels[i]

        # 跳过O标签和非标签
        if not label.startswith('B-'):
            i += 1
            continue

        # 提取实体类型
        entity_type = label[2:]

        # 找到实体的结束位置（连续I-标签的末尾）
        j = i + 1
        while j < n and bio_labels[j] == f'I-{entity_type}':
            j += 1

        # 提取实体文本（使用原始文本切片）
        start_pos = i
        end_pos = j

        # 置信度处理
        confidence = 1.0
        if confidences is not None:
            # 取实体范围内置信度的最大值
            confidence = max(confidences[i:j]) if j > i else confidences[i]

        # 构建实体
        entity = {
            'text': text[start_pos:end_pos],
            'start_pos': start_pos,
            'end_pos': end_pos,
            'type': entity_type,
            'confidence': float(confidence)
        }
        entities.append(entity)

        i = j

    return entities


# =============================================================================
# 字符级标签与BERT Token对齐
# =============================================================================
def align_labels_with_tokens(
    char_labels: List[str],
    tokenizer,
    word_ids: List[int]
) -> List[int]:
    """
    将字符级BIO标签对齐到BERT token级

    对于子词token（word_id为None或重复），标签设为-100（损失忽略）
    对于CLS/SEP/Padding，标签设为-100

    Args:
        char_labels: 字符级BIO标签列表
        tokenizer: BERT分词器
        word_ids: token对应的word_ids

    Returns:
        token级标签列表
    """
    token_labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # CLS, SEP, PAD等特殊token → 忽略
            token_labels.append(-100)
        elif word_id != prev_word_id:
            # 单词的第一个子词 → 使用该单词的标签
            if word_id < len(char_labels):
                token_labels.append(char_labels[word_id])
            else:
                token_labels.append(-100)
        else:
            # 单词的后续子词 → 忽略（已由第一个子词处理）
            token_labels.append(-100)

        prev_word_id = word_id

    return token_labels


# =============================================================================
# 中文病历分句
# =============================================================================
def split_sentences(text: str) -> List[str]:
    """
    按中文标点将病历文本分句

    句子结束符：。！？；
    保留分句后的文本片段

    Args:
        text: 原始病历文本

    Returns:
        句子列表
    """
    # 按句子结束符分割
    sentences = re.split(r'[。！？；]', text)

    # 过滤空句子，保留有效文本
    result = [s.strip() for s in sentences if s.strip()]

    return result


# =============================================================================
# 类别权重计算 - 缓解类别不平衡
# =============================================================================
def compute_class_weights(
    dataset,
    label_column: str = 'labels',
    label_list: Optional[List[str]] = None
) -> List[float]:
    """
    计算类别权重，用于加权CrossEntropyLoss

    基于训练集中各类别样本数量的倒数，缓解'O'标签占比过高问题

    Args:
        dataset: HuggingFace Dataset对象
        label_column: 标签列名
        label_list: 预定义的标签列表（按顺序）

    Returns:
        类别权重列表，与label_list顺序对应
    """
    from collections import Counter

    # 统计各类别样本数
    all_labels = []
    for item in dataset:
        labels = item[label_column]
        # 过滤-100（忽略的标签）
        filtered = [l for l in labels if l != -100]
        all_labels.extend(filtered)

    counter = Counter(all_labels)

    # 确定标签列表顺序
    if label_list is None:
        # 从数据中推断标签列表（假设是ID列表）
        unique_labels = sorted(set(all_labels))
    else:
        unique_labels = label_list

    # 计算权重：总数/（类别数 × 该类别数量）
    total = len(all_labels)
    num_classes = len(unique_labels)

    weights = []
    for label in unique_labels:
        count = counter.get(label, 1)  # 避免除零
        weight = total / (num_classes * count)
        weights.append(weight)

    # 归一化
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    return normalized_weights


# =============================================================================
# JSON文件读写工具
# =============================================================================
def save_json(data, file_path: str | Path) -> None:
    """
    保存数据为JSON文件

    Args:
        data: 要保存的数据（需可序列化）
        file_path: 文件路径
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: str | Path) -> any:
    """
    从JSON文件加载数据

    Args:
        file_path: 文件路径

    Returns:
        加载的数据
    """
    file_path = Path(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_training_data(file_path: str | Path) -> List[Dict]:
    """加载JSONL训练数据（每行一个JSON对象）。"""
    file_path = Path(file_path)
    data: List[Dict] = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def load_test_data(file_path: str | Path) -> List[Dict]:
    """
    加载测试数据，兼容JSON数组与JSONL两种格式。

    某些数据文件虽然后缀是.json，但内容实际是逐行JSON对象。
    """
    file_path = Path(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except json.JSONDecodeError:
        data: List[Dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data


# =============================================================================
# 滑动窗口实体合并 - 推理时去重
# =============================================================================
def merge_overlapping_entities(
    entities: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    合并重叠窗口预测的实体，去重并保留最高置信度

    合并策略：
    1. 先按(文本, 类型, 位置)完全相同的实体去重
    2. 对于重叠实体（相同类型、位置接近），保留置信度最高的

    Args:
        entities: 实体列表
        iou_threshold: IoU阈值，超过则认为是同一实体

    Returns:
        合并后的实体列表
    """
    if not entities:
        return []

    # 按置信度降序排序
    sorted_entities = sorted(entities, key=lambda x: x.get('confidence', 1.0), reverse=True)

    merged = []
    used_indices = set()

    for i, entity in enumerate(sorted_entities):
        if i in used_indices:
            continue

        # 查找与当前实体重叠的其他实体
        overlapped = [entity]
        used_indices.add(i)

        for j, other in enumerate(sorted_entities):
            if j in used_indices:
                continue

            # 检查是否应该合并
            if should_merge(entity, other):
                overlapped.append(other)
                used_indices.add(j)

        # 从重叠实体中选择最佳（置信度最高且位置最完整的）
        best = select_best_entity(overlapped)
        merged.append(best)

    # 最终去重：完全相同的(文本, 类型, 位置)
    final = []
    seen = set()
    for e in merged:
        key = (e['text'], e['type'], e['start_pos'], e['end_pos'])
        if key not in seen:
            seen.add(key)
            final.append(e)

    return final


def should_merge(e1: Dict, e2: Dict) -> bool:
    """
    判断两个实体是否应该合并

    合并条件：类型相同且位置高度重叠
    """
    if e1['type'] != e2['type']:
        return False

    # 计算字符位置重叠程度
    start1, end1 = e1['start_pos'], e1['end_pos']
    start2, end2 = e2['start_pos'], e2['end_pos']

    # 计算交集
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)

    if union == 0:
        return False

    iou = intersection / union

    return iou >= 0.5


def select_best_entity(entities: List[Dict]) -> Dict:
    """
    从重叠实体列表中选择最佳实体

    选择标准：
    1. 置信度最高
    2. 位置信息最完整（end_pos - start_pos 较大，即更长的实体）
    """
    if len(entities) == 1:
        return entities[0]

    # 优先选择置信度最高的
    best = max(entities, key=lambda x: (x.get('confidence', 1.0), x['end_pos'] - x['start_pos']))

    return best


# =============================================================================
# 标签映射管理
# =============================================================================
LABEL_TO_ID = {
    'O': 0,
    'B-DISEASE': 1,
    'I-DISEASE': 2,
    'B-TESTIMAGE': 3,
    'I-TESTIMAGE': 4,
    'B-TESTLAB': 5,
    'I-TESTLAB': 6,
    'B-OPERATION': 7,
    'I-OPERATION': 8,
    'B-DRUG': 9,
    'I-DRUG': 10,
    'B-ANATOMY': 11,
    'I-ANATOMY': 12,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# 实体类型标签（不含BIO前缀）
ENTITY_TYPES = ['DISEASE', 'TESTIMAGE', 'TESTLAB', 'OPERATION', 'DRUG', 'ANATOMY']

# 中文标签类型到英文的映射
CHINESE_TO_ENGLISH = {
    '疾病和诊断': 'DISEASE',
    '检查': 'TESTIMAGE',
    '影像检查': 'TESTIMAGE',
    '检验': 'TESTLAB',
    '实验室检验': 'TESTLAB',
    '手术': 'OPERATION',
    '药物': 'DRUG',
    '解剖部位': 'ANATOMY',
}


def get_label_id(label: str) -> int:
    """获取标签ID"""
    return LABEL_TO_ID.get(label, 0)


def get_id_label(id: int) -> str:
    """根据ID获取标签"""
    return ID_TO_LABEL.get(id, 'O')


def normalize_entity_type(entity_type: str) -> str:
    """
    标准化实体类型名称

    将中文标签类型转换为英文标签类型
    """
    return CHINESE_TO_ENGLISH.get(entity_type, entity_type)


# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT / 'data' / 'datasets' / 'CCKS2019'
MODEL_ROOT = PROJECT_ROOT / 'train' / 'base-models' / 'chinese-bert-wwm-ext'
OUTPUT_ROOT = PROJECT_ROOT / 'train'
PROCESSED_DATA_DIR = DATA_ROOT / 'processed'
MODEL_OUTPUT_DIR = OUTPUT_ROOT / 'models' / 'ner'
RESULTS_DIR = OUTPUT_ROOT / 'results'
