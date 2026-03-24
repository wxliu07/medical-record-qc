# -*- coding: utf-8 -*-
"""
数据预处理模块
CCKS 2019 Medical NER Training Pipeline

功能：
- 读取CCKS2019数据集（JSONL训练集 + JSON测试集）
- 字符级BIO标签转换
- 滑动窗口处理长文本
- WordPiece标签对齐
- 数据集划分（9:1训练/验证）
- 保存为HuggingFace Dataset格式
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast

from utils import (
    set_seed, save_json, load_json, PROJECT_ROOT, DATA_ROOT,
    PROCESSED_DATA_DIR, MODEL_ROOT, LABEL_TO_ID, ID_TO_LABEL,
    normalize_entity_type, get_label_id
)


# =============================================================================
# 配置
# =============================================================================
@dataclass
class PreprocessConfig:
    """预处理配置"""
    # 数据路径
    train_file: Path = DATA_ROOT / 'subtask1_training.txt'
    test_file: Path = DATA_ROOT / 'subtask1_test_set_with_answer.json'

    # 模型路径
    model_path: Path = MODEL_ROOT

    # 分词器配置
    max_length: int = 512  # BERT最大长度
    stride: int = 128      # 滑动窗口步长

    # 数据集划分
    train_val_split: float = 0.1  # 验证集比例

    # 随机种子
    seed: int = 42

    # 输出路径
    output_dir: Path = PROCESSED_DATA_DIR


# =============================================================================
# 数据加载
# =============================================================================
def load_training_data(file_path: Path) -> List[Dict]:
    """
    加载训练数据（JSONL格式）

    每行一个JSON对象：{"originalText": "...", "entities": [...]}

    Args:
        file_path: 训练文件路径

    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} training examples")
    return data


def load_test_data(file_path: Path) -> List[Dict]:
    """
    加载测试数据（JSON数组格式）

    Args:
        file_path: 测试文件路径

    Returns:
        数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    if not raw:
        data = []
    else:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                data = parsed
            elif isinstance(parsed, dict):
                data = [parsed]
            else:
                data = []
        except json.JSONDecodeError:
            # 兼容JSONL格式（每行一个JSON对象）
            data = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    print(f"Loaded {len(data)} test examples")
    return data


# =============================================================================
# BIO标签转换
# =============================================================================
def text_to_bio_labels(
    text: str,
    entities: List[Dict]
) -> List[str]:
    """
    将实体标注转换为字符级BIO标签

    Args:
        text: 原始病历文本
        entities: 实体列表，每项包含start_pos, end_pos, label_type

    Returns:
        BIO标签列表，与文本字符一一对应
    """
    # 初始化全'O'标签
    labels = ['O'] * len(text)

    for entity in entities:
        start_pos = entity['start_pos']
        end_pos = entity['end_pos']
        label_type = normalize_entity_type(entity['label_type'])

        # 边界检查
        if start_pos < 0 or end_pos > len(text) or start_pos >= end_pos:
            continue

        # 跳过嵌套实体（overlap=1表示嵌套）
        if entity.get('overlap', 0) == 1:
            continue

        # 设置B-标签
        labels[start_pos] = f'B-{label_type}'

        # 设置I-标签
        for i in range(start_pos + 1, end_pos):
            labels[i] = f'I-{label_type}'

    return labels


# =============================================================================
# 滑动窗口处理
# =============================================================================
def create_sliding_window_examples(
    text: str,
    char_labels: List[str],
    tokenizer: BertTokenizerFast,
    max_length: int = 512,
    stride: int = 128
) -> List[Dict]:
    """
    创建滑动窗口训练样本

    简化策略：实体只在其start_pos所在的窗口中完整标记（B-XXX + I-XXX），
    其他窗口中该实体位置统一标记为'O'，避免训练时标签不一致

    Args:
        text: 原始病历文本
        char_labels: 字符级BIO标签
        tokenizer: BERT分词器
        max_length: 最大token长度
        stride: 滑动窗口步长

    Returns:
        样本列表，每项包含input_ids, attention_mask, labels, offset_mapping等
    """
    # 编码文本
    encoding = tokenizer(
        text,
        add_special_tokens=False,  # 手动添加CLS/SEP
        return_offsets_mapping=True,
        return_attention_mask=False
    )

    # 处理空文本或超短文本
    if len(text) == 0:
        return []

    # 短文本：直接返回单个样本
    if len(text) <= max_length - 2:  # -2 for CLS and SEP
        return [_create_single_example(text, char_labels, tokenizer, (0, len(text)))]

    # 长文本：滑动窗口
    examples = []
    text_len = len(text)

    for start_idx in range(0, text_len, stride):
        end_idx = min(start_idx + max_length - 2, text_len)  # -2 for CLS and SEP

        # 确保窗口不会太小
        if end_idx - start_idx < 50:
            continue

        example = _create_single_example(text, char_labels, tokenizer, (start_idx, end_idx))

        if example:
            examples.append(example)

        # 如果到达文本末尾，停止
        if end_idx >= text_len:
            break

    return examples


def _create_single_example(
    text: str,
    char_labels: List[str],
    tokenizer: BertTokenizerFast,
    window_range: Tuple[int, int]
) -> Optional[Dict]:
    """
    创建单个训练样本

    Args:
        text: 原始文本
        char_labels: 字符级BIO标签
        tokenizer: 分词器
        window_range: (start_idx, end_idx) 字符级窗口范围

    Returns:
        样本字典，包含input_ids, labels等
    """
    start_idx, end_idx = window_range
    window_text = text[start_idx:end_idx]

    # 编码窗口文本
    encoding = tokenizer(
        window_text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors=None
    )

    # 获取offset_mapping（相对于窗口文本的偏移）
    offset_mapping = encoding['offset_mapping']

    # 计算字符到token的对齐
    # 对于每个字符位置，找到对应的token索引
    char_to_token_indices = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:  # CLS, SEP, PAD
            char_to_token_indices.append(-1)
        else:
            # 这个token覆盖的字符范围是[start, end)
            char_to_token_indices.append((start, end))

    # 构建token级标签
    # 首先确定每个字符对应的原始标签（考虑窗口偏移）
    token_labels = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            # CLS, SEP, PAD → 设为-100
            token_labels.append(-100)
            continue

        # 找到原始文本中对应位置
        orig_char_start = start_idx + start
        orig_char_end = start_idx + end

        # 边界检查
        if orig_char_start >= len(char_labels):
            token_labels.append(-100)
            continue

        # 获取字符级标签
        char_label = char_labels[orig_char_start]

        # 特殊处理：检查实体边界
        # 如果实体的start_pos在窗口内，则标记B-XXX/I-XXX
        # 否则标记为'O'
        label_to_assign = _get_label_for_position(
            char_label, orig_char_start, start_idx, char_labels
        )

        token_labels.append(label_to_assign)

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding.get('token_type_ids', [0] * len(encoding['input_ids'])),
        'labels': token_labels,
        'offset_mapping': offset_mapping,
        'text': window_text,
        'original_text': text,
        'window_start': start_idx,
    }


def _get_label_for_position(
    current_char_label: str,
    char_pos: int,
    window_start: int,
    char_labels: List[str]
) -> int:
    """
    根据字符位置确定对应的标签

    核心逻辑：只有当实体的start_pos落在当前窗口内时，才完整标记该实体
    否则，即使字符在实体内，也标记为'O'
    """
    # 如果当前是'O'，直接返回'O'
    if current_char_label == 'O':
        return get_label_id('O')

    # 如果是B-标签，检查start_pos是否在窗口内
    if current_char_label.startswith('B-'):
        # B标签本身就是实体起点；当前字符来自该窗口，直接保留B标签
        return get_label_id(current_char_label)

    # 如果是I-标签，检查对应的B-标签位置
    if current_char_label.startswith('I-'):
        entity_type = current_char_label[2:]

        # 向前找到B-标签的位置
        b_pos = char_pos
        while b_pos > 0 and char_labels[b_pos] != f'B-{entity_type}':
            b_pos -= 1

        # 如果B-标签在窗口内，则当前I-标签有效
        if b_pos >= window_start and char_labels[b_pos] == f'B-{entity_type}':
            return get_label_id(current_char_label)
        else:
            return get_label_id('O')

    return get_label_id('O')


# =============================================================================
# 标签对齐（WordPiece）
# =============================================================================
def align_labels_with_subwords(
    char_labels: List[str],
    tokenizer: BertTokenizerFast,
    word_ids: List[int]
) -> List[int]:
    """
    将字符级BIO标签对齐到子词级

    Args:
        char_labels: 字符级标签列表
        tokenizer: 分词器
        word_ids: token对应的word_ids

    Returns:
        token级标签列表（-100表示忽略）
    """
    token_labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # 特殊token（CLS, SEP, PAD）→ 忽略
            token_labels.append(-100)
        elif word_id != prev_word_id:
            # 单词的第一个子词
            if word_id < len(char_labels):
                token_labels.append(get_label_id(char_labels[word_id]))
            else:
                token_labels.append(-100)
        else:
            # 单词的后续子词 → 忽略
            token_labels.append(-100)

        prev_word_id = word_id

    return token_labels


# =============================================================================
# 数据集划分
# =============================================================================
def split_train_val(
    dataset: Dataset,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    划分训练集和验证集

    Args:
        dataset: 完整数据集
        val_split: 验证集比例
        seed: 随机种子

    Returns:
        (训练集, 验证集)
    """
    return dataset.train_test_split(
        test_size=val_split,
        shuffle=True,
        seed=seed
    )


# =============================================================================
# 主预处理流程
# =============================================================================
def preprocess_dataset(config: PreprocessConfig) -> DatasetDict:
    """
    完整预处理流程

    Args:
        config: 预处理配置

    Returns:
        HuggingFace DatasetDict（包含train/validation/test）
    """
    set_seed(config.seed)

    # 加载分词器
    print(f"Loading tokenizer from {config.model_path}")
    tokenizer = BertTokenizerFast.from_pretrained(str(config.model_path))

    # 加载数据
    print("Loading training data...")
    train_raw = load_training_data(config.train_file)

    print("Loading test data...")
    test_raw = load_test_data(config.test_file)

    # 处理训练数据
    print("Processing training data with sliding window...")
    train_examples = []
    for item in train_raw:
        text = item['originalText']
        entities = item['entities']

        # 转换为BIO标签
        char_labels = text_to_bio_labels(text, entities)

        # 滑动窗口
        examples = create_sliding_window_examples(
            text, char_labels, tokenizer,
            max_length=config.max_length,
            stride=config.stride
        )

        train_examples.extend(examples)

    print(f"Created {len(train_examples)} training examples")

    # 处理测试数据
    print("Processing test data with sliding window...")
    test_examples = []
    for item in test_raw:
        text = item['originalText']
        entities = item['entities']

        char_labels = text_to_bio_labels(text, entities)

        examples = create_sliding_window_examples(
            text, char_labels, tokenizer,
            max_length=config.max_length,
            stride=config.stride
        )

        test_examples.extend(examples)

    print(f"Created {len(test_examples)} test examples")

    # 创建Dataset
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    # 划分训练/验证集
    print(f"Splitting train/validation (val_ratio={config.train_val_split})...")
    split_ds = split_train_val(
        train_dataset,
        val_split=config.train_val_split,
        seed=config.seed
    )

    # 构建DatasetDict
    dataset_dict = DatasetDict({
        'train': split_ds['train'],
        'validation': split_ds['test'],
        'test': test_dataset
    })

    # 保存处理后的数据
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed data to {config.output_dir}")
    dataset_dict.save_to_disk(str(config.output_dir))

    # 保存标签映射
    label_mapping = {
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
    }
    save_json(label_mapping, config.output_dir / 'label_mapping.json')

    # 保存配置
    config_dict = {
        'max_length': config.max_length,
        'stride': config.stride,
        'train_samples': len(train_examples),
        'val_samples': len(split_ds['test']),
        'test_samples': len(test_examples),
    }
    save_json(config_dict, config.output_dir / 'preprocess_config.json')

    print("\n" + "="*60)
    print("Preprocessing Summary:")
    print(f"  Train samples: {len(split_ds['train'])}")
    print(f"  Validation samples: {len(split_ds['test'])}")
    print(f"  Test samples: {len(test_examples)}")
    print(f"  Output directory: {config.output_dir}")
    print("="*60)

    return dataset_dict


# =============================================================================
# 入口
# =============================================================================
def main():
    """主入口"""
    config = PreprocessConfig()
    preprocess_dataset(config)


if __name__ == '__main__':
    main()
