# -*- coding: utf-8 -*-
"""
模型评估脚本
CCKS 2019 Medical NER Training Pipeline

功能：
- 在测试集上评估NER模型
- 计算实体级精确匹配的P/R/F1
- 按实体类别分别评测
- 错误分析，保存错误案例
- 生成评估报告和可视化
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from datasets import load_from_disk
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt
import matplotlib
from safetensors.torch import load_file as load_safetensors

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 本地模块
from utils import (
    set_seed, load_json, PROJECT_ROOT, MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR, RESULTS_DIR, MODEL_ROOT, ID_TO_LABEL, LABEL_TO_ID,
    bio_to_entities, merge_overlapping_entities, load_training_data, load_test_data,
    normalize_entity_type
)
from model import create_ner_model


# =============================================================================
# 评估器类
# =============================================================================
class NEREvaluator:
    """
    NER模型评估器

    支持：
    - 滑动窗口合并预测
    - 实体级精确匹配评测
    - 错误分析
    - 报告生成
    """

    def __init__(
        self,
        model_path: Path = MODEL_OUTPUT_DIR / 'best_model',
        data_dir: Path = PROCESSED_DATA_DIR,
        raw_test_path: Path = None,
        output_dir: Path = RESULTS_DIR,
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.raw_test_path = raw_test_path
        self.output_dir = output_dir

        # 自动检测设备
        self.device = self._detect_device()
        print(f"Using device: {self.device}")

        # 加载模型和分词器
        self._load_model()

        # 加载数据
        self._load_data()

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _detect_device(self) -> str:
        """检测可用设备"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def _load_model(self):
        """加载模型和分词器"""
        # 加载分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(str(self.model_path))
        print(f"Tokenizer loaded from {self.model_path}")

        # 加载标签映射
        label_mapping = load_json(self.data_dir / 'label_mapping.json')
        self.num_labels = len(label_mapping['label_to_id'])

        # 处理best_model目录缺少config.json的情况：回退到基础BERT配置
        backbone_path = self.model_path if (self.model_path / 'config.json').exists() else MODEL_ROOT

        # 创建并加载模型
        self.model = create_ner_model(
            model_path=backbone_path,
            num_labels=self.num_labels,
        )

        # 加载训练好的权重（优先safetensors，回退pytorch_model.bin）
        safetensors_file = self.model_path / 'model.safetensors'
        pytorch_file = self.model_path / 'pytorch_model.bin'

        if safetensors_file.exists():
            model_state = load_safetensors(str(safetensors_file))
        elif pytorch_file.exists():
            model_state = torch.load(pytorch_file, map_location=self.device)
        else:
            raise FileNotFoundError(
                f"No model weights found under {self.model_path} (expected model.safetensors or pytorch_model.bin)"
            )

        self.model.load_state_dict(model_state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {self.model_path}")

    def _load_data(self):
        """加载原始测试数据"""
        # 加载处理后的数据集（用于模型预测）
        self.test_dataset = load_from_disk(str(self.data_dir))['test']
        print(f"Test dataset loaded: {len(self.test_dataset)} examples")

        # 加载原始测试数据（用于评估）
        if self.raw_test_path is None:
            self.raw_test_path = PROJECT_ROOT / 'data' / 'datasets' / 'CCKS2019' / 'subtask1_test_set_with_answer.json'

        self.raw_test_data = load_test_data(self.raw_test_path)
        print(f"Raw test data loaded: {len(self.raw_test_data)} examples")

    def predict_single(self, text: str) -> List[Dict]:
        """
        对单条文本进行预测

        Args:
            text: 原始病历文本

        Returns:
            预测实体列表
        """
        # 编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs['logits']

            # 获取预测标签和置信度
            probs = torch.softmax(logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)[0]
            confidences = torch.max(probs, dim=-1)[0][0]

        # 转换为实体列表
        pred_labels_list = [ID_TO_LABEL[l.item()] for l in pred_labels]
        confidences_list = [c.item() for c in confidences]

        # 提取offset_mapping
        offset_mapping = encoding['offset_mapping'][0].tolist()

        # 过滤特殊token，获取有效token的标签
        valid_labels = []
        valid_confs = []
        valid_offsets = []

        for i, (label, conf, offset) in enumerate(zip(pred_labels_list, confidences_list, offset_mapping)):
            if offset[0] == 0 and offset[1] == 0:  # 特殊token
                continue
            valid_labels.append(label)
            valid_confs.append(conf)
            valid_offsets.append(offset)

        # 构建字符级标签
        char_labels = ['O'] * len(text)
        char_confs = [0.0] * len(text)

        for label, conf, offset in zip(valid_labels, valid_confs, valid_offsets):
            start, end = offset
            if start < len(text) and end <= len(text):
                if label != 'O':
                    char_labels[start] = label
                    char_confs[start] = conf
                    for j in range(start + 1, min(end, len(text))):
                        char_labels[j] = label.replace('B-', 'I-')
                        char_confs[j] = conf

        # 转换为实体
        entities = bio_to_entities(text, char_labels, char_confs)

        return entities

    def predict_raw_data(self) -> Dict[str, List[Dict]]:
        """
        对原始测试数据进行预测

        Returns:
            按originalText索引的预测实体字典
        """
        predictions = {}

        for i, item in enumerate(self.raw_test_data):
            text = item['originalText']
            pred_entities = self.predict_single(text)
            predictions[text] = pred_entities

            if (i + 1) % 50 == 0:
                print(f"Predicted {i + 1}/{len(self.raw_test_data)} examples")

        return predictions

    def evaluate(self) -> Dict:
        """
        执行完整评估

        Returns:
            评估结果字典
        """
        print("\n" + "="*60)
        print("Running evaluation on test set...")
        print("="*60)

        # 预测
        predictions = self.predict_raw_data()

        # 计算指标
        entity_types = ['DISEASE', 'TESTIMAGE', 'TESTLAB', 'OPERATION', 'DRUG', 'ANATOMY']

        # 统计量
        total_tp = defaultdict(int)
        total_fp = defaultdict(int)
        total_fn = defaultdict(int)

        all_gold_entities = []
        all_pred_entities = []

        # 错误案例
        errors = []

        for i, item in enumerate(self.raw_test_data):
            text = item['originalText']
            gold_entities = item['entities']
            pred_entities = predictions.get(text, [])

            # 标准化gold实体
            gold_std = []
            for e in gold_entities:
                gold_std.append({
                    'text': text[e['start_pos']:e['end_pos']],
                    'start_pos': e['start_pos'],
                    'end_pos': e['end_pos'],
                    'type': normalize_entity_type(e['label_type'])
                })

            all_gold_entities.append(gold_std)
            all_pred_entities.append(pred_entities)

            # 评估每个类型
            for etype in entity_types:
                gold_set = set()
                for e in gold_std:
                    if e['type'] == etype:
                        gold_set.add((e['text'], e['start_pos'], e['end_pos']))

                pred_set = set()
                for e in pred_entities:
                    if e['type'] == etype:
                        pred_set.add((e['text'], e['start_pos'], e['end_pos']))

                # 计算TP/FP/FN
                tp = len(gold_set & pred_set)
                fp = len(pred_set - gold_set)
                fn = len(gold_set - pred_set)

                total_tp[etype] += tp
                total_fp[etype] += fp
                total_fn[etype] += fn

                # 记录错误案例
                if fp > 0 or fn > 0:
                    for e in pred_set - gold_set:
                        errors.append({
                            'text': text,
                            'gold_text': None,
                            'gold_type': etype,
                            'gold_pos': None,
                            'pred_text': e[0],
                            'pred_type': etype,
                            'pred_pos': f"{e[1]}-{e[2]}",
                            'error_type': '误检' if e in pred_set and e not in gold_set else '边界错误'
                        })

                if fn > 0:
                    for e in gold_set - pred_set:
                        errors.append({
                            'text': text,
                            'gold_text': e[0],
                            'gold_type': etype,
                            'gold_pos': f"{e[1]}-{e[2]}",
                            'pred_text': None,
                            'pred_type': etype,
                            'pred_pos': None,
                            'error_type': '漏检'
                        })

        # 计算各类别的P/R/F1
        results = {}
        micro_tp = 0
        micro_fp = 0
        micro_fn = 0

        for etype in entity_types:
            tp = total_tp[etype]
            fp = total_fp[etype]
            fn = total_fn[etype]

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[etype] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        # 计算micro和macro
        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        macro_precision = sum(r['precision'] for r in results.values()) / len(results)
        macro_recall = sum(r['recall'] for r in results.values()) / len(results)
        macro_f1 = sum(r['f1'] for r in results.values()) / len(results)

        results['micro'] = {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        }

        results['macro'] = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }

        # 保存错误分析
        self._save_errors(errors)

        # 生成报告
        self._generate_report(results)

        # 绘制F1柱状图
        self._plot_f1_scores(results)

        return results

    def _save_errors(self, errors: List[Dict]):
        """保存错误案例到CSV"""
        error_file = self.output_dir / 'ner_errors.csv'

        with open(error_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                '原文', '金标实体文本', '金标类型', '金标起止位置',
                '预测实体文本', '预测类型', '预测起止位置', '错误类型'
            ])
            writer.writeheader()

            for e in errors:
                writer.writerow({
                    '原文': e['text'][:200] + '...' if len(e['text']) > 200 else e['text'],
                    '金标实体文本': e.get('gold_text', ''),
                    '金标类型': e.get('gold_type', ''),
                    '金标起止位置': e.get('gold_pos', ''),
                    '预测实体文本': e.get('pred_text', ''),
                    '预测类型': e.get('pred_type', ''),
                    '预测起止位置': e.get('pred_pos', ''),
                    '错误类型': e.get('error_type', '')
                })

        print(f"Errors saved to {error_file}")

    def _generate_report(self, results: Dict):
        """生成评估报告"""
        report_file = self.output_dir / 'evaluation_report.txt'

        entity_types = ['DISEASE', 'TESTIMAGE', 'TESTLAB', 'OPERATION', 'DRUG', 'ANATOMY']

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CCKS 2019 Medical NER Evaluation Report\n")
            f.write("="*70 + "\n\n")

            # 整体结果
            f.write("整体结果 (Overall Results):\n")
            f.write("-"*40 + "\n")
            f.write(f"Micro Precision: {results['micro']['precision']:.4f}\n")
            f.write(f"Micro Recall:    {results['micro']['recall']:.4f}\n")
            f.write(f"Micro F1:        {results['micro']['f1']:.4f}\n")
            f.write(f"Macro Precision: {results['macro']['precision']:.4f}\n")
            f.write(f"Macro Recall:    {results['macro']['recall']:.4f}\n")
            f.write(f"Macro F1:        {results['macro']['f1']:.4f}\n\n")

            # 各类别结果
            f.write("分类结果 (Per-Entity Results):\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Entity Type':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}\n")
            f.write("-"*70 + "\n")

            for etype in entity_types:
                r = results[etype]
                f.write(f"{etype:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} "
                       f"{r['tp']:>8} {r['fp']:>8} {r['fn']:>8}\n")

            f.write("-"*70 + "\n")

            # 计算总实体数
            total_tp = sum(results[etype]['tp'] for etype in entity_types)
            total_fp = sum(results[etype]['fp'] for etype in entity_types)
            total_fn = sum(results[etype]['fn'] for etype in entity_types)

            f.write(f"{'Total':<15} {results['micro']['precision']:>10.4f} {results['micro']['recall']:>10.4f} "
                   f"{results['micro']['f1']:>10.4f} {total_tp:>8} {total_fp:>8} {total_fn:>8}\n\n")

            f.write("="*70 + "\n")
            f.write("Evaluation completed.\n")

        print(f"Report saved to {report_file}")

    def _plot_f1_scores(self, results: Dict):
        """绘制F1分数柱状图"""
        entity_types = ['DISEASE', 'TESTIMAGE', 'TESTLAB', 'OPERATION', 'DRUG', 'ANATOMY']

        f1_scores = [results[etype]['f1'] for etype in entity_types]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(entity_types, f1_scores, color=[
            '#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'
        ])

        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=10)

        plt.xlabel('Entity Type', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('CCKS 2019 Medical NER F1 Scores by Entity Type', fontsize=14)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)

        # 保存
        plt_file = self.output_dir / 'ner_f1_score.png'
        plt.savefig(plt_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"F1 plot saved to {plt_file}")


# =============================================================================
# 主入口
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Medical NER Model')

    parser.add_argument('--model_path', type=str,
                       default=str(MODEL_OUTPUT_DIR / 'best_model'),
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str,
                       default=str(PROCESSED_DATA_DIR),
                       help='Processed data directory')
    parser.add_argument('--raw_test_path', type=str, default=None,
                       help='Raw test data file path')
    parser.add_argument('--output_dir', type=str,
                       default=str(RESULTS_DIR),
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建评估器
    evaluator = NEREvaluator(
        model_path=Path(args.model_path),
        data_dir=Path(args.data_dir),
        raw_test_path=Path(args.raw_test_path) if args.raw_test_path else None,
        output_dir=Path(args.output_dir),
    )

    # 执行评估
    results = evaluator.evaluate()

    # 打印摘要
    print("\n" + "="*60)
    print("Evaluation Summary:")
    print(f"  Micro F1: {results['micro']['f1']:.4f}")
    print(f"  Macro F1: {results['macro']['f1']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
