# -*- coding: utf-8 -*-
"""
NER模型训练脚本
CCKS 2019 Medical NER Training Pipeline

使用HuggingFace Transformers Trainer API进行训练
支持早停、TensorBoard日志、类别权重
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import load_from_disk, DatasetDict
from transformers import (
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, precision_score, recall_score

# 本地模块
from utils import (
    set_seed, load_json, PROJECT_ROOT, MODEL_ROOT,
    PROCESSED_DATA_DIR, MODEL_OUTPUT_DIR, ID_TO_LABEL,
    compute_class_weights, set_seed
)
from model import create_ner_model, BERTForMedicalNER


# =============================================================================
# 评估指标计算
# =============================================================================
def compute_metrics_seqeval(p):
    """
    计算序列标注评估指标（实体级）

    使用seqeval计算token级的P/R/F1，然后汇总为实体级指标

    Args:
        p: Trainer预测结果，包含predictions和label_ids

    Returns:
        指标字典
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 转换为标签序列
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_tag_seq = []
        pred_tag_seq = []

        for pred, label in zip(pred_seq, label_seq):
            # 忽略-100（特殊token）
            if label != -100:
                true_tag_seq.append(ID_TO_LABEL.get(label, 'O'))
                pred_tag_seq.append(ID_TO_LABEL.get(pred, 'O'))

        if true_tag_seq:  # 避免空序列
            true_labels.append(true_tag_seq)
            pred_labels.append(pred_tag_seq)

    # 计算实体级指标
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # 计算各类别指标
    entity_types = ['DISEASE', 'TESTIMAGE', 'TESTLAB', 'OPERATION', 'DRUG', 'ANATOMY']
    type_metrics = {}

    for etype in entity_types:
        # 提取该类型的实体
        true_filtered = []
        pred_filtered = []

        for true_seq, pred_seq in zip(true_labels, pred_labels):
            true_filtered_seq = []
            pred_filtered_seq = []

            for t, p in zip(true_seq, pred_seq):
                # 只保留该类型的标签
                if etype in t:
                    true_filtered_seq.append(t)
                else:
                    true_filtered_seq.append('O')

                if etype in p:
                    pred_filtered_seq.append(p)
                else:
                    pred_filtered_seq.append('O')

            true_filtered.append(true_filtered_seq)
            pred_filtered.append(pred_filtered_seq)

        try:
            type_p = precision_score(true_filtered, pred_filtered)
            type_r = recall_score(true_filtered, pred_filtered)
            type_f1 = f1_score(true_filtered, pred_filtered)
        except:
            type_p, type_r, type_f1 = 0.0, 0.0, 0.0

        type_metrics[f'{etype}_precision'] = type_p
        type_metrics[f'{etype}_recall'] = type_r
        type_metrics[f'{etype}_f1'] = type_f1

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **type_metrics
    }


class MedicalNERTrainer:
    """
    医疗NER训练器

    封装模型、数据、训练配置
    """

    def __init__(
        self,
        model_path: Path = MODEL_ROOT,
        data_dir: Path = PROCESSED_DATA_DIR,
        output_dir: Path = MODEL_OUTPUT_DIR,
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir

        # 自动检测CUDA
        self.device = self._detect_device()
        print(f"Using device: {self.device}")

        # 加载数据和分词器
        self._load_data()

        # 创建模型
        self._create_model()

    def _detect_device(self) -> str:
        """检测可用设备"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def _load_data(self):
        """加载数据集和分词器"""
        # 加载分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(str(self.model_path))
        print(f"Tokenizer loaded from {self.model_path}")

        # 加载处理后的数据集
        self.dataset = load_from_disk(str(self.data_dir))
        print(f"Dataset loaded from {self.data_dir}")
        print(f"  Train: {len(self.dataset['train'])}")
        print(f"  Validation: {len(self.dataset['validation'])}")
        print(f"  Test: {len(self.dataset['test'])}")

        # 加载标签映射
        label_mapping = load_json(self.data_dir / 'label_mapping.json')
        self.num_labels = len(label_mapping['label_to_id'])
        print(f"Number of labels: {self.num_labels}")

    def _create_model(self):
        """创建模型"""
        # 计算类别权重（可选）
        class_weights = None
        try:
            class_weights = compute_class_weights(
                self.dataset['train'],
                label_column='labels'
            )
            print(f"Computed class weights: {class_weights}")
        except Exception as e:
            print(f"Warning: Could not compute class weights: {e}")

        # 创建模型
        self.model = create_ner_model(
            model_path=self.model_path,
            num_labels=self.num_labels,
            class_weights=class_weights,
        )

        self.model.to(self.device)
        print(f"Model created and moved to {self.device}")

    def train(
        self,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 3e-5,
        num_train_epochs: int = 15,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 2,
        early_stopping_patience: int = 3,
        fp16: bool = True,
        logging_steps: int = 50,
        save_steps: int = 100,
    ):
        """
        训练模型

        Args:
            per_device_train_batch_size: 训练批大小
            per_device_eval_batch_size: 评估批大小
            learning_rate: 学习率
            num_train_epochs: 训练轮数
            weight_decay: 权重衰减
            warmup_ratio: 预热比例
            gradient_accumulation_steps: 梯度累积步数
            early_stopping_patience: 早停耐心值
            fp16: 是否使用混合精度
            logging_steps: 日志输出步数
            save_steps: 模型保存步数
        """
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 计算总步数
        total_steps = (len(self.dataset['train']) //
                      per_device_train_batch_size *
                      num_train_epochs) // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16 and self.device == 'cuda',  # 仅CUDA支持fp16
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=logging_steps,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            report_to=['tensorboard'],
            save_total_limit=3,
            seed=42,
        )

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_seqeval,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        # 开始训练
        print("\n" + "="*60)
        print("Starting training...")
        print(f"  Total epochs: {num_train_epochs}")
        print(f"  Batch size: {per_device_train_batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print("="*60 + "\n")

        trainer.train()

        # 保存最佳模型
        best_model_dir = self.output_dir / 'best_model'
        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))
        print(f"\nBest model saved to {best_model_dir}")

        # 最终评估
        print("\n" + "="*60)
        print("Final evaluation on test set...")
        print("="*60)
        eval_result = trainer.evaluate(self.dataset['test'])
        print(eval_result)

        return trainer


# =============================================================================
# 主入口
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train Medical NER Model')

    # 路径配置
    parser.add_argument('--model_path', type=str, default=str(MODEL_ROOT),
                       help='Pretrained model path')
    parser.add_argument('--data_dir', type=str, default=str(PROCESSED_DATA_DIR),
                       help='Processed data directory')
    parser.add_argument('--output_dir', type=str, default=str(MODEL_OUTPUT_DIR),
                       help='Output directory')

    # 训练超参数
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--num_train_epochs', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--no_fp16', action='store_true', help='Disable fp16')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建训练器
    trainer = MedicalNERTrainer(
        model_path=Path(args.model_path),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )

    # 开始训练
    trainer.train(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        fp16=not args.no_fp16,
    )

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
