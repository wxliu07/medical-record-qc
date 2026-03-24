# -*- coding: utf-8 -*-
"""
NER模型定义
CCKS 2019 Medical NER Training Pipeline

基于预训练中文BERT的token分类模型
支持加权CrossEntropyLoss处理类别不平衡
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from pathlib import Path
from typing import Optional, List

from utils import (
    LABEL_TO_ID, ID_TO_LABEL, MODEL_ROOT, LABEL_TO_ID
)


# =============================================================================
# 模型定义
# =============================================================================
class BERTForMedicalNER(nn.Module):
    """
    基于中文BERT的医疗实体识别模型

    结构：BERT编码器 + 线性分类头
    损失函数：加权CrossEntropyLoss（可选）
    """

    def __init__(
        self,
        model_path: str | Path,
        num_labels: int = 13,  # 12个实体标签 + 1个'O'标签
        label_to_id: dict = None,
        id_to_label: dict = None,
        class_weights: Optional[List[float]] = None,
        dropout_rate: float = 0.1,
    ):
        """
        初始化BERT NER模型

        Args:
            model_path: 预训练模型路径
            num_labels: 标签数量
            label_to_id: 标签到ID的映射
            id_to_label: ID到标签的映射
            class_weights: 类别权重，用于加权损失
            dropout_rate: Dropout比例
        """
        super().__init__()

        self.model_path = Path(model_path)
        self.num_labels = num_labels
        self.class_weights = class_weights

        # 保存标签映射
        self.label_to_id = label_to_id or LABEL_TO_ID
        self.id_to_label = id_to_label or ID_TO_LABEL

        # 加载预训练BERT
        self.bert = BertModel.from_pretrained(str(self.model_path))

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 分类头
        hidden_size = self.bert.config.hidden_size  # 768 for BERT-base
        self.classifier = nn.Linear(hidden_size, num_labels)

        # 损失函数
        if class_weights is not None:
            # 转换为tensor并移动到正确设备
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32),
                ignore_index=-100  # 忽略特殊token的损失
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化分类器权重"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        """
        前向传播

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            labels: 标签（训练时使用）

        Returns:
            包含loss和logits的字典（训练时）
            或仅logits（推理时）
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # 分类
        logits = self.classifier(sequence_output)

        # 计算损失（仅训练时）
        loss = None
        if labels is not None:
            # Reshape logits和labels
            # logits: (batch_size, seq_len, num_labels)
            # labels: (batch_size, seq_len)
            loss = self.loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        return {
            'loss': loss,
            'logits': logits,
        }

    def predict(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ) -> torch.Tensor:
        """
        推理预测

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs

        Returns:
            预测的标签IDs
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs['logits']

            # 取最大概率的标签
            predictions = torch.argmax(logits, dim=-1)

        return predictions

    def predict_with_confidence(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ) -> tuple:
        """
        推理预测（带置信度）

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs

        Returns:
            (预测标签IDs, 置信度)
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs['logits']

            # Softmax获取概率
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)

        return predictions, confidences

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        **kwargs
    ):
        """
        从预训练路径加载模型

        Args:
            model_path: 模型路径

        Returns:
            加载的模型
        """
        return cls(model_path=model_path, **kwargs)


# =============================================================================
# 模型创建工具函数
# =============================================================================
def create_ner_model(
    model_path: str | Path = MODEL_ROOT,
    num_labels: int = 13,
    class_weights: Optional[List[float]] = None,
    dropout_rate: float = 0.1,
) -> BERTForMedicalNER:
    """
    创建NER模型

    Args:
        model_path: 预训练模型路径
        num_labels: 标签数量
        class_weights: 类别权重
        dropout_rate: Dropout比例

    Returns:
        BERTForMedicalNER模型
    """
    model = BERTForMedicalNER(
        model_path=model_path,
        num_labels=num_labels,
        label_to_id=LABEL_TO_ID,
        id_to_label=ID_TO_LABEL,
        class_weights=class_weights,
        dropout_rate=dropout_rate,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 测试
# =============================================================================
if __name__ == '__main__':
    # 测试模型创建
    print("Testing model creation...")

    model = create_ner_model()
    print(f"Model loaded from {MODEL_ROOT}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    # 测试前向传播
    batch_size = 2
    seq_length = 128

    input_ids = torch.randint(0, 21128, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    labels = torch.randint(0, 13, (batch_size, seq_length))

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    print("Model test passed!")
