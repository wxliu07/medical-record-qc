# -*- coding: utf-8 -*-
"""
端到端推理脚本
CCKS 2019 Medical NER Training Pipeline

整合NER模型 + 规则引擎，实现：
- 单文本推理：end2end_predict(text)
- 批量推理：batch_predict(texts)
- 结果保存：JSON/CSV格式
- Gradio Web Demo（可选）
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from transformers import BertTokenizerFast
from safetensors.torch import load_file as load_safetensors

# 本地模块
from utils import (
    set_seed, PROJECT_ROOT, MODEL_OUTPUT_DIR, RESULTS_DIR, MODEL_ROOT,
    ID_TO_LABEL, bio_to_entities
)
from model import create_ner_model
from rule_engine import extract_relations, MedicalRelationExtractor


# =============================================================================
# 推理器类
# =============================================================================
class MedicalNERREInference:
    """
    医疗NER+RE端到端推理器

    整合NER模型和规则引擎，支持单条和批量推理
    """

    def __init__(
        self,
        model_path: Path = MODEL_OUTPUT_DIR / 'best_model',
        device: Optional[str] = None,
    ):
        """
        初始化推理器

        Args:
            model_path: 训练好的模型路径
            device: 推理设备（cuda/cpu/mps），None则自动检测
        """
        self.model_path = model_path

        # 自动检测设备
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # 加载模型和分词器
        self._load_model()

        # 初始化规则引擎
        self._init_rule_engine()

    def _load_model(self):
        """加载NER模型和分词器"""
        # 加载分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(str(self.model_path))
        print(f"Tokenizer loaded from {self.model_path}")

        # 获取标签数量
        num_labels = len(ID_TO_LABEL)

        # 处理best_model目录缺少config.json的情况：回退到基础BERT配置
        backbone_path = self.model_path if (self.model_path / 'config.json').exists() else MODEL_ROOT

        # 创建模型
        self.model = create_ner_model(
            model_path=backbone_path,
            num_labels=num_labels,
        )

        # 加载权重（优先safetensors，回退pytorch_model.bin）
        safetensors_file = self.model_path / 'model.safetensors'
        pytorch_file = self.model_path / 'pytorch_model.bin'

        if safetensors_file.exists():
            model_state = load_safetensors(str(safetensors_file))
            self.model.load_state_dict(model_state, strict=False)
        elif pytorch_file.exists():
            model_state = torch.load(pytorch_file, map_location=self.device)
            self.model.load_state_dict(model_state, strict=False)
        else:
            print(
                f"Warning: no model weights found under {self.model_path}; "
                "inference will run with randomly initialized head"
            )

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {self.model_path}")

    def _init_rule_engine(self):
        """初始化规则引擎"""
        self.rule_engine = MedicalRelationExtractor()
        print("Rule engine initialized")

    def predict_ner(self, text: str) -> List[Dict]:
        """
        NER预测

        Args:
            text: 原始病历文本

        Returns:
            实体列表，每项包含 text, start_pos, end_pos, type, confidence
        """
        if not text or len(text.strip()) == 0:
            return []

        # 文本过长处理
        if len(text) > 5000:
            print(f"Warning: Text too long ({len(text)} chars), truncating to 5000 chars")
            text = text[:5000]

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

        # 推理
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

        # 转换为Python列表
        pred_labels_list = [ID_TO_LABEL[l.item()] for l in pred_labels]
        confidences_list = [c.item() for c in confidences]
        offset_mapping = encoding['offset_mapping'][0].tolist()

        # 构建字符级标签和置信度
        char_labels = ['O'] * len(text)
        char_confs = [0.0] * len(text)

        for i, (label, conf, offset) in enumerate(zip(pred_labels_list, confidences_list, offset_mapping)):
            if offset[0] == 0 and offset[1] == 0:  # 特殊token
                continue

            start, end = offset
            if start < len(text) and end <= len(text):
                if label != 'O':
                    char_labels[start] = label
                    char_confs[start] = conf
                    for j in range(start + 1, min(end, len(text))):
                        char_labels[j] = label.replace('B-', 'I-')
                        char_confs[j] = conf

        # BIO转实体
        entities = bio_to_entities(text, char_labels, char_confs)

        return entities

    def predict_re(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        关系抽取预测

        Args:
            text: 原始病历文本
            entities: NER实体列表

        Returns:
            关系三元组列表
        """
        if not entities:
            return []

        relations = extract_relations(text, entities)
        return relations

    def end2end_predict(self, text: str) -> Dict:
        """
        端到端预测：NER + RE

        Args:
            text: 原始病历文本

        Returns:
            {
                'original_text': str,   # 原始文本
                'entities': [           # 实体列表
                    {'text': str, 'start_pos': int, 'end_pos': int,
                     'type': str, 'confidence': float}
                ],
                'relations': [          # 关系列表
                    {'head_text': str, 'head_type': str,
                     'tail_text': str, 'tail_type': str,
                     'relation': str, 'confidence': float, 'match_rule': str}
                ]
            }
        """
        if not text or len(text.strip()) == 0:
            return {
                'original_text': text,
                'entities': [],
                'relations': []
            }

        # NER
        entities = self.predict_ner(text)

        # RE
        relations = self.predict_re(text, entities)

        return {
            'original_text': text,
            'entities': entities,
            'relations': relations
        }

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        批量预测

        Args:
            texts: 病历文本列表

        Returns:
            结果列表
        """
        results = []
        for i, text in enumerate(texts):
            result = self.end2end_predict(text)
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"Predicted {i + 1}/{len(texts)} examples")

        return results


# =============================================================================
# 结果保存
# =============================================================================
def save_results_json(results: List[Dict], output_path: Path) -> None:
    """保存结果为JSON格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")


def save_results_csv(results: List[Dict], output_path: Path) -> None:
    """保存结果为CSV格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        # CSV头
        fieldnames = [
            'original_text', 'entity_text', 'entity_type',
            'entity_start', 'entity_end', 'entity_confidence',
            'head_text', 'head_type', 'tail_text', 'tail_type',
            'relation', 'confidence', 'match_rule'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 写入数据
        for result in results:
            text = result['original_text']
            entities = result.get('entities', [])
            relations = result.get('relations', [])

            if not relations:
                # 无关系，只写实体
                for e in entities:
                    writer.writerow({
                        'original_text': text[:200] + '...' if len(text) > 200 else text,
                        'entity_text': e['text'],
                        'entity_type': e['type'],
                        'entity_start': e['start_pos'],
                        'entity_end': e['end_pos'],
                        'entity_confidence': e.get('confidence', ''),
                        'head_text': '',
                        'head_type': '',
                        'tail_text': '',
                        'tail_type': '',
                        'relation': '',
                        'confidence': '',
                        'match_rule': ''
                    })
            else:
                # 有关系
                for r in relations:
                    writer.writerow({
                        'original_text': text[:200] + '...' if len(text) > 200 else text,
                        'entity_text': '',
                        'entity_type': '',
                        'entity_start': '',
                        'entity_end': '',
                        'entity_confidence': '',
                        'head_text': r['head_text'],
                        'head_type': r['head_type'],
                        'tail_text': r['tail_text'],
                        'tail_type': r['tail_type'],
                        'relation': r['relation'],
                        'confidence': r.get('confidence', ''),
                        'match_rule': r.get('match_rule', '')
                    })

    print(f"Results saved to {output_path}")


# =============================================================================
# 示例推理
# =============================================================================
def run_example(inference: MedicalNERREInference):
    """运行示例推理"""
    # 示例病历
    example_texts = [
        "患者3月前因直肠癌于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗。",
        "胃癌患者，行胃癌根治术，术后病理示胃底腺癌，CT示肝部有阴影。",
        "患者因肺癌入院，行胸腔镜手术，术后给予紫杉醇化疗。",
    ]

    print("\n" + "="*60)
    print("Running Example Inference")
    print("="*60)

    for i, text in enumerate(example_texts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Input: {text[:100]}...")

        result = inference.end2end_predict(text)

        print(f"\nEntities ({len(result['entities'])}):")
        for e in result['entities']:
            print(f"  - {e['text']} [{e['type']}] pos={e['start_pos']}-{e['end_pos']} conf={e.get('confidence', 0):.2f}")

        print(f"\nRelations ({len(result['relations'])}):")
        for r in result['relations']:
            print(f"  - {r['head_text']}({r['head_type']}) --[{r['relation']}]--> "
                  f"{r['tail_text']}({r['tail_type']})")

        print()

    print("="*60)


# =============================================================================
# Gradio Web Demo（可选）
# =============================================================================
def create_gradio_demo(inference: MedicalNERREInference):
    """创建Gradio Web Demo"""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return None

    def predict(text: str) -> str:
        if not text.strip():
            return "请输入病历文本"

        result = inference.end2end_predict(text)

        # 格式化输出
        lines = []
        lines.append("="*50)
        lines.append("【实体列表】")
        for e in result['entities']:
            lines.append(f"  {e['text']} [{e['type']}] (位置:{e['start_pos']}-{e['end_pos']}, 置信度:{e.get('confidence', 0):.2f})")

        lines.append("\n【关系三元组】")
        if result['relations']:
            for r in result['relations']:
                lines.append(f"  {r['head_text']}({r['head_type']}) --[{r['relation']}]--> {r['tail_text']}({r['tail_type']}) "
                           f"[置信度:{r.get('confidence', 0):.2f}, 规则:{r.get('match_rule', '')}]")
        else:
            lines.append("  未检测到关系")

        lines.append("="*50)

        return "\n".join(lines)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="病历文本", placeholder="请输入病历文本..."),
        outputs=gr.Textbox(label="识别结果", interactive=False),
        title="医疗实体识别与关系抽取",
        description="输入中文病历文本，自动识别医疗实体和关系三元组",
        examples=[
            ["患者因胃癌于全麻上行胃癌根治术，术后给予奥沙利铂化疗。"],
            ["CT示：肝部有阴影，疑似肝癌。"],
        ]
    )

    return demo


# =============================================================================
# 主入口
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Medical NER+RE Inference')

    parser.add_argument('--model_path', type=str,
                       default=str(MODEL_OUTPUT_DIR / 'best_model'),
                       help='Path to trained model')
    parser.add_argument('--text', type=str, default=None,
                       help='Input text for prediction')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file with texts (JSON array)')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Output JSON file')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Output CSV file')
    parser.add_argument('--example', action='store_true',
                       help='Run example inference')
    parser.add_argument('--gradio', action='store_true',
                       help='Launch Gradio demo')
    parser.add_argument('--port', type=int, default=7860,
                       help='Gradio port')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu/mps)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 创建推理器
    inference = MedicalNERREInference(
        model_path=Path(args.model_path),
        device=args.device
    )

    # 示例推理
    if args.example:
        run_example(inference)
        return

    # Gradio Demo
    if args.gradio:
        demo = create_gradio_demo(inference)
        if demo:
            print(f"\nStarting Gradio demo on port {args.port}...")
            demo.launch(server_port=args.port)
        return

    # 单条文本推理
    if args.text:
        result = inference.end2end_predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 批量推理
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)

        results = inference.batch_predict(texts)

        if args.output_json:
            save_results_json(results, Path(args.output_json))

        if args.output_csv:
            save_results_csv(results, Path(args.output_csv))

        print(f"\nProcessed {len(results)} examples")
        return

    # 无参数：运行示例
    print("No input provided. Running example inference...")
    run_example(inference)


if __name__ == '__main__':
    main()
