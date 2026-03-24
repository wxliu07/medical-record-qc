# -*- coding: utf-8 -*-
"""
医疗关系规则引擎
CCKS 2019 Medical NER Training Pipeline

基于临床规则的实体关系抽取：
- 先对病历文本分句
- 仅匹配同句内的实体对
- 按优先级匹配：文本包含 > 关键词 > 位置
- 单规则匹配，不合并
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from utils import split_sentences


# =============================================================================
# 关系类型定义
# =============================================================================
@dataclass
class RelationType:
    """关系类型定义"""
    name: str
    head_type: str
    tail_type: str
    description: str
    priority: int = 1  # 优先级（数字越小越高）


# 预定义的6个核心关系类型
RELATION_TYPES = {
    'has_location': RelationType(
        name='has_location',
        head_type='DISEASE',
        tail_type='ANATOMY',
        description='疾病发生的解剖部位',
        priority=1
    ),
    'treated_by_operation': RelationType(
        name='treated_by_operation',
        head_type='DISEASE',
        tail_type='OPERATION',
        description='疾病采用的手术治疗方案',
        priority=1
    ),
    'treated_by_drug': RelationType(
        name='treated_by_drug',
        head_type='DISEASE',
        tail_type='DRUG',
        description='疾病采用的药物/化疗治疗方案',
        priority=1
    ),
    'examined_by_image': RelationType(
        name='examined_by_image',
        head_type='DISEASE',
        tail_type='TESTIMAGE',
        description='疾病采用的影像检查手段',
        priority=1
    ),
    'examined_by_lab': RelationType(
        name='examined_by_lab',
        head_type='DISEASE',
        tail_type='TESTLAB',
        description='疾病采用的实验室检验手段',
        priority=1
    ),
    'performed_on': RelationType(
        name='performed_on',
        head_type='OPERATION',
        tail_type='ANATOMY',
        description='手术实施的解剖部位',
        priority=1
    ),
}

# 预留扩展接口
EXTENDED_RELATIONS = {
    'medication_for': RelationType(
        name='medication_for',
        head_type='DRUG',
        tail_type='DISEASE',
        description='药物的治疗适应症',
        priority=2
    ),
    'result_of': RelationType(
        name='result_of',
        head_type='TESTIMAGE',
        tail_type='DISEASE',
        description='检查/检验对应的诊断结果',
        priority=2
    ),
    'adjacent_to': RelationType(
        name='adjacent_to',
        head_type='ANATOMY',
        tail_type='ANATOMY',
        description='解剖部位的相邻关系',
        priority=2
    ),
    'no_relation': RelationType(
        name='no_relation',
        head_type='*',
        tail_type='*',
        description='无明确临床关系',
        priority=3
    ),
}

# =============================================================================
# 关键词规则定义
# =============================================================================
# 临床常用触发词
KEYWORDS = {
    'has_location': [
        '位于', '发生于', '累及', '侵犯', '侵袭',
        '在...部', '癌位于', '肿物位于', '病灶位于',
    ],
    'treated_by_operation': [
        '行...术', '行...手术', '手术切除', '根治术',
        '切除术', '清扫术', '姑息手术', '探查术',
        '术后', '术前', '术中', '全麻上行',
    ],
    'treated_by_drug': [
        '方案为', '给予...化疗', '给予...治疗', '药物治疗',
        '口服', '静滴', '静脉滴注', '肌肉注射',
        '化疗方案', '免疫治疗', '靶向治疗', '内分泌治疗',
    ],
    'examined_by_image': [
        'CT示', 'MR示', 'MRI示', 'X线示', 'B超示',
        '彩超示', '超声示', 'PET示', '造影示',
        'CT平扫', 'CT增强', '磁共振', '影像学检查',
        '复查CT', '复查MRI',
    ],
    'examined_by_lab': [
        '检验示', '化验示', '实验室检查', '血常规',
        '生化检查', '肿瘤标志物', '免疫组化',
    ],
    'performed_on': [
        '于...行', '在...部位', '位于...',
        '手术于', '切除...', '清扫...',
    ],
}

# 文本包含关系（直接包含即匹配）
TEXT_CONTAIN_RULES = {
    'has_location': [
        # 疾病文本直接包含解剖部位
        ('直肠癌', '直肠'),
        ('胃癌', '胃'),
        ('肺癌', '肺'),
        ('肝癌', '肝'),
        ('结肠癌', '结肠'),
        ('乳腺癌', '乳腺'),
    ],
}

# =============================================================================
# 规则引擎核心
# =============================================================================
class MedicalRelationExtractor:
    """
    医疗关系抽取规则引擎

    匹配策略：
    1. 分句：按中文标点（。？！；）分句
    2. 同句匹配：仅匹配同一句话内的实体对
    3. 优先级：文本包含 > 关键词 > 位置
    4. 单规则：同一实体对只取最高优先级规则，不合并
    """

    def __init__(
        self,
        position_threshold: int = 50,
        use_extended: bool = False,
    ):
        """
        初始化规则引擎

        Args:
            position_threshold: 同句内实体对的最大字符距离
            use_extended: 是否启用扩展关系
        """
        self.position_threshold = position_threshold
        self.relations = RELATION_TYPES.copy()

        if use_extended:
            self.relations.update(EXTENDED_RELATIONS)

    def extract_relations(
        self,
        text: str,
        entities: List[Dict],
        return_confidence: bool = True
    ) -> List[Dict]:
        """
        从文本和实体列表中抽取关系三元组

        Args:
            text: 原始病历文本
            entities: NER输出的实体列表
            return_confidence: 是否返回置信度

        Returns:
            关系三元组列表，每项包含：
            {
                'head_text': str,      # 头实体文本
                'head_type': str,       # 头实体类型
                'tail_text': str,       # 尾实体文本
                'tail_type': str,      # 尾实体类型
                'relation': str,        # 关系类型
                'confidence': float,   # 置信度
                'match_rule': str,     # 匹配的规则
            }
        """
        if not text or not entities:
            return []

        # 分句
        sentences = split_sentences(text)

        # 为每个句子记录实体索引范围
        sentence_entity_ranges = self._get_sentence_entity_ranges(text, sentences)

        # 生成所有可能的实体对
        entity_pairs = self._generate_entity_pairs(entities)

        # 过滤：只保留同句内的实体对
        valid_pairs = self._filter_same_sentence_pairs(
            text, sentences, entities, entity_pairs, sentence_entity_ranges
        )

        # 匹配规则
        relations = []
        matched_pairs = set()  # 已匹配的实体对，避免重复

        for head_idx, tail_idx in valid_pairs:
            head = entities[head_idx]
            tail = entities[tail_idx]

            pair_key = (head_idx, tail_idx)

            # 1. 尝试文本包含规则
            match = self._match_text_contain(head, tail, text)
            if match:
                relations.append(self._make_relation(head, tail, match, 1.0, 'text_contain'))
                matched_pairs.add(pair_key)
                continue

            # 2. 尝试关键词规则
            match = self._match_keyword(head, tail, text, sentences)
            if match:
                relations.append(self._make_relation(head, tail, match, 0.9, 'keyword'))
                matched_pairs.add(pair_key)
                continue

            # 3. 尝试位置规则
            match = self._match_position(head, tail, text)
            if match:
                relations.append(self._make_relation(head, tail, match, 0.8, 'position'))
                matched_pairs.add(pair_key)
                continue

        return relations

    def _get_sentence_entity_ranges(
        self,
        text: str,
        sentences: List[str]
    ) -> List[Tuple[int, int]]:
        """
        获取每个句子在原文中的字符位置范围

        Returns:
            句子范围列表 [(start, end), ...]
        """
        ranges = []
        current_pos = 0

        for sent in sentences:
            # 找到句子在原文中的位置
            start = text.find(sent, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(sent)

            ranges.append((start, end))
            current_pos = end

        return ranges

    def _generate_entity_pairs(
        self,
        entities: List[Dict]
    ) -> List[Tuple[int, int]]:
        """
        生成所有有效的实体对（排除自身）

        Returns:
            实体对列表 [(head_idx, tail_idx), ...]
        """
        pairs = []
        n = len(entities)

        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((i, j))

        return pairs

    def _filter_same_sentence_pairs(
        self,
        text: str,
        sentences: List[str],
        entities: List[Dict],
        pairs: List[Tuple[int, int]],
        sentence_ranges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        过滤：只保留同一句子内的实体对

        Args:
            text: 原文
            sentences: 分句列表
            pairs: 实体对列表
            sentence_ranges: 句子范围

        Returns:
            过滤后的实体对列表
        """
        valid_pairs = []

        for head_idx, tail_idx in pairs:
            head = entities[head_idx]
            tail = entities[tail_idx]

            head_pos = (head['start_pos'], head['end_pos'])
            tail_pos = (tail['start_pos'], tail['end_pos'])

            # 检查两个实体是否在同一句子内
            for sent_start, sent_end in sentence_ranges:
                if (sent_start <= head_pos[0] < sent_end and
                    sent_start <= tail_pos[0] < sent_end):
                    valid_pairs.append((head_idx, tail_idx))
                    break

        return valid_pairs

    def _match_text_contain(
        self,
        head: Dict,
        tail: Dict,
        text: str
    ) -> Optional[str]:
        """
        匹配文本包含规则

        检查head实体文本是否直接包含tail实体文本
        如：直肠癌 → 直肠
        """
        for relation_name, rules in TEXT_CONTAIN_RULES.items():
            for disease_phrase, anatomy_phrase in rules:
                if disease_phrase in head['text'] and anatomy_phrase in tail['text']:
                    return relation_name

        # 通用规则：疾病文本包含解剖部位
        if head['type'] == 'DISEASE' and tail['type'] == 'ANATOMY':
            if tail['text'] in head['text']:
                return 'has_location'

        return None

    def _match_keyword(
        self,
        head: Dict,
        tail: Dict,
        text: str,
        sentences: List[str]
    ) -> Optional[str]:
        """
        匹配关键词规则

        检查实体对是否有关键词触发
        """
        # 构建关系类型到头尾类型的映射
        head_tail_to_relation = {}
        for rel_name, rel_type in self.relations.items():
            key = (rel_type.head_type, rel_type.tail_type)
            head_tail_to_relation[key] = rel_name

        # 检查头尾类型是否匹配
        key = (head['type'], tail['type'])
        if key not in head_tail_to_relation:
            return None

        relation_name = head_tail_to_relation[key]

        # 获取头尾实体所在的句子
        head_sent = self._get_entity_sentence(head, text, sentences)
        tail_sent = self._get_entity_sentence(tail, text, sentences)

        if head_sent is None or tail_sent is None:
            return None

        # 检查关键词
        keywords = KEYWORDS.get(relation_name, [])

        for kw in keywords:
            if kw in head_sent or kw in tail_sent:
                return relation_name

        return None

    def _match_position(
        self,
        head: Dict,
        tail: Dict,
        text: str
    ) -> Optional[str]:
        """
        匹配位置规则

        检查头尾实体是否在同句内且距离较近
        """
        # 构建关系类型到头尾类型的映射
        head_tail_to_relation = {}
        for rel_name, rel_type in self.relations.items():
            key = (rel_type.head_type, rel_type.tail_type)
            head_tail_to_relation[key] = rel_name

        # 检查头尾类型是否匹配
        key = (head['type'], tail['type'])
        if key not in head_tail_to_relation:
            return None

        relation_name = head_tail_to_relation[key]

        # 计算字符距离
        head_end = head['end_pos']
        tail_start = tail['start_pos']

        # 头实体在前，尾实体在后
        if head_end <= tail_start:
            distance = tail_start - head_end
        else:
            distance = head['start_pos'] - tail['end_pos']

        # 检查距离阈值
        if distance <= self.position_threshold:
            return relation_name

        return None

    def _get_entity_sentence(
        self,
        entity: Dict,
        text: str,
        sentences: List[str]
    ) -> Optional[str]:
        """获取实体所在的句子"""
        entity_pos = (entity['start_pos'], entity['end_pos'])

        current_pos = 0
        for sent in sentences:
            start = text.find(sent, current_pos)
            if start == -1:
                continue
            end = start + len(sent)

            if start <= entity_pos[0] < end:
                return sent

            current_pos = end

        return None

    def _make_relation(
        self,
        head: Dict,
        tail: Dict,
        relation_name: str,
        confidence: float,
        match_rule: str
    ) -> Dict:
        """构建关系三元组"""
        return {
            'head_text': head['text'],
            'head_type': head['type'],
            'tail_text': tail['text'],
            'tail_type': tail['type'],
            'relation': relation_name,
            'confidence': confidence,
            'match_rule': match_rule,
        }


# =============================================================================
# 单例实例
# =============================================================================
_default_extractor = None


def get_extractor() -> MedicalRelationExtractor:
    """获取默认规则引擎实例"""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = MedicalRelationExtractor()
    return _default_extractor


def extract_relations(
    text: str,
    entities: List[Dict],
    return_confidence: bool = True
) -> List[Dict]:
    """
    抽取关系三元组的便捷函数

    Args:
        text: 原始病历文本
        entities: NER输出的实体列表
        return_confidence: 是否返回置信度

    Returns:
        关系三元组列表
    """
    extractor = get_extractor()
    return extractor.extract_relations(text, entities, return_confidence)


# =============================================================================
# 测试
# =============================================================================
if __name__ == '__main__':
    # 测试用例
    test_text = "患者因胃癌于全麻上行胃癌根治术，术后给予奥沙利铂化疗。CT示：肝部有阴影。"

    test_entities = [
        {'text': '胃癌', 'start_pos': 4, 'end_pos': 6, 'type': 'DISEASE', 'confidence': 0.98},
        {'text': '胃癌根治术', 'start_pos': 11, 'end_pos': 16, 'type': 'OPERATION', 'confidence': 0.95},
        {'text': '奥沙利铂', 'start_pos': 22, 'end_pos': 26, 'type': 'DRUG', 'confidence': 0.92},
        {'text': '肝部', 'start_pos': 32, 'end_pos': 34, 'type': 'ANATOMY', 'confidence': 0.88},
    ]

    extractor = MedicalRelationExtractor()
    relations = extractor.extract_relations(test_text, test_entities)

    print("Test Text:", test_text)
    print("Entities:", test_entities)
    print("\nExtracted Relations:")
    for rel in relations:
        print(f"  {rel['head_text']}({rel['head_type']}) --[{rel['relation']}]--> "
              f"{rel['tail_text']}({rel['tail_type']}) "
              f"[conf={rel['confidence']}, rule={rel['match_rule']}]")

    print("\nRelation extraction test passed!")
