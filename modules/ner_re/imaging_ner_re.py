import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


_NER_RUNTIME: Dict[str, Any] = {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_train_import_path() -> None:
    """确保可从项目根目录运行时导入 train 目录下模块。"""
    train_dir = _project_root() / "train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))


def _build_text(report: Dict[str, Any]) -> str:
    """拼接影像报告文本。"""
    c = report.get("content", {})
    return "\n".join(
        [
            f"描述: {c.get('描述', '')}",
            f"检查所见: {c.get('检查所见', '')}",
            f"检查提示: {c.get('检查提示', '')}",
        ]
    )


def _resolve_model_path(model_config: Dict[str, Any]) -> Path:
    imaging_cfg = model_config.get("imaging_extraction", {})
    configured = str(imaging_cfg.get("ner_model_path", "train/models/ner/best_model")).strip()
    p = Path(configured)
    if not p.is_absolute():
        p = _project_root() / p
    return p


def _load_local_ner_runtime(model_path: Path) -> Dict[str, Any]:
    cache_key = str(model_path.resolve())
    if cache_key in _NER_RUNTIME:
        return _NER_RUNTIME[cache_key]

    _ensure_train_import_path()
    from transformers import BertTokenizerFast
    from train.model import create_ner_model
    from train.utils import ID_TO_LABEL

    tokenizer = BertTokenizerFast.from_pretrained(str(model_path))

    # best_model 目录可能缺少 config.json，回退到训练基座模型目录。
    backbone_path = model_path if (model_path / "config.json").exists() else (_project_root() / "train" / "base-models" / "chinese-bert-wwm-ext")
    model = create_ner_model(model_path=backbone_path, num_labels=len(ID_TO_LABEL))

    safetensors_file = model_path / "model.safetensors"
    pytorch_file = model_path / "pytorch_model.bin"
    if safetensors_file.exists():
        from safetensors.torch import load_file as load_safetensors

        state = load_safetensors(str(safetensors_file))
    elif pytorch_file.exists():
        state = torch.load(pytorch_file, map_location="cpu")
    else:
        raise RuntimeError(f"missing model weight under {model_path}")

    model.load_state_dict(state, strict=False)
    model.eval()

    runtime = {
        "tokenizer": tokenizer,
        "model": model,
        "id_to_label": ID_TO_LABEL,
    }
    _NER_RUNTIME[cache_key] = runtime
    return runtime


def _bio_to_entities(text: str, labels: List[str], confs: List[float]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    i = 0
    n = len(labels)

    while i < n:
        tag = labels[i]
        if not tag.startswith("B-"):
            i += 1
            continue

        ent_type = tag[2:]
        j = i + 1
        while j < n and labels[j] == f"I-{ent_type}":
            j += 1

        entities.append(
            {
                "text": text[i:j],
                "start_pos": i,
                "end_pos": j,
                "type": ent_type,
                "confidence": float(max(confs[i:j]) if j > i else confs[i]),
            }
        )
        i = j

    return entities


def _predict_entities_with_local_ner(text: str, model_path: Path) -> List[Dict[str, Any]]:
    runtime = _load_local_ner_runtime(model_path)
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    id_to_label = runtime["id_to_label"]

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids"),
        )
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)[0].tolist()
        pred_confs = torch.max(probs, dim=-1)[0][0].tolist()

    offsets = encoding["offset_mapping"][0].tolist()
    char_labels = ["O"] * len(text)
    char_confs = [0.0] * len(text)

    for idx, offset in enumerate(offsets):
        start, end = int(offset[0]), int(offset[1])
        if start == 0 and end == 0:
            continue
        if start >= len(text) or end > len(text):
            continue

        tag = id_to_label.get(int(pred_ids[idx]), "O")
        conf = float(pred_confs[idx])
        if tag == "O":
            continue

        char_labels[start] = tag
        char_confs[start] = conf
        for p in range(start + 1, end):
            if p < len(text):
                char_labels[p] = tag.replace("B-", "I-")
                char_confs[p] = conf

    return _bio_to_entities(text, char_labels, char_confs)


def _extract_with_local_ner(report: Dict[str, Any], qc_rules: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
    findings = report.get("content", {}).get("检查所见", "")
    impression = report.get("content", {}).get("检查提示", "")
    text = f"{findings} {impression}".strip()
    if not text:
        raise RuntimeError("empty imaging text")

    model_path = _resolve_model_path(model_config)
    entities = _predict_entities_with_local_ner(text, model_path)

    anatomy_terms = []
    lesion_terms = []
    for e in entities:
        if e.get("type") == "ANATOMY" and e.get("text"):
            anatomy_terms.append(e["text"])
        if e.get("type") == "DISEASE" and e.get("text"):
            lesion_terms.append(e["text"])

    unique_anatomy = list(dict.fromkeys(anatomy_terms))
    unique_lesion = list(dict.fromkeys(lesion_terms))
    if not unique_anatomy and not unique_lesion:
        raise RuntimeError("local ner extracted no anatomy/disease entities")

    anatomy_id = {term: f"anat_{i}" for i, term in enumerate(unique_anatomy, start=1)}
    lesion_id = {term: f"les_{i}" for i, term in enumerate(unique_lesion, start=1)}

    nodes: List[Dict[str, Any]] = []
    for term in unique_anatomy:
        nodes.append({"id": anatomy_id[term], "text": term, "type": "anatomy"})
    for term in unique_lesion:
        nodes.append({"id": lesion_id[term], "text": term, "type": "lesion"})

    lesion_anatomy_map = qc_rules.get("imaging", {}).get("lesion_anatomy_map", {})
    sentences = _split_sentences(text)
    edges: List[Dict[str, Any]] = []
    edge_seen = set()

    for sentence in sentences:
        sentence_lesions = [l for l in unique_lesion if l in sentence]
        sentence_anatomy = [a for a in unique_anatomy if a in sentence]
        if not sentence_lesions:
            continue

        for lesion in sentence_lesions:
            preferred = [a for a in lesion_anatomy_map.get(lesion, []) if a in sentence_anatomy]
            targets = preferred or sentence_anatomy
            if not targets:
                targets = [a for a in lesion_anatomy_map.get(lesion, []) if a in unique_anatomy]
            for anatomy in targets:
                key = (lesion, anatomy)
                if key in edge_seen:
                    continue
                edge_seen.add(key)
                edges.append({"source": lesion_id[lesion], "target": anatomy_id[anatomy], "relation": "located_at"})

    return {
        "entities": {
            "患者信息": report.get("content", {}).get("描述", ""),
            "检查所见": findings,
            "检查提示": impression,
            "ner_entities": entities,
        },
        "nodes": nodes,
        "edges": edges,
        "relations": edges,
        "source": "train_ner_imaging",
        "degraded": False,
        "raw_output": {"model_path": str(model_path), "entity_count": len(entities)},
    }


def _parse_llm_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right != -1 and right > left:
            return json.loads(text[left : right + 1])
        raise


def _extract_frame_content(frame: Any) -> str:
    for attr in ["gen_text", "generated_text", "output", "content"]:
        value = getattr(frame, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    if hasattr(frame, "get_generated_text"):
        try:
            value = frame.get_generated_text()
            if isinstance(value, str) and value.strip():
                return value
        except Exception:
            pass

    if hasattr(frame, "frame") and isinstance(getattr(frame, "frame", None), dict):
        return json.dumps(frame.frame, ensure_ascii=False)

    if hasattr(frame, "data") and isinstance(getattr(frame, "data", None), dict):
        return json.dumps(frame.data, ensure_ascii=False)

    return str(frame)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[。！？!?；;\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def _build_vocab(qc_rules: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    lesion_anatomy_map = qc_rules.get("imaging", {}).get("lesion_anatomy_map", {})
    lesion_vocab = set(lesion_anatomy_map.keys())
    anatomy_vocab = set()
    for candidates in lesion_anatomy_map.values():
        anatomy_vocab.update(candidates)

    lesion_vocab.update(["结节", "病变", "梗死", "出血", "占位", "渗出", "炎症", "钙化", "斑片影"])
    anatomy_vocab.update(["肺", "右肺", "左肺", "上叶", "下叶", "脑", "额叶", "顶叶", "基底节", "肝", "肝右叶", "肝左叶"])

    return sorted(lesion_vocab, key=len, reverse=True), sorted(anatomy_vocab, key=len, reverse=True), lesion_anatomy_map


def _fallback_imaging_parse(report: Dict[str, Any], qc_rules: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    """LLM失败时的规则降级解析。"""
    findings = report.get("content", {}).get("检查所见", "")
    impression = report.get("content", {}).get("检查提示", "")
    text = f"{findings} {impression}"

    lesion_vocab, anatomy_vocab, lesion_anatomy_map = _build_vocab(qc_rules)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    anatomy_terms: List[str] = []
    lesion_terms: List[str] = []

    for term in anatomy_vocab:
        if term in text:
            anatomy_terms.append(term)
    for term in lesion_vocab:
        if term in text:
            lesion_terms.append(term)

    unique_anatomy = list(dict.fromkeys(anatomy_terms))
    unique_lesion = list(dict.fromkeys(lesion_terms))

    anatomy_id = {term: f"anat_{i}" for i, term in enumerate(unique_anatomy, start=1)}
    lesion_id = {term: f"les_{i}" for i, term in enumerate(unique_lesion, start=1)}

    for term in unique_anatomy:
        nodes.append({"id": anatomy_id[term], "text": term, "type": "anatomy"})
    for term in unique_lesion:
        nodes.append({"id": lesion_id[term], "text": term, "type": "lesion"})

    sentences = _split_sentences(text)
    edge_seen = set()

    for sentence in sentences:
        sentence_lesions = [l for l in unique_lesion if l in sentence]
        sentence_anatomy = [a for a in unique_anatomy if a in sentence]
        if not sentence_lesions:
            continue

        for lesion in sentence_lesions:
            preferred = [a for a in lesion_anatomy_map.get(lesion, []) if a in sentence_anatomy]
            targets = preferred or sentence_anatomy
            if not targets:
                targets = [a for a in lesion_anatomy_map.get(lesion, []) if a in unique_anatomy]

            for anatomy in targets:
                key = (lesion, anatomy)
                if key in edge_seen:
                    continue
                edge_seen.add(key)
                edges.append({"source": lesion_id[lesion], "target": anatomy_id[anatomy], "relation": "located_at"})

    return {
        "entities": {
            "患者信息": report.get("content", {}).get("描述", ""),
            "检查所见": findings,
            "检查提示": impression,
        },
        "nodes": nodes,
        "edges": edges,
        "relations": edges,
        "source": "fallback_imaging_parser",
        "degraded": True,
        "raw_output": _build_text(report),
        "error": reason,
    }


def _extract_with_llm(report: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
    llm_cfg = model_config.get("llm", {})
    api_key = os.getenv(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY for imaging llm extraction")

    try:
        from llm_ie.chunkers import SentenceUnitChunker
        from llm_ie.engines import LiteLLMInferenceEngine
        from llm_ie.extractors import DirectFrameExtractor
    except Exception as exc:
        raise RuntimeError(f"llm-ie unavailable: {exc}") from exc

    prompt = """
你是中文医疗影像信息抽取助手。
请从输入中抽取影像信息，并仅输出严格 JSON：
{
  "nodes": [
    {"text": "实体文本", "type": "lesion 或 anatomy"}
  ],
  "edges": [
    {"source_text": "病变实体文本", "target_text": "解剖实体文本", "relation": "located_at"}
  ]
}

要求：
1) 所有字段必须是中文原文中的短语。
2) 仅保留与影像病变-部位相关的实体。
3) 如果没有可抽取关系，edges 返回空数组。

报告文本如下：
{{text}}
"""

    text_input = _build_text(report)
    engine_kwargs = {
        "model": llm_cfg.get("model", "deepseek/deepseek-chat"),
        "api_key": api_key,
        "base_url": llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
        "timeout": float(llm_cfg.get("timeout", 20)),
    }
    try:
        engine = LiteLLMInferenceEngine(**engine_kwargs)
    except TypeError as exc:
        # 兼容旧版 llm-ie：LiteLLMInferenceEngine 可能不支持 timeout 参数
        if "unexpected keyword argument 'timeout'" not in str(exc):
            raise
        engine_kwargs.pop("timeout", None)
        engine = LiteLLMInferenceEngine(**engine_kwargs)
    extractor = DirectFrameExtractor(
        inference_engine=engine,
        unit_chunker=SentenceUnitChunker(),
        prompt_template=prompt,
    )
    frames = extractor.extract(text_input)
    if not frames:
        raise RuntimeError("llm returned empty frames")

    content = _extract_frame_content(frames[0])
    if not str(content).strip():
        raise RuntimeError("empty llm response content")
    parsed = _parse_llm_json(content)

    raw_nodes = parsed.get("nodes", []) if isinstance(parsed, dict) else []
    raw_edges = parsed.get("edges", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise RuntimeError("invalid llm json structure")

    nodes: List[Dict[str, Any]] = []
    text_to_id: Dict[str, str] = {}
    lesion_idx = 1
    anatomy_idx = 1

    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        node_type = str(item.get("type", "")).strip().lower()
        if not text or node_type not in {"lesion", "anatomy"}:
            continue
        if text in text_to_id:
            continue
        node_id = f"les_{lesion_idx}" if node_type == "lesion" else f"anat_{anatomy_idx}"
        if node_type == "lesion":
            lesion_idx += 1
        else:
            anatomy_idx += 1
        text_to_id[text] = node_id
        nodes.append({"id": node_id, "text": text, "type": node_type})

    edges: List[Dict[str, Any]] = []
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        source_text = str(item.get("source_text", "")).strip()
        target_text = str(item.get("target_text", "")).strip()
        relation = str(item.get("relation", "located_at")).strip() or "located_at"
        source_id = text_to_id.get(source_text)
        target_id = text_to_id.get(target_text)
        if not source_id or not target_id:
            continue
        edges.append({"source": source_id, "target": target_id, "relation": relation})

    if not nodes:
        raise RuntimeError("llm extracted empty nodes")

    return {
        "entities": {
            "患者信息": report.get("content", {}).get("描述", ""),
            "检查所见": report.get("content", {}).get("检查所见", ""),
            "检查提示": report.get("content", {}).get("检查提示", ""),
        },
        "nodes": nodes,
        "edges": edges,
        "relations": edges,
        "source": "llm_ie_imaging",
        "degraded": False,
        "raw_output": parsed,
    }


def extract_imaging_ner_re(
    report: Dict[str, Any],
    qc_rules: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """影像报告抽取主函数：支持 train_ner 与 llm_ie 两种后端。"""
    imaging_cfg = model_config.get("imaging_extraction", {})
    enabled = bool(imaging_cfg.get("enabled", True))
    allow_fallback = bool(imaging_cfg.get("fallback_enabled", True))
    backend = str(imaging_cfg.get("backend", "llm_ie")).strip().lower()

    if not enabled:
        return _fallback_imaging_parse(report, qc_rules, reason="imaging llm extraction disabled by config")

    try:
        if backend == "train_ner":
            return _extract_with_local_ner(report, qc_rules, model_config)
        return _extract_with_llm(report, model_config)
    except Exception as exc:
        if allow_fallback:
            return _fallback_imaging_parse(report, qc_rules, reason=str(exc))
        raise