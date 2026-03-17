import json
import os
import re
from typing import Any, Dict, List


def _build_text(report: Dict[str, Any]) -> str:
    c = report.get("content", {})
    return "\n".join(
        [
            f"描述: {c.get('描述', '')}",
            f"检查所见: {c.get('检查所见', '')}",
            f"检查提示: {c.get('检查提示', '')}",
        ]
    )


def _extract_with_regex(report: Dict[str, Any], qc_rules: Dict[str, Any]) -> Dict[str, Any]:
    findings = report.get("content", {}).get("检查所见", "")
    indicator_rules = qc_rules.get("indicator", {}).get("ranges", {})

    entities: Dict[str, Any] = {
        "患者信息": report.get("content", {}).get("描述", ""),
        "检查所见": findings,
        "检查提示": report.get("content", {}).get("检查提示", ""),
        "指标": [],
    }
    relations: List[Dict[str, Any]] = []

    for name, cfg in indicator_rules.items():
        pattern = rf"{re.escape(name)}\s*([0-9]+(?:\.[0-9]+)?)\s*([\^0-9A-Za-z/]+)?"
        match = re.search(pattern, findings)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2) or cfg.get("unit", "")
        low = float(cfg.get("low", 0))
        high = float(cfg.get("high", 0))
        status = "正常" if low <= value <= high else "异常"

        entities["指标"].append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "reference_range": [low, high],
                "status": status,
            }
        )
        relations.append(
            {
                "head": name,
                "tail": f"{value} {unit}".strip(),
                "relation": "指标-数值",
                "status": status,
            }
        )

    return {
        "entities": entities,
        "relations": relations,
        "source": "regex_fallback",
        "degraded": True,
        "raw_output": None,
    }


def _extract_with_llm(report: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
    llm_cfg = model_config.get("llm", {})
    api_key = os.getenv(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found")

    try:
        from llm_ie.chunkers import SentenceUnitChunker
        from llm_ie.engines import LiteLLMInferenceEngine
        from llm_ie.extractors import DirectFrameExtractor
    except Exception as exc:
        raise RuntimeError(f"llm-ie import failed: {exc}") from exc

    prompt = """
你是医疗信息抽取助手。请从下面的指标类检查报告中抽取实体与关系，并输出严格JSON：
{
  "entities": {
    "患者信息": "",
    "检查所见": "",
    "检查提示": "",
    "指标": [
      {"name":"白细胞计数","value":6.2,"unit":"10^9/L","reference_range":[3.5,9.5],"status":"正常"}
    ]
  },
  "relations": [
    {"head":"白细胞计数","tail":"6.2 10^9/L","relation":"指标-数值","status":"正常"}
  ]
}
报告如下：
{{text}}
"""

    engine = LiteLLMInferenceEngine(
        model=llm_cfg.get("model", "deepseek/deepseek-chat"),
        api_key=api_key,
        base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
    )
    extractor = DirectFrameExtractor(
        inference_engine=engine,
        unit_chunker=SentenceUnitChunker(),
        prompt_template=prompt,
    )

    frames = extractor.extract(_build_text(report))
    if not frames:
        raise RuntimeError("llm-ie returned empty frames")

    frame = frames[0]
    content = None
    for attr in ["gen_text", "generated_text", "output", "content"]:
        val = getattr(frame, attr, None)
        if isinstance(val, str) and val.strip():
            content = val
            break
    if content is None and hasattr(frame, "get_generated_text"):
        try:
            content = frame.get_generated_text()
        except Exception:
            content = None
    if content is None:
        content = str(frame)

    clean = content.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]

    result = json.loads(clean.strip())
    result.setdefault("source", "llm_ie")
    result.setdefault("degraded", False)
    result.setdefault("raw_output", content)
    return result


def extract_indicator_ner_re(
    report: Dict[str, Any],
    qc_rules: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return _extract_with_llm(report, model_config)
    except Exception as exc:
        fallback = _extract_with_regex(report, qc_rules)
        fallback["error"] = str(exc)
        return fallback
