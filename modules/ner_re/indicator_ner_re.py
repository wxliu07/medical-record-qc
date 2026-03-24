import json
import os
import re
from multiprocessing import get_context
from typing import Any, Dict, List


_LLM_BACKEND_ERROR: str = ""


def _build_text(report: Dict[str, Any]) -> str:
    c = report.get("content", {})
    return "\n".join(
        [
            f"描述: {c.get('描述', '')}",
            f"检查所见: {c.get('检查所见', '')}",
            f"检查提示: {c.get('检查提示', '')}",
        ]
    )


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

    frames = extractor.extract(_build_text(report))
    if not frames:
        raise RuntimeError("llm-ie returned empty frames")

    content = _extract_frame_content(frames[0])
    result = _parse_llm_json(content)
    result.setdefault("source", "llm_ie")
    result.setdefault("degraded", False)
    result.setdefault("raw_output", content)
    return result


def _llm_worker(report: Dict[str, Any], model_config: Dict[str, Any], queue: Any) -> None:
    try:
        result = _extract_with_llm(report, model_config)
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _extract_with_llm_timeout(report: Dict[str, Any], model_config: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    ctx = get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_llm_worker, args=(report, model_config, queue), daemon=True)
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
        raise TimeoutError(f"indicator llm_ie timeout after {timeout_seconds}s")

    if queue.empty():
        raise RuntimeError("indicator llm_ie worker exited without result")

    payload = queue.get()
    if payload.get("ok"):
        return payload["result"]
    raise RuntimeError(str(payload.get("error", "indicator llm_ie failed")))


def extract_indicator_ner_re(
    report: Dict[str, Any],
    qc_rules: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    global _LLM_BACKEND_ERROR

    indicator_cfg = model_config.get("indicator_extraction", {})
    timeout_seconds = float(indicator_cfg.get("llm_timeout_seconds", 6))

    # llm_ie 在当前进程已判定不可用时，避免每条报告重复等待超时。
    if _LLM_BACKEND_ERROR:
        fallback = _extract_with_regex(report, qc_rules)
        fallback["error"] = f"llm_ie unavailable in current run: {_LLM_BACKEND_ERROR}"
        return fallback

    try:
        return _extract_with_llm_timeout(report, model_config, timeout_seconds=timeout_seconds)
    except Exception as exc:
        _LLM_BACKEND_ERROR = str(exc)
        fallback = _extract_with_regex(report, qc_rules)
        fallback["error"] = str(exc)
        return fallback
