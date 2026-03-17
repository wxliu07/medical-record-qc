import json
import os
from typing import Any, Dict, List


def run_llm_reasoning_qc(
    report: Dict[str, Any],
    extraction: Dict[str, Any],
    rule_issues: List[Dict[str, Any]],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    llm_cfg = model_config.get("llm", {})
    if not llm_cfg.get("enable_reasoning_qc", True):
        return {
            "result": "unknown",
            "reason": "reasoning qc disabled by config",
            "source": "config",
        }

    api_key = os.getenv(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        return {
            "result": "unknown",
            "reason": "missing DEEPSEEK_API_KEY",
            "source": "fallback",
        }

    try:
        from llm_ie.chunkers import SentenceUnitChunker
        from llm_ie.engines import LiteLLMInferenceEngine
        from llm_ie.extractors import DirectFrameExtractor
    except Exception as exc:
        return {
            "result": "unknown",
            "reason": f"llm-ie unavailable: {exc}",
            "source": "fallback",
        }

    prompt = f"""
你是医疗报告质控复核助手。基于已有抽取与规则问题，判断当前结论是否合理。
请输出严格JSON：
{{
  "result": "correct 或 error",
  "reason": "简要原因"
}}

报告：
{json.dumps(report, ensure_ascii=False)}

抽取结果：
{json.dumps(extraction, ensure_ascii=False)}

规则问题：
{json.dumps(rule_issues, ensure_ascii=False)}
"""

    try:
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
        frames = extractor.extract("")
        if not frames:
            return {"result": "unknown", "reason": "llm returned empty frames", "source": "llm_ie"}

        frame = frames[0]
        content = None
        for attr in ["gen_text", "generated_text", "output", "content"]:
            value = getattr(frame, attr, None)
            if isinstance(value, str) and value.strip():
                content = value
                break
        if content is None and hasattr(frame, "get_generated_text"):
            try:
                content = frame.get_generated_text()
            except Exception:
                content = None

        if content is None:
            content = str(frame)

        text = content.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        result = json.loads(text.strip())
        if "result" not in result:
            result["result"] = "unknown"
        if "reason" not in result:
            result["reason"] = "llm response missing reason"
        result["source"] = "llm_ie"
        return result
    except Exception as exc:
        return {
            "result": "unknown",
            "reason": f"llm reasoning failed: {exc}",
            "source": "fallback",
        }
