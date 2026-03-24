import json
import os
from typing import Any, Dict, List


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
        # Some responses may include extra prose before/after JSON; try to recover the first object block.
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


def _rule_reasoning_fallback(rule_issues: List[Dict[str, Any]], cause: str) -> Dict[str, Any]:
    if rule_issues:
        first = rule_issues[0]
        message = str(first.get("message", "存在规则质控问题")).strip()
        return {
            "result": "error",
            "reason": f"rule fallback: {message}; cause={cause}",
            "source": "rule_fallback",
        }

    return {
        "result": "correct",
        "reason": f"rule fallback: no issues; cause={cause}",
        "source": "rule_fallback",
    }


def run_llm_reasoning_qc(
    report: Dict[str, Any],
    extraction: Dict[str, Any],
    rule_issues: List[Dict[str, Any]],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    llm_cfg = model_config.get("llm", {})
    degraded = bool(extraction.get("degraded", False))

    # Fast path: for clean reports, avoid network/model calls to reduce blocking risk.
    if not rule_issues and not degraded and llm_cfg.get("reasoning_on_clean_reports", False) is False:
        return {
            "result": "correct",
            "reason": "rule fastpath: no issues",
            "source": "rule_fastpath",
        }

    if not llm_cfg.get("enable_reasoning_qc", True):
        return _rule_reasoning_fallback(rule_issues, "reasoning qc disabled by config")

    api_key = os.getenv(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        return _rule_reasoning_fallback(rule_issues, "missing DEEPSEEK_API_KEY")

    prompt = """
你是医疗报告质控复核助手。请基于输入信息判断当前结论是否合理。
仅输出严格 JSON，不要附加任何解释文本：
{
  "result": "correct 或 error",
  "reason": "简要原因"
}

输入信息如下：
{{text}}
"""

    try:
        from llm_ie.chunkers import SentenceUnitChunker
        from llm_ie.engines import LiteLLMInferenceEngine
        from llm_ie.extractors import DirectFrameExtractor
    except Exception as exc:
        return _rule_reasoning_fallback(rule_issues, f"llm-ie unavailable: {exc}")

    reasoning_input = json.dumps(
        {
            "report": report,
            "extraction": extraction,
            "rule_issues": rule_issues,
        },
        ensure_ascii=False,
    )

    try:
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
        # llm-ie requires one placeholder in prompt when text_content is a string.
        frames = extractor.extract(reasoning_input)
        if not frames:
            return _rule_reasoning_fallback(rule_issues, "llm returned empty frames")

        frame = frames[0]
        content = _extract_frame_content(frame)

        result = _parse_llm_json(content)
        if "result" not in result:
            result["result"] = "error" if rule_issues else "correct"
        if "reason" not in result:
            result["reason"] = "llm response missing reason"
        result["source"] = "llm_ie"
        return result
    except json.JSONDecodeError:
        return _rule_reasoning_fallback(rule_issues, "llm response not valid json")
    except Exception as exc:
        return _rule_reasoning_fallback(rule_issues, f"llm reasoning failed: {type(exc).__name__}")
