from modules.ner_re.imaging_ner_re import extract_imaging_ner_re


def _sample_report() -> dict:
    return {
        "report_id": "IMG-T",
        "report_type": "影像类",
        "report_subtype": "CT",
        "content": {
            "描述": "胸部CT",
            "检查所见": "右肺结节",
            "检查提示": "建议复查",
        },
        "label": "矛盾",
    }


def test_imaging_fallback_when_no_api_key() -> None:
    report = _sample_report()
    qc_rules = {
        "imaging": {
            "lesion_anatomy_map": {
                "结节": ["肺", "右肺", "左肺"],
            }
        }
    }
    model_config = {
        "llm": {"api_key_env": "DEEPSEEK_API_KEY"},
        "imaging_extraction": {"enabled": True, "fallback_enabled": True},
    }

    result = extract_imaging_ner_re(report, qc_rules, model_config)
    assert result["degraded"] is True
    assert result["source"] == "fallback_imaging_parser"
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)
    assert "missing DEEPSEEK_API_KEY" in str(result.get("error", ""))


def test_imaging_fallback_uses_sentence_level_relation() -> None:
    report = {
        "report_id": "IMG-T2",
        "report_type": "影像类",
        "report_subtype": "CT",
        "content": {
            "描述": "胸部CT",
            "检查所见": "右肺见结节。肝实质未见异常。",
            "检查提示": "建议随访。",
        },
        "label": "矛盾",
    }
    qc_rules = {
        "imaging": {
            "lesion_anatomy_map": {
                "结节": ["肺", "右肺", "左肺"],
            }
        }
    }
    model_config = {
        "llm": {"api_key_env": "DEEPSEEK_API_KEY"},
        "imaging_extraction": {"enabled": True, "fallback_enabled": True},
    }

    result = extract_imaging_ner_re(report, qc_rules, model_config)
    assert result["degraded"] is True
    assert result["source"] == "fallback_imaging_parser"

    node_text = {n["id"]: n["text"] for n in result["nodes"]}
    linked_pairs = {(node_text[e["source"]], node_text[e["target"]]) for e in result["edges"]}
    assert ("结节", "右肺") in linked_pairs or ("结节", "肺") in linked_pairs
    assert ("结节", "肝") not in linked_pairs


def test_imaging_use_llm_result_when_available(monkeypatch) -> None:
    import modules.ner_re.imaging_ner_re as imaging_ner_re

    mock_result = {
        "entities": {"患者信息": "胸部CT", "检查所见": "右肺结节", "检查提示": "建议复查"},
        "nodes": [
            {"id": "les_1", "text": "结节", "type": "lesion"},
            {"id": "anat_1", "text": "右肺", "type": "anatomy"},
        ],
        "edges": [{"source": "les_1", "target": "anat_1", "relation": "located_at"}],
        "relations": [{"source": "les_1", "target": "anat_1", "relation": "located_at"}],
        "source": "llm_ie_imaging",
        "degraded": False,
        "raw_output": {},
    }
    monkeypatch.setattr(imaging_ner_re, "_extract_with_llm", lambda report, model_config: mock_result)

    report = _sample_report()
    qc_rules = {"imaging": {"lesion_anatomy_map": {"结节": ["右肺"]}}}
    model_config = {
        "imaging_extraction": {
            "enabled": True,
            "fallback_enabled": True,
        }
    }

    result = extract_imaging_ner_re(report, qc_rules, model_config)

    assert result["degraded"] is False
    assert result["source"] == "llm_ie_imaging"


def test_imaging_llm_prompt_has_single_placeholder(monkeypatch) -> None:
    import sys
    import types
    import modules.ner_re.imaging_ner_re as imaging_ner_re

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    class FakeEngine:
        def __init__(self, model=None, api_key=None, base_url=None):
            self.model = model
            self.api_key = api_key
            self.base_url = base_url

    class FakeExtractor:
        def __init__(self, inference_engine=None, unit_chunker=None, prompt_template=None):
            # llm-ie 要求：text_content 为 str 时，必须有且仅有一个占位符
            assert prompt_template.count("{{text}}") == 1

        def extract(self, text):
            frame = types.SimpleNamespace(
                content='{"nodes":[{"text":"结节","type":"lesion"},{"text":"右肺","type":"anatomy"}],"edges":[{"source_text":"结节","target_text":"右肺","relation":"located_at"}]}'
            )
            return [frame]

    fake_chunkers = types.SimpleNamespace(SentenceUnitChunker=type("SentenceUnitChunker", (), {}))
    fake_engines = types.SimpleNamespace(LiteLLMInferenceEngine=FakeEngine)
    fake_extractors = types.SimpleNamespace(DirectFrameExtractor=FakeExtractor)

    monkeypatch.setitem(sys.modules, "llm_ie.chunkers", fake_chunkers)
    monkeypatch.setitem(sys.modules, "llm_ie.engines", fake_engines)
    monkeypatch.setitem(sys.modules, "llm_ie.extractors", fake_extractors)

    report = {
        "report_id": "IMG-T3",
        "report_type": "影像类",
        "report_subtype": "CT",
        "content": {
            "描述": "胸部CT",
            "检查所见": "右肺结节",
            "检查提示": "建议复查",
        },
        "label": "异常",
    }
    model_config = {
        "llm": {
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": "deepseek/deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "timeout": 20,
        }
    }

    result = imaging_ner_re._extract_with_llm(report, model_config)
    assert result["degraded"] is False
    assert result["source"] == "llm_ie_imaging"
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
