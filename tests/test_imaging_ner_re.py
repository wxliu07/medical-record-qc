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


def test_imaging_fallback_when_disabled() -> None:
    report = _sample_report()
    qc_rules = {"imaging": {}}
    model_config = {"radgraph": {"enabled": False, "fallback_enabled": True}}

    result = extract_imaging_ner_re(report, qc_rules, model_config)
    assert result["degraded"] is True
    assert result["source"] == "fallback_imaging_parser"
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)


def test_imaging_radgraph_passes_model_type(monkeypatch) -> None:
    import modules.ner_re.imaging_ner_re as imaging_ner_re

    captured = {}

    class FakeRadGraph:
        def __init__(self, model_type=None):
            captured["model_type"] = model_type

        def __call__(self, texts):
            return {
                "0": {
                    "entities": {
                        "1": {"label": "Observation", "tokens": "结节", "relations": []}
                    }
                }
            }

    class FakeModule:
        RadGraph = FakeRadGraph

    def fake_import_module(name):
        assert name == "radgraph"
        return FakeModule

    monkeypatch.setattr(imaging_ner_re.importlib, "import_module", fake_import_module)

    report = _sample_report()
    qc_rules = {"imaging": {}}
    model_config = {
        "radgraph": {
            "enabled": True,
            "fallback_enabled": True,
            "model_type": "radgraph-xl",
        }
    }

    result = extract_imaging_ner_re(report, qc_rules, model_config)

    assert captured["model_type"] == "radgraph-xl"
    assert result["degraded"] is False
    assert result["source"] == "radgraph"
