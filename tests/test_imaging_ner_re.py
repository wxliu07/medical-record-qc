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
