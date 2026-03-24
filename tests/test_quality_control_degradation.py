from modules.quality_control.llm_reasoning_qc import run_llm_reasoning_qc
from modules.quality_control.rule_based_qc import run_rule_based_qc


def _sample_imaging_report() -> dict:
    return {
        "report_id": "TEST001",
        "report_type": "影像类",
        "report_subtype": "CT",
        "content": {
            "描述": "头部CT平扫",
            "检查所见": "脑实质未见明显异常密度影，脑室系统未见扩大，中线结构居中。",
            "检查提示": "颅内CT平扫未见明显异常。",
        },
        "label": "正常",
    }


def test_rule_qc_flags_degraded_imaging_extraction() -> None:
    report = _sample_imaging_report()
    extraction = {
        "entities": {
            "患者信息": report["content"]["描述"],
            "检查所见": report["content"]["检查所见"],
            "检查提示": report["content"]["检查提示"],
        },
        "nodes": [{"id": "anat_1", "text": "脑", "type": "anatomy"}],
        "edges": [],
        "relations": [],
        "source": "fallback_imaging_parser",
        "degraded": True,
        "error": "missing DEEPSEEK_API_KEY for imaging llm extraction",
    }
    qc_rules = {
        "completeness": {"required_fields": ["描述", "检查所见", "检查提示"]},
        "imaging": {
            "normal_keywords": ["未见明显异常"],
            "abnormal_keywords": ["病变", "异常", "占位", "结节", "梗死", "出血"],
            "lesion_anatomy_map": {},
        },
    }

    issues = run_rule_based_qc(report, extraction, qc_rules)
    assert any(i.get("message") == "影像抽取降级，需人工复核后再生成正式病历" for i in issues)


def test_reasoning_not_fastpath_when_degraded_and_issues(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    report = _sample_imaging_report()
    extraction = {"degraded": True}
    rule_issues = [
        {
            "type": "缺失",
            "severity": "high",
            "message": "影像抽取降级，需人工复核后再生成正式病历",
            "evidence": {"source": "fallback_imaging_parser"},
            "source": "system",
        }
    ]
    model_config = {
        "llm": {
            "enable_reasoning_qc": True,
            "api_key_env": "DEEPSEEK_API_KEY",
            "reasoning_on_clean_reports": False,
        }
    }

    result = run_llm_reasoning_qc(report, extraction, rule_issues, model_config)
    assert result["source"] == "rule_fallback"
    assert result["result"] == "error"
