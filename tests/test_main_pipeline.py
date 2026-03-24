import json
from pathlib import Path

import main


def test_run_pipeline_with_mocks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        main,
        "load_config",
        lambda _: {
            "qc_rules": {"completeness": {"required_fields": ["描述", "检查所见", "检查提示"]}},
            "model_config": {"llm": {}, "imaging_extraction": {"enabled": True, "fallback_enabled": True}},
        },
    )

    mock_reports = [
        {
            "report_id": "R-1",
            "report_type": "指标类",
            "report_subtype": "血常规",
            "content": {"描述": "男，36岁，体检复查。", "检查所见": "白细胞计数6.0 10^9/L。", "检查提示": "未见明显异常。"},
            "label": "正常",
        },
        {
            "report_id": "R-2",
            "report_type": "影像类",
            "report_subtype": "CT",
            "content": {"描述": "胸部CT复查。", "检查所见": "双肺纹理清晰，未见明显结节。", "检查提示": "未见明显异常。"},
            "label": "正常",
        },
    ]
    monkeypatch.setattr(main, "load_simulated_reports", lambda _: mock_reports)

    monkeypatch.setattr(
        main,
        "extract_by_report_type",
        lambda report, qc_rules, model_config: {
            "entities": {
                "患者信息": report["content"]["描述"],
                "检查所见": report["content"]["检查所见"],
                "检查提示": report["content"]["检查提示"],
                "指标": [],
            },
            "relations": [],
            "nodes": [],
            "edges": [],
            "degraded": False,
        },
    )
    monkeypatch.setattr(main, "run_rule_based_qc", lambda report, extraction, qc_rules: [])
    monkeypatch.setattr(main, "run_llm_reasoning_qc", lambda report, extraction, issues, model_cfg: {"result": "correct"})
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)

    result = main.run_pipeline()

    assert result["total_reports"] == 2
    assert result["summary"]["正常级"] == 2
    assert result["degraded_count"] == 0

    out_dir = Path(result["output_dir"])
    assert (out_dir / "graded_dataset.json").exists()
    assert (out_dir / "physical_summaries.json").exists()
    assert (out_dir / "medical_records.json").exists()
    assert (out_dir / "run_summary.json").exists()

    with (out_dir / "run_summary.json").open("r", encoding="utf-8") as f:
        run_summary = json.load(f)
    assert run_summary["total_reports"] == 2
