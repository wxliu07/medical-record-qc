from modules.dataset.grade_dataset import aggregate_grade_dataset, build_graded_record


def test_grade_priority_contradiction_over_missing() -> None:
    report = {
        "report_id": "T-1",
        "report_type": "指标类",
        "report_subtype": "血常规",
        "label": "矛盾",
        "content": {"描述": "a", "检查所见": "b", "检查提示": "c"},
    }
    extraction = {"entities": {}, "relations": []}
    issues = [
        {"type": "缺失", "message": "缺字段"},
        {"type": "矛盾", "message": "逻辑冲突"},
    ]

    graded = build_graded_record(report, extraction, issues, {"result": "error"})
    assert graded["grade_label"] == "矛盾级"


def test_aggregate_grade_dataset_summary() -> None:
    records = [
        {"grade_label": "正常级"},
        {"grade_label": "缺失级"},
        {"grade_label": "矛盾级"},
        {"grade_label": "矛盾级"},
    ]
    data = aggregate_grade_dataset(records)
    assert data["summary"] == {"正常级": 1, "缺失级": 1, "矛盾级": 2}
