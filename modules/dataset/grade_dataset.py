from typing import Any, Dict, List


def _grade_from_issues(issues: List[Dict[str, Any]]) -> str:
    has_contradiction = any(i.get("type") == "矛盾" for i in issues)
    has_missing = any(i.get("type") == "缺失" for i in issues)

    if has_contradiction:
        return "矛盾级"
    if has_missing:
        return "缺失级"
    return "正常级"


def build_graded_record(
    report: Dict[str, Any],
    extraction: Dict[str, Any],
    rule_issues: List[Dict[str, Any]],
    reasoning: Dict[str, Any],
) -> Dict[str, Any]:
    grade = _grade_from_issues(rule_issues)
    return {
        "report_id": report.get("report_id"),
        "report_type": report.get("report_type"),
        "report_subtype": report.get("report_subtype"),
        "original_label": report.get("label"),
        "grade_label": grade,
        "content": report.get("content", {}),
        "extraction": extraction,
        "qc_issues": rule_issues,
        "reasoning": reasoning,
    }


def aggregate_grade_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {"正常级": 0, "缺失级": 0, "矛盾级": 0}
    for r in records:
        grade = r.get("grade_label")
        if grade in summary:
            summary[grade] += 1

    return {
        "summary": summary,
        "records": records,
    }
