from typing import Any, Dict, List


def generate_standard_medical_record(graded_record: Dict[str, Any], corrected: bool = False) -> Dict[str, Any]:
    grade = graded_record.get("grade_label", "")
    extraction = graded_record.get("extraction", {})
    entities = extraction.get("entities", {})
    issues: List[Dict[str, Any]] = graded_record.get("qc_issues", [])

    can_generate = grade == "正常级" or corrected

    base_record = {
        "report_id": graded_record.get("report_id"),
        "report_type": graded_record.get("report_type"),
        "report_subtype": graded_record.get("report_subtype"),
        "患者信息": entities.get("患者信息", graded_record.get("content", {}).get("描述", "")),
        "检查结果": entities.get("检查所见", graded_record.get("content", {}).get("检查所见", "")),
        "异常提示": entities.get("检查提示", graded_record.get("content", {}).get("检查提示", "")),
        "建议": "结合临床随访。",
    }

    if can_generate:
        return {
            "generated": True,
            "record": base_record,
            "pending_issues": [],
        }

    return {
        "generated": False,
        "record": base_record,
        "pending_issues": issues,
        "note": "报告存在质控问题，已标注问题，待修正后可生成正式病历。",
    }
