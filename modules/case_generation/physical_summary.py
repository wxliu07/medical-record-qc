from typing import Any, Dict


def generate_physical_summary(graded_record: Dict[str, Any], corrected: bool = False) -> Dict[str, Any]:
    grade = graded_record.get("grade_label", "")
    extraction = graded_record.get("extraction", {})
    entities = extraction.get("entities", {})

    allowed = grade == "正常级" or corrected
    if not allowed:
        return {
            "generated": False,
            "summary": "",
            "note": "报告存在缺失/矛盾问题，需修正后再生成体检总结。",
        }

    patient_info = entities.get("患者信息") or graded_record.get("content", {}).get("描述", "")
    indicators = entities.get("指标", [])
    prompt = entities.get("检查提示", graded_record.get("content", {}).get("检查提示", ""))

    metric_desc = []
    for item in indicators:
        name = item.get("name")
        value = item.get("value")
        unit = item.get("unit", "")
        status = item.get("status", "")
        metric_desc.append(f"{name}{value}{unit}({status})")

    summary = f"{patient_info}"
    if metric_desc:
        summary += " 主要指标：" + "；".join(metric_desc) + "。"
    if prompt:
        summary += f" 结论：{prompt}"

    return {
        "generated": True,
        "summary": summary.strip(),
        "note": "",
    }
