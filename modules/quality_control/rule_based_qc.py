from typing import Any, Dict, List


Issue = Dict[str, Any]


def _new_issue(issue_type: str, severity: str, message: str, evidence: Any, source: str = "rule") -> Issue:
    return {
        "type": issue_type,
        "severity": severity,
        "message": message,
        "evidence": evidence,
        "source": source,
    }


def _check_completeness(content: Dict[str, Any], qc_rules: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    required = qc_rules.get("completeness", {}).get("required_fields", [])
    for field in required:
        value = str(content.get(field, "")).strip()
        if not value:
            issues.append(
                _new_issue(
                    issue_type="缺失",
                    severity="high",
                    message=f"缺失关键字段: {field}",
                    evidence={"field": field},
                )
            )
    return issues


def _indicator_conflicts(extraction: Dict[str, Any], qc_rules: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    indicator_rules = qc_rules.get("indicator", {}).get("ranges", {})
    abnormal_keywords = qc_rules.get("indicator", {}).get("abnormal_keywords", [])
    normal_keywords = qc_rules.get("indicator", {}).get("normal_keywords", [])

    entities = extraction.get("entities", {})
    indicators = entities.get("指标", [])
    prompt_text = str(entities.get("检查提示", ""))

    for item in indicators:
        name = item.get("name")
        value = item.get("value")
        if name not in indicator_rules:
            continue
        try:
            value_f = float(value)
        except Exception:
            continue
        low = float(indicator_rules[name]["low"])
        high = float(indicator_rules[name]["high"])
        if value_f < low or value_f > high:
            issues.append(
                _new_issue(
                    issue_type="矛盾",
                    severity="high",
                    message=f"{name}数值异常: {value_f}, 正常范围[{low}, {high}]",
                    evidence={"name": name, "value": value_f, "range": [low, high]},
                )
            )

            if any(k in prompt_text for k in normal_keywords):
                issues.append(
                    _new_issue(
                        issue_type="矛盾",
                        severity="high",
                        message=f"{name}异常但结论提示正常",
                        evidence={"name": name, "提示": prompt_text},
                    )
                )

    if any(k in prompt_text for k in abnormal_keywords) and not indicators:
        issues.append(
            _new_issue(
                issue_type="缺失",
                severity="medium",
                message="提示存在异常但未抽取到指标实体",
                evidence={"提示": prompt_text},
            )
        )

    return issues


def _imaging_conflicts(report: Dict[str, Any], extraction: Dict[str, Any], qc_rules: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    entities = extraction.get("entities", {})
    findings = str(entities.get("检查所见", report.get("content", {}).get("检查所见", "")))
    impression = str(entities.get("检查提示", report.get("content", {}).get("检查提示", "")))

    img_rules = qc_rules.get("imaging", {})
    normal_keywords = img_rules.get("normal_keywords", [])
    abnormal_keywords = img_rules.get("abnormal_keywords", [])

    findings_normal = any(k in findings for k in normal_keywords)
    impression_abnormal = any(k in impression for k in abnormal_keywords)
    if findings_normal and impression_abnormal:
        issues.append(
            _new_issue(
                issue_type="矛盾",
                severity="high",
                message="影像所见提示正常，但结论提示异常",
                evidence={"检查所见": findings, "检查提示": impression},
            )
        )

    lesion_anatomy_map = img_rules.get("lesion_anatomy_map", {})
    text = f"{findings} {impression}"
    for lesion, anatomy_candidates in lesion_anatomy_map.items():
        if lesion in text and not any(a in text for a in anatomy_candidates):
            issues.append(
                _new_issue(
                    issue_type="矛盾",
                    severity="medium",
                    message=f"病变与解剖结构不匹配: {lesion}",
                    evidence={"lesion": lesion, "expected_anatomy": anatomy_candidates},
                )
            )

    return issues


def run_rule_based_qc(report: Dict[str, Any], extraction: Dict[str, Any], qc_rules: Dict[str, Any]) -> List[Issue]:
    content = report.get("content", {})
    issues: List[Issue] = []

    issues.extend(_check_completeness(content, qc_rules))

    report_type = report.get("report_type")
    if report_type == "指标类":
        issues.extend(_indicator_conflicts(extraction, qc_rules))
    elif report_type == "影像类":
        issues.extend(_imaging_conflicts(report, extraction, qc_rules))

    return issues
