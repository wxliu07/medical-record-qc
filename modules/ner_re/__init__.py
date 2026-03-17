from typing import Any, Dict

from .indicator_ner_re import extract_indicator_ner_re
from .imaging_ner_re import extract_imaging_ner_re


def extract_by_report_type(
    report: Dict[str, Any],
    qc_rules: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    report_type = report.get("report_type")
    if report_type == "指标类":
        return extract_indicator_ner_re(report, qc_rules, model_config)
    if report_type == "影像类":
        return extract_imaging_ner_re(report, qc_rules, model_config)

    return {
        "entities": {},
        "relations": [],
        "nodes": [],
        "edges": [],
        "source": "unknown",
        "degraded": True,
        "error": f"unsupported report_type: {report_type}",
    }
