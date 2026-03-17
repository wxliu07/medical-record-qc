import json
from pathlib import Path
from typing import Any, Dict, List


def load_json_file(file_path: Path) -> Any:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    qc_rules = load_json_file(config_dir / "qc_rules.json")
    model_config = load_json_file(config_dir / "model_config.json")
    return {
        "qc_rules": qc_rules,
        "model_config": model_config,
    }


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate_and_normalize_reports(raw_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    required_top = ["report_id", "report_type", "report_subtype", "content", "label"]
    required_content = ["描述", "检查所见", "检查提示"]

    for item in raw_reports:
        missing_top = [k for k in required_top if k not in item]
        if missing_top:
            raise ValueError(f"报告缺少顶层字段: {missing_top}, report={item}")

        content = item.get("content", {})
        if not isinstance(content, dict):
            raise ValueError(f"content 必须为对象: report_id={item.get('report_id')}")

        for field in required_content:
            if field not in content:
                content[field] = ""

        report_type = _normalize_text(item.get("report_type"))
        if report_type not in {"指标类", "影像类"}:
            raise ValueError(f"不支持的 report_type: {report_type}, report_id={item.get('report_id')}")

        normalized.append(
            {
                "report_id": _normalize_text(item.get("report_id")),
                "report_type": report_type,
                "report_subtype": _normalize_text(item.get("report_subtype")),
                "content": {
                    "描述": _normalize_text(content.get("描述")),
                    "检查所见": _normalize_text(content.get("检查所见")),
                    "检查提示": _normalize_text(content.get("检查提示")),
                },
                "label": _normalize_text(item.get("label")),
            }
        )

    return normalized


def load_simulated_reports(data_file: Path) -> List[Dict[str, Any]]:
    raw = load_json_file(data_file)
    if not isinstance(raw, list):
        raise ValueError("simulate_data.json 顶层必须为列表")
    return validate_and_normalize_reports(raw)
