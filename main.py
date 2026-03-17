import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_FILE = BASE_DIR / "data" / "simulate_data.json"
OUTPUT_DIR = BASE_DIR / "outputs"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.case_generation import generate_physical_summary, generate_standard_medical_record
from modules.data_process.data_loader import load_config, load_simulated_reports
from modules.dataset import aggregate_grade_dataset, build_graded_record
from modules.ner_re import extract_by_report_type
from modules.quality_control import run_llm_reasoning_qc, run_rule_based_qc


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _apply_env_model_overrides(model_config: Dict[str, Any]) -> None:
    llm_cfg = model_config.setdefault("llm", {})
    env_model = os.getenv("DEEPSEEK_MODEL", "").strip()
    env_base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()

    if env_model:
        llm_cfg["model"] = env_model
    if env_base_url:
        llm_cfg["base_url"] = env_base_url


def _save_json(data: Any, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_pipeline() -> Dict[str, Any]:
    _load_env_file(BASE_DIR / ".env")

    cfg = load_config(CONFIG_DIR)
    qc_rules = cfg["qc_rules"]
    model_config = cfg["model_config"]
    _apply_env_model_overrides(model_config)

    reports = load_simulated_reports(DATA_FILE)

    graded_records: List[Dict[str, Any]] = []
    medical_records: List[Dict[str, Any]] = []
    physical_summaries: List[Dict[str, Any]] = []

    for report in reports:
        extraction = extract_by_report_type(report, qc_rules, model_config)
        rule_issues = run_rule_based_qc(report, extraction, qc_rules)
        reasoning = run_llm_reasoning_qc(report, extraction, rule_issues, model_config)

        graded = build_graded_record(report, extraction, rule_issues, reasoning)
        graded_records.append(graded)

        is_corrected = False
        summary = generate_physical_summary(graded, corrected=is_corrected)
        med_record = generate_standard_medical_record(graded, corrected=is_corrected)

        physical_summaries.append({"report_id": report["report_id"], **summary})
        medical_records.append({"report_id": report["report_id"], **med_record})

    graded_dataset = aggregate_grade_dataset(graded_records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / timestamp

    _save_json(graded_dataset, out_dir / "graded_dataset.json")
    _save_json(physical_summaries, out_dir / "physical_summaries.json")
    _save_json(medical_records, out_dir / "medical_records.json")

    result = {
        "output_dir": str(out_dir),
        "summary": graded_dataset["summary"],
        "total_reports": len(reports),
        "degraded_count": sum(1 for r in graded_records if r.get("extraction", {}).get("degraded")),
    }
    _save_json(result, out_dir / "run_summary.json")
    return result


if __name__ == "__main__":
    run_result = run_pipeline()
    print(json.dumps(run_result, ensure_ascii=False, indent=2))
