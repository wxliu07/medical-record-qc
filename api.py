# api.py
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

sys.path.insert(0, str(BASE_DIR))

from modules.case_generation import generate_physical_summary, generate_standard_medical_record
from modules.data_process.data_loader import load_config, validate_and_normalize_reports
from modules.dataset import build_graded_record, aggregate_grade_dataset
from modules.ner_re import extract_by_report_type
from modules.quality_control import run_llm_reasoning_qc, run_rule_based_qc


def _load_env_file():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    import os
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _stream_event(event_type: str, data: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def run_pipeline_stream(reports: List[Dict[str, Any]]):
    # Load config
    _load_env_file()
    cfg = load_config(CONFIG_DIR)
    qc_rules = cfg["qc_rules"]
    model_config = cfg["model_config"]

    # Apply env overrides
    llm_cfg = model_config.setdefault("llm", {})
    import os
    if os.getenv("DEEPSEEK_MODEL", "").strip():
        llm_cfg["model"] = os.getenv("DEEPSEEK_MODEL", "").strip()
    if os.getenv("DEEPSEEK_BASE_URL", "").strip():
        llm_cfg["base_url"] = os.getenv("DEEPSEEK_BASE_URL", "").strip()

    total = 6 * len(reports) + 1  # steps per report + start
    yield _stream_event("start", {"total": total})

    step_num = 0
    graded_records = []
    all_summaries = []
    all_records = []

    for report in reports:
        # Step: Load (already validated)
        step_num += 1
        yield _stream_event("step_complete", {
            "step": "load",
            "step_num": step_num,
            "report_id": report.get("report_id"),
            "status": "complete",
            "input": {"reports": [{"report_id": report.get("report_id")}]},
            "output": {"loaded": 1, "report_type": report.get("report_type")},
        })

        # Step: NER/RE
        step_num += 1
        try:
            extraction = extract_by_report_type(report, qc_rules, model_config)
            yield _stream_event("step_complete", {
                "step": "ner_re",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report},
                "output": extraction,
            })
        except Exception as e:
            yield _stream_event("step_error", {
                "step": "ner_re",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "error",
                "error": str(e),
            })
            continue

        # Step: Rule QC
        step_num += 1
        try:
            rule_issues = run_rule_based_qc(report, extraction, qc_rules)
            yield _stream_event("step_complete", {
                "step": "rule_qc",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report, "extraction": extraction},
                "output": {"rule_issues": rule_issues},
            })
        except Exception as e:
            yield _stream_event("step_error", {
                "step": "rule_qc",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "error",
                "error": str(e),
            })
            continue

        # Step: LLM Reasoning QC
        step_num += 1
        try:
            reasoning = run_llm_reasoning_qc(report, extraction, rule_issues, model_config)
            yield _stream_event("step_complete", {
                "step": "llm_qc",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report, "extraction": extraction, "rule_issues": rule_issues},
                "output": reasoning,
            })
        except Exception as e:
            yield _stream_event("step_error", {
                "step": "llm_qc",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "error",
                "error": str(e),
            })
            continue

        # Step: Grade
        step_num += 1
        try:
            graded = build_graded_record(report, extraction, rule_issues, reasoning)
            graded_records.append(graded)
            yield _stream_event("step_complete", {
                "step": "grade",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report, "extraction": extraction, "rule_issues": rule_issues, "reasoning": reasoning},
                "output": graded,
            })
        except Exception as e:
            yield _stream_event("step_error", {
                "step": "grade",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "error",
                "error": str(e),
            })
            continue

        # Step: Generate
        step_num += 1
        try:
            summary = generate_physical_summary(graded, corrected=False)
            med_record = generate_standard_medical_record(graded, corrected=False)
            all_summaries.append({"report_id": report["report_id"], **summary})
            all_records.append({"report_id": report["report_id"], **med_record})
            yield _stream_event("step_complete", {
                "step": "generate",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"graded": graded},
                "output": {"physical_summary": summary, "medical_record": med_record},
            })
        except Exception as e:
            yield _stream_event("step_error", {
                "step": "generate",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "error",
                "error": str(e),
            })

    # Done
    graded_dataset = aggregate_grade_dataset(graded_records)
    yield _stream_event("done", {
        "summary": graded_dataset["summary"],
        "total_reports": len(reports),
        "degraded_count": sum(1 for r in graded_records if r.get("extraction", {}).get("degraded")),
    })


app = FastAPI(title="MQC Pipeline API")


@app.post("/api/pipeline")
async def pipeline(request: Request):
    body = await request.json()
    reports = body.get("reports", [])

    # Validate input
    if not isinstance(reports, list):
        return JSONResponse(status_code=400, content={"error": "reports must be a list"})
    if len(reports) == 0:
        return JSONResponse(status_code=400, content={"error": "reports cannot be empty"})

    # Normalize
    try:
        reports = validate_and_normalize_reports(reports)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return StreamingResponse(
        run_pipeline_stream(reports),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
