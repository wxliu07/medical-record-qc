# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Medical QC Framework that processes examination reports through NER/RE extraction, quality control, and document generation. It handles two report types:
- **指标类** (Indicator reports): Blood tests, lab results
- **影像类** (Imaging reports): Radiology findings (X-ray, CT, MRI)

## Running the Pipeline

```bash
# Backend pipeline (batch processing)
python main.py

# Frontend API server (interactive processing)
cd front-end && python api.py
```

Output goes to `outputs/<timestamp>/`:
- `graded_dataset.json` - Reports with quality grades
- `physical_summaries.json` - Physical examination summaries
- `medical_records.json` - Standardized medical records
- `run_summary.json` - Processing statistics

## Environment Setup

Set `DEEPSEEK_API_KEY` in `.env` or environment variable. Optional overrides:
- `DEEPSEEK_MODEL` - Override model name
- `DEEPSEEK_BASE_URL` - Override API endpoint

The frontend API server runs on `http://127.0.0.1:8000` with endpoint `POST /api/pipeline` accepting SSE streaming responses.

## Architecture

### Pipeline Flow (main.py:90, front-end/api.py:42)
```
load reports → extract entities → rule QC → LLM reasoning QC → grade → generate documents
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `modules/ner_re/` | Entity extraction by report type (`extract_by_report_type` dispatches) |
| `modules/quality_control/` | Rule-based QC + LLM reasoning QC |
| `modules/dataset/` | Grading logic (正常级/缺失级/矛盾级) |
| `modules/case_generation/` | Generates summaries and medical records |
| `modules/data_process/` | Data loading and validation |

### Frontend API Structure (front-end/api.py)

The FastAPI server exposes the same pipeline as a streaming endpoint:
- `POST /api/pipeline` - Streaming SSE response with step-by-step progress
- `GET /health` - Health check

Each report generates 6 steps reported via SSE events: `load`, `ner_re`, `rule_qc`, `llm_qc`, `grade`, `generate`.

### Grading Priority
矛盾级 > 缺失级 > 正常级

Grading is determined by `rule_issues` - if any issue has type "矛盾" (contradiction), it's 矛盾级; else if any "缺失" (missing), it's 缺失级; otherwise 正常级.

## Degradation Strategy

The system degrades gracefully when optional dependencies are unavailable:
- **llm-ie unavailable**: Falls back to rule-based extraction with `degraded=True` flag
- **radgraph unavailable**: Uses fallback keyword-matching parser (`_fallback_imaging_parse`)
- **LLM reasoning fails**: Returns `result=unknown` with failure reason

All extractions include a `degraded` boolean flag to indicate fallback mode.

## Configuration

- `config/qc_rules.json` - QC rules including indicator ranges and keywords
- `config/model_config.json` - LLM provider (litellm) and radgraph settings

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_imaging_ner_re.py

# Run tests matching a pattern
pytest -k "test_name_pattern"

# Run with verbose output
pytest -v

# Run with coverage (if installed)
pytest --cov=modules --cov-report=term-missing
```

Test files use `conftest.py` which adds the project root to `sys.path`.

## Data Format

Input reports (`data/simulate_data.json`) must have:
```json
{
  "report_id": "...",
  "report_type": "指标类" | "影像类",
  "report_subtype": "...",
  "content": {
    "描述": "...",
    "检查所见": "...",
    "检查提示": "..."
  },
  "label": "..."
}
```
