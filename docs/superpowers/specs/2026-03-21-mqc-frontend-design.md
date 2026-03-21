# MQC Frontend Visualization Design

## Overview

Build a web frontend to visualize the Medical QC Framework pipeline. Users input custom JSON report data and see step-by-step processing results.

## Architecture

```
electronic-record/
├── front-end/                    # React + Vite frontend
├── api.py                        # FastAPI backend (new)
└── main.py                       # CLI pipeline (unchanged)
```

**Interaction**: User enters JSON → Frontend → FastAPI (SSE) → Frontend displays streaming results

## Tech Stack

| Layer | Choice |
|-------|--------|
| Frontend | React 18 + Vite |
| Backend | FastAPI (Python) |
| Communication | Server-Sent Events (SSE) |
| Styling | Tailwind CSS |

## API Design

### POST /api/pipeline

**Request Body:**
```json
{
  "reports": [
    {
      "report_id": "...",
      "report_type": "影像类" | "指标类",
      "report_subtype": "...",
      "content": {
        "描述": "...",
        "检查所见": "...",
        "检查提示": "..."
      },
      "label": "..."
    }
  ]
}
```

**Response:** SSE stream of events

### SSE Event Types

| Event | Payload |
|-------|---------|
| `start` | `{"total": 5}` |
| `step_complete` | `{"step": "load\|ner_re\|rule_qc\|llm_qc\|grade\|generate", "status": "complete", "input": {...}, "output": {...}}` |
| `step_error` | `{"step": "...", "status": "error", "error": "..."}` |
| `done` | `{"summary": {...}}` |

### Step Names (Chinese)

1. `load` - 数据加载
2. `ner_re` - 实体识别 (NER/RE)
3. `rule_qc` - 规则质量控制
4. `llm_qc` - LLM 推理质量控制
5. `grade` - 分级记录
6. `generate` - 文档生成

## Pipeline Steps Detail

| Step | Description | Input | Output |
|------|-------------|-------|--------|
| 数据加载 | Validate and parse input JSON | `{"reports": [...]}` | `{"loaded": N, "types": [...]}` |
| 实体识别 | NER/RE extraction by report type | Single report | `{"entities": {}, "relations": [], "degraded": bool}` |
| 规则QC | Rule-based quality control | report + extraction | `{"rule_issues": []}` |
| LLM推理QC | LLM reasoning quality control | report + extraction + rule_issues | `{"result": "normal\|issue\|unknown", "reasoning": "..."}` |
| 分级记录 | Build graded record | All above | `{"grade": "正常级\|缺失级\|矛盾级", ...}` |
| 文档生成 | Generate summaries and records | graded record | `{"physical_summary": {...}, "medical_record": {...}}` |

## UI Design

### Color Palette (Light Theme)

| Token | Value | Usage |
|-------|-------|-------|
| `bg` | `#f8fafc` | Page background |
| `card` | `#ffffff` | Card background |
| `border` | `#e2e8f0` | Borders, dividers |
| `primary` | `#3b82f6` | Buttons, active states |
| `success` | `#22c55e` | Completed steps |
| `warning` | `#f59e0b` | In-progress steps |
| `error` | `#ef4444` | Error states |
| `text` | `#1e293b` | Primary text |
| `text-muted` | `#64748b` | Secondary text |

### Layout

```
┌─────────────────────────────────────────────────────┐
│  Header: "医学QC流水线可视化"                        │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐        │
│  │ JSON Input                              │        │
│  │ ┌─────────────────────────────────────┐  │        │
│  │ │ (Textarea with JSON)               │  │        │
│  │ └─────────────────────────────────────┘  │        │
│  │                        [运行流水线 ▶]   │        │
│  └─────────────────────────────────────────┘        │
├─────────────────────────────────────────────────────┤
│  ▼ Step 1: 数据加载                    ✓ 完成      │
│    ┌─────────────────────────────────────────────┐  │
│    │ Input:  {"reports": [...]}                  │  │
│    │ ─────────────────────────────────────────── │  │
│    │ Output: {"loaded": 2, "types": ["影像类"]} │  │
│    └─────────────────────────────────────────────┘  │
│    ──────────────────────────────────────────────  │
│  ▼ Step 2: 实体识别 (NER/RE)           ✓ 完成      │
│    ┌─────────────────────────────────────────────┐  │
│    │ Output: {"entities": {...}, "relations": []}│  │
│    └─────────────────────────────────────────────┘  │
│    ...                                             │
└─────────────────────────────────────────────────────┘
```

### Step Card States

| State | Visual |
|-------|--------|
| `pending` | Collapsed, muted text, no icon |
| `in_progress` | Expanded, warning color, spinner |
| `completed` | Expanded, success color, checkmark |
| `error` | Expanded, error color, X icon |

### Components

| Component | Purpose |
|-----------|---------|
| `JsonInput` | Textarea for JSON input + Run button |
| `PipelineFlow` | Container for all step cards |
| `StepCard` | Single step with expand/collapse, input/output display |
| `ResultPanel` | Final summary after pipeline completes |

## Frontend File Structure

```
front-end/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── index.css
│   ├── components/
│   │   ├── JsonInput.jsx
│   │   ├── PipelineFlow.jsx
│   │   ├── StepCard.jsx
│   │   └── ResultPanel.jsx
│   └── services/
│       └── api.js
```

## Backend File Structure

```
api.py          # FastAPI app with /api/pipeline endpoint
```

## Dependencies

### Backend (api.py)
- fastapi
- uvicorn
- python-multipart (for SSE)

### Frontend (front-end/)
- react
- react-dom
- vite
- tailwindcss
- autoprefixer
- postcss

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid JSON input | Show validation error in input area |
| Pipeline error | Stream `step_error` event, show error in step card |
| Network error | Show reconnect button |
| LLM unavailable | Extraction continues with `degraded=true` flag |

## Implementation Order

1. Create `api.py` with SSE endpoint
2. Create `front-end/` scaffold with Vite
3. Implement `JsonInput` component
4. Implement `StepCard` component
5. Implement `PipelineFlow` component
6. Implement `ResultPanel` component
7. Wire up SSE streaming
8. Style with Tailwind light theme
