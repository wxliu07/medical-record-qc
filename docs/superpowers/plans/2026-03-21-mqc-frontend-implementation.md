# MQC Frontend Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a React + FastAPI web app that visualizes the Medical QC pipeline step-by-step via SSE streaming. User inputs JSON reports, frontend shows real-time streaming results for each pipeline step.

**Architecture:**
- `api.py` (FastAPI): receives JSON reports, runs pipeline steps, streams results via SSE
- `front-end/` (React + Vite + Tailwind): receives SSE stream, displays waterfall-style step cards
- `main.py`: remains unchanged as CLI-only entry point

**Tech Stack:**
- Backend: FastAPI, uvicorn, python-multipart, sse-starlette
- Frontend: React 18, Vite, Tailwind CSS, Lucide React (icons)

---

## File Map

### New Files
| File | Purpose |
|------|---------|
| `api.py` | FastAPI app with SSE `/api/pipeline` endpoint |
| `front-end/package.json` | Frontend dependencies |
| `front-end/vite.config.js` | Vite bundler config |
| `front-end/tailwind.config.js` | Tailwind theme (light) |
| `front-end/postcss.config.js` | PostCSS for Tailwind |
| `front-end/index.html` | Entry HTML |
| `front-end/src/main.jsx` | React entry point |
| `front-end/src/index.css` | Tailwind directives + base styles |
| `front-end/src/App.jsx` | Root component |
| `front-end/src/services/api.js` | SSE client service |
| `front-end/src/components/JsonInput.jsx` | JSON textarea + Run button |
| `front-end/src/components/PipelineFlow.jsx` | Step card container |
| `front-end/src/components/StepCard.jsx` | Individual step card |
| `front-end/src/components/ResultPanel.jsx` | Final summary panel |

### Modified Files
| File | Change |
|------|--------|
| `main.py` | No changes (CLI only) |

---

## Tasks

### Task 1: Create FastAPI Backend (`api.py`)

**Files:**
- Create: `api.py`

- [ ] **Step 1: Write `api.py`**

```python
# api.py
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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


async def _stream_event(event_type: str, data: Dict[str, Any]):
    yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run_pipeline_stream(reports: List[Dict[str, Any]]):
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
    yield await _stream_event("start", {"total": total})

    step_num = 0
    graded_records = []
    all_summaries = []
    all_records = []

    for report in reports:
        # Step: Load (already validated)
        step_num += 1
        yield await _stream_event("step", {
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
            yield await _stream_event("step", {
                "step": "ner_re",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report},
                "output": extraction,
            })
        except Exception as e:
            yield await _stream_event("step", {
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
            yield await _stream_event("step", {
                "step": "rule_qc",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report, "extraction": extraction},
                "output": {"rule_issues": rule_issues},
            })
        except Exception as e:
            yield await _stream_event("step", {
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
            yield await _stream_event("step", {
                "step": "llm_qc",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report, "extraction": extraction, "rule_issues": rule_issues},
                "output": reasoning,
            })
        except Exception as e:
            yield await _stream_event("step", {
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
            yield await _stream_event("step", {
                "step": "grade",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"report": report, "extraction": extraction, "rule_issues": rule_issues, "reasoning": reasoning},
                "output": graded,
            })
        except Exception as e:
            yield await _stream_event("step", {
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
            yield await _stream_event("step", {
                "step": "generate",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "complete",
                "input": {"graded": graded},
                "output": {"physical_summary": summary, "medical_record": med_record},
            })
        except Exception as e:
            yield await _stream_event("step", {
                "step": "generate",
                "step_num": step_num,
                "report_id": report.get("report_id"),
                "status": "error",
                "error": str(e),
            })

    # Done
    graded_dataset = aggregate_grade_dataset(graded_records)
    yield await _stream_event("done", {
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
        return {"error": "reports must be a list"}
    if len(reports) == 0:
        return {"error": "reports cannot be empty"}

    # Normalize
    try:
        reports = validate_and_normalize_reports(reports)
    except ValueError as e:
        return {"error": str(e)}

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
```

- [ ] **Step 2: Test API imports**

Run: `cd e:/Codes/electronic-record && python -c "import api; print('OK')"`
Expected: `OK` (may show deprecation warnings)

- [ ] **Step 3: Commit**

```bash
git add api.py
git commit -m "feat: add FastAPI backend with SSE pipeline endpoint"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

### Task 2: Scaffold Frontend Project

**Files:**
- Create: `front-end/package.json`, `front-end/vite.config.js`, `front-end/tailwind.config.js`, `front-end/postcss.config.js`, `front-end/index.html`

- [ ] **Step 1: Create `front-end/package.json`**

```json
{
  "name": "mqc-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "lucide-react": "^0.468.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.17",
    "vite": "^6.0.5"
  }
}
```

- [ ] **Step 2: Create `front-end/vite.config.js`**

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 3: Create `front-end/tailwind.config.js`**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#f8fafc',
        card: '#ffffff',
        border: '#e2e8f0',
        primary: '#3b82f6',
        success: '#22c55e',
        warning: '#f59e0b',
        error: '#ef4444',
        text: '#1e293b',
        'text-muted': '#64748b',
      },
    },
  },
  plugins: [],
}
```

- [ ] **Step 4: Create `front-end/postcss.config.js`**

```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

- [ ] **Step 5: Create `front-end/index.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>医学QC流水线可视化</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 6: Install dependencies**

Run: `cd e:/Codes/electronic-record/front-end && npm install`
Expected: packages installed

- [ ] **Step 7: Commit**

```bash
git add front-end/
git commit -m "feat: scaffold React + Vite + Tailwind frontend"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

### Task 3: Frontend Entry Files

**Files:**
- Create: `front-end/src/main.jsx`, `front-end/src/index.css`, `front-end/src/App.jsx`

- [ ] **Step 1: Create `front-end/src/main.jsx`**

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

- [ ] **Step 2: Create `front-end/src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  background-color: #f8fafc;
  color: #1e293b;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
```

- [ ] **Step 3: Create `front-end/src/App.jsx`**

```jsx
import { useState, useCallback } from 'react'
import JsonInput from './components/JsonInput.jsx'
import PipelineFlow from './components/PipelineFlow.jsx'
import ResultPanel from './components/ResultPanel.jsx'

const DEFAULT_JSON = JSON.stringify([
  {
    report_id: "TEST001",
    report_type: "影像类",
    report_subtype: "CT",
    content: {
      "描述": "头部CT平扫",
      "检查所见": "脑实质未见明显异常密度影，脑室系统未见扩大，中线结构居中。",
      "检查提示": "颅内CT平扫未见明显异常。"
    },
    label: "正常"
  }
], null, 2)

export default function App() {
  const [steps, setSteps] = useState([])
  const [summary, setSummary] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState(null)

  const handleRun = useCallback(async (jsonText) => {
    setSteps([])
    setSummary(null)
    setError(null)
    setIsRunning(true)

    try {
      const reports = JSON.parse(jsonText)
      const response = await fetch('/api/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reports }),
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.error || 'API error')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            const eventType = line.slice(7).trim()
            continue
          }
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6))
            if (data.step) {
              setSteps(prev => [...prev, data])
            } else if (data.total) {
              // start event
            } else if (data.summary) {
              setSummary(data)
            }
          }
        }
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setIsRunning(false)
    }
  }, [])

  return (
    <div className="min-h-screen bg-bg">
      <header className="bg-white border-b border-border px-6 py-4">
        <h1 className="text-xl font-semibold text-text">医学QC流水线可视化</h1>
      </header>
      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        <JsonInput defaultValue={DEFAULT_JSON} onRun={handleRun} isRunning={isRunning} error={error} />
        {steps.length > 0 && <PipelineFlow steps={steps} />}
        {summary && <ResultPanel summary={summary} />}
      </main>
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add front-end/src/main.jsx front-end/src/index.css front-end/src/App.jsx
git commit -m "feat: add frontend entry files and App component"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

### Task 4: JsonInput Component

**Files:**
- Create: `front-end/src/components/JsonInput.jsx`

- [ ] **Step 1: Create `front-end/src/components/JsonInput.jsx`**

```jsx
import { useState } from 'react'
import { Play, AlertCircle } from 'lucide-react'

export default function JsonInput({ defaultValue, onRun, isRunning, error }) {
  const [value, setValue] = useState(defaultValue)
  const [jsonError, setJsonError] = useState(null)

  const handleValidate = () => {
    try {
      JSON.parse(value)
      setJsonError(null)
      return true
    } catch (e) {
      setJsonError(e.message)
      return false
    }
  }

  const handleRun = () => {
    if (!handleValidate()) return
    onRun(value)
  }

  return (
    <div className="bg-card rounded-lg border border-border shadow-sm">
      <div className="px-4 py-3 border-b border-border">
        <h2 className="font-medium text-text">输入报告数据 (JSON)</h2>
      </div>
      <div className="p-4">
        <textarea
          className="w-full h-64 p-3 font-mono text-sm border border-border rounded-md bg-bg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-y"
          value={value}
          onChange={e => setValue(e.target.value)}
          onBlur={handleValidate}
          placeholder="输入JSON格式的报告数据..."
        />
        {(jsonError || error) && (
          <div className="mt-2 flex items-center gap-2 text-error text-sm">
            <AlertCircle size={16} />
            <span>{jsonError || error}</span>
          </div>
        )}
        <div className="mt-4 flex justify-end">
          <button
            onClick={handleRun}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Play size={16} />
            {isRunning ? '运行中...' : '运行流水线'}
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add front-end/src/components/JsonInput.jsx
git commit -m "feat: add JsonInput component"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

### Task 5: StepCard Component

**Files:**
- Create: `front-end/src/components/StepCard.jsx`

- [ ] **Step 1: Create `front-end/src/components/StepCard.jsx`**

```jsx
import { useState } from 'react'
import { ChevronDown, ChevronRight, CheckCircle, Circle, AlertCircle, Loader } from 'lucide-react'

const STEP_LABELS = {
  load: '数据加载',
  ner_re: '实体识别 (NER/RE)',
  rule_qc: '规则质量控制',
  llm_qc: 'LLM 推理质量控制',
  grade: '分级记录',
  generate: '文档生成',
}

const STATUS_ICONS = {
  complete: <CheckCircle size={18} className="text-success" />,
  in_progress: <Loader size={18} className="text-warning animate-spin" />,
  error: <AlertCircle size={18} className="text-error" />,
  pending: <Circle size={18} className="text-text-muted" />,
}

export default function StepCard({ step, isFirst = false }) {
  const [expanded, setExpanded] = useState(true)

  const label = STEP_LABELS[step.step] || step.step
  const status = step.status
  const Icon = STATUS_ICONS[status] || STATUS_ICONS.pending

  return (
    <div className="bg-card rounded-lg border border-border shadow-sm overflow-hidden">
      <button
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-bg/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          {Icon}
          <span className="font-medium text-text">
            {label}
          </span>
          <span className="text-sm text-text-muted">#{step.step_num}</span>
          {step.report_id && (
            <span className="text-xs text-text-muted bg-bg px-2 py-0.5 rounded">
              {step.report_id}
            </span>
          )}
        </div>
        {expanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          {status === 'error' ? (
            <div className="p-3 bg-error/10 border border-error/20 rounded-md">
              <p className="text-error text-sm">{step.error}</p>
            </div>
          ) : (
            <>
              {step.input && (
                <div>
                  <h4 className="text-xs font-medium text-text-muted uppercase mb-1">输入</h4>
                  <pre className="p-3 bg-bg rounded-md text-xs font-mono overflow-x-auto">
                    {JSON.stringify(step.input, null, 2)}
                  </pre>
                </div>
              )}
              {step.output && (
                <div>
                  <h4 className="text-xs font-medium text-text-muted uppercase mb-1">输出</h4>
                  <pre className="p-3 bg-bg rounded-md text-xs font-mono overflow-x-auto">
                    {JSON.stringify(step.output, null, 2)}
                  </pre>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add front-end/src/components/StepCard.jsx
git commit -m "feat: add StepCard component"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

### Task 6: PipelineFlow Component

**Files:**
- Create: `front-end/src/components/PipelineFlow.jsx`

- [ ] **Step 1: Create `front-end/src/components/PipelineFlow.jsx`**

```jsx
import StepCard from './StepCard.jsx'

export default function PipelineFlow({ steps }) {
  if (steps.length === 0) return null

  // Group steps by report_id
  const byReport = {}
  for (const step of steps) {
    const rid = step.report_id || 'unknown'
    if (!byReport[rid]) byReport[rid] = []
    byReport[rid].push(step)
  }

  return (
    <div className="space-y-6">
      {Object.entries(byReport).map(([reportId, reportSteps]) => (
        <div key={reportId}>
          <h3 className="text-sm font-semibold text-text-muted mb-3">
            报告: {reportId}
          </h3>
          <div className="space-y-3">
            {reportSteps.map((step, idx) => (
              <StepCard key={`${reportId}-${step.step}-${idx}`} step={step} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add front-end/src/components/PipelineFlow.jsx
git commit -m "feat: add PipelineFlow component"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

### Task 7: ResultPanel Component

**Files:**
- Create: `front-end/src/components/ResultPanel.jsx`

- [ ] **Step 1: Create `front-end/src/components/ResultPanel.jsx`**

```jsx
import { FileText, Shield } from 'lucide-react'

export default function ResultPanel({ summary }) {
  const gradeColors = {
    '正常级': 'text-success bg-success/10',
    '缺失级': 'text-warning bg-warning/10',
    '矛盾级': 'text-error bg-error/10',
  }

  return (
    <div className="bg-card rounded-lg border border-border shadow-sm">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <FileText size={18} className="text-primary" />
        <h2 className="font-medium text-text">处理完成</h2>
      </div>
      <div className="p-4">
        <div className="grid grid-cols-3 gap-4 mb-4">
          {Object.entries(summary).map(([grade, count]) => (
            <div
              key={grade}
              className={`p-4 rounded-lg border ${gradeColors[grade] || 'text-text bg-bg'}`}
            >
              <div className="text-2xl font-bold">{count}</div>
              <div className="text-sm flex items-center gap-1">
                <Shield size={14} />
                {grade}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add front-end/src/components/ResultPanel.jsx
git commit -m "feat: add ResultPanel component"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

### Task 8: Verify End-to-End

- [ ] **Step 1: Start backend**

Run: `cd e:/Codes/electronic-record && python api.py &`
Wait: Server starts on port 8000

- [ ] **Step 2: Start frontend**

Run: `cd e:/Codes/electronic-record/front-end && npm run dev &`
Wait: Vite dev server starts on port 5173

- [ ] **Step 3: Open browser**

Open: `http://localhost:5173`
Verify: Page loads with JSON input textarea and Run button

- [ ] **Step 4: Click Run Pipeline**

Verify: Steps appear one by one as SSE events arrive
Verify: Final ResultPanel shows grade summary

- [ ] **Step 5: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: complete MQC frontend visualization with SSE streaming"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Verification Commands

| Check | Command | Expected |
|-------|---------|----------|
| Backend health | `curl http://localhost:8000/health` | `{"status":"ok"}` |
| API smoke | `curl -X POST http://localhost:8000/api/pipeline -H "Content-Type: application/json" -d '{"reports":[]}'` | `{"error":"reports cannot be empty"}` |
| Frontend build | `cd front-end && npm run build` | No errors |
