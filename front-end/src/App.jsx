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
      let currentEvent = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Process complete SSE messages (event: X\ndata: Y\n\n)
        while (buffer.includes('\n\n')) {
          const msgEnd = buffer.indexOf('\n\n')
          const message = buffer.slice(0, msgEnd)
          buffer = buffer.slice(msgEnd + 2)

          const eventMatch = message.match(/^event: ([^\n]+)\n/)
          const dataMatch = message.match(/^data: (.+)$/s)  // /s makes . match newlines

          if (eventMatch) {
            currentEvent = eventMatch[1].trim()
          }
          if (dataMatch && currentEvent) {
            const data = JSON.parse(dataMatch[1])
            if (currentEvent === 'step_complete') {
              setSteps(prev => [...prev, data])
            } else if (currentEvent === 'step_error') {
              setSteps(prev => [...prev, data])
            } else if (currentEvent === 'start') {
              // start event - data has {total: N}
            } else if (currentEvent === 'done') {
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
