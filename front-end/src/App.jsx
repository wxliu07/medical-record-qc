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

        // Decode this chunk (final chunk when done=true)
        if (value) {
          buffer += decoder.decode(value, { stream: !done })
        }

        if (done) break
      }

      // Process all complete SSE messages in buffer
      // Each message format: "event: TYPE\ndata: JSON\n\n"
      const messages = buffer.split(/\n\n/)
      for (const rawMsg of messages) {
        if (!rawMsg.trim()) continue

        const lines = rawMsg.split('\n')
        let eventType = currentEvent
        let jsonData = ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            jsonData = line.slice(6)
          }
        }

        if (!eventType || !jsonData) continue

        try {
          const data = JSON.parse(jsonData)
          if (eventType === 'step_complete' || eventType === 'step_error') {
            setSteps(prev => [...prev, data])
          } else if (eventType === 'done') {
            setSummary(data)
          }
        } catch (e) {
          console.error('JSON parse error:', e, jsonData)
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
