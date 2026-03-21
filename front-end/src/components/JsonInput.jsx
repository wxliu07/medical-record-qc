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
