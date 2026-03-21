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
