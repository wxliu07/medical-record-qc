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
