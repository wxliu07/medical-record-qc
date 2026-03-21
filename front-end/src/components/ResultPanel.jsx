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
