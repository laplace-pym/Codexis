import React from 'react'
import { Wrench, CheckCircle, XCircle, ChevronRight } from 'lucide-react'

/**
 * Panel showing tool calls in Agent mode.
 */
export function ToolPanel({ toolCalls, isVisible }) {
  if (!isVisible || toolCalls.length === 0) {
    return null
  }

  return (
    <div className="border-l border-slate-200 w-80 bg-slate-50 overflow-y-auto">
      <div className="p-4 border-b border-slate-200 bg-white sticky top-0">
        <h3 className="font-medium text-slate-900 flex items-center gap-2">
          <Wrench size={16} />
          Tool Calls
        </h3>
      </div>

      <div className="p-4 space-y-3">
        {toolCalls.map((call, index) => (
          <ToolCallItem key={index} call={call} />
        ))}
      </div>
    </div>
  )
}

function ToolCallItem({ call }) {
  const isCall = call.type === 'call'
  const isResult = call.type === 'result'

  return (
    <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
      {isCall && (
        <>
          <div className="px-3 py-2 bg-slate-100 border-b border-slate-200 flex items-center gap-2">
            <ChevronRight size={14} className="text-blue-500" />
            <span className="font-mono text-sm text-slate-700">{call.tool}</span>
          </div>
          {call.args && (
            <pre className="px-3 py-2 text-xs text-slate-600 overflow-x-auto">
              {JSON.stringify(call.args, null, 2)}
            </pre>
          )}
        </>
      )}

      {isResult && (
        <div className="px-3 py-2">
          <div className="flex items-center gap-2 mb-1">
            {call.success ? (
              <CheckCircle size={14} className="text-green-500" />
            ) : (
              <XCircle size={14} className="text-red-500" />
            )}
            <span className="text-xs text-slate-500">
              {call.success ? 'Success' : 'Failed'}
            </span>
          </div>
          {call.content && (
            <p className="text-xs text-slate-600 line-clamp-3">
              {call.content}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default ToolPanel
