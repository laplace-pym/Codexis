import React from 'react'
import { MessageSquare, Wrench } from 'lucide-react'

/**
 * Mode selector for switching between Chat and Agent modes.
 */
export function ModeSelector({ mode, onModeChange, disabled }) {
  return (
    <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-1">
      <button
        onClick={() => onModeChange('chat')}
        disabled={disabled}
        className={`
          flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all
          ${mode === 'chat'
            ? 'bg-white text-slate-900 shadow-sm'
            : 'text-slate-600 hover:text-slate-900'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        <MessageSquare size={16} />
        Chat
      </button>
      <button
        onClick={() => onModeChange('agent')}
        disabled={disabled}
        className={`
          flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all
          ${mode === 'agent'
            ? 'bg-white text-slate-900 shadow-sm'
            : 'text-slate-600 hover:text-slate-900'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        <Wrench size={16} />
        Agent
      </button>
    </div>
  )
}

export default ModeSelector
