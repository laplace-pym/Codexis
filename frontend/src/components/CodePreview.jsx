import React, { useState, useEffect } from 'react'
import { X, Play, SkipForward, Edit3, Check, Copy, CheckCheck } from 'lucide-react'

/**
 * Code preview modal for interactive mode.
 *
 * Shows generated code and allows user to:
 * - Execute in sandbox (verify)
 * - Skip verification (just save)
 * - Modify the code before execution
 */
export function CodePreview({
  code,
  language,
  filePath,
  requestId,
  onExecute,
  onSkip,
  onModify,
  onClose
}) {
  const [editMode, setEditMode] = useState(false)
  const [editedCode, setEditedCode] = useState(code)
  const [copied, setCopied] = useState(false)

  // Update edited code when code prop changes
  useEffect(() => {
    setEditedCode(code)
  }, [code])

  // Handle copy to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(editMode ? editedCode : code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  // Handle execute action
  const handleExecute = () => {
    if (editMode && editedCode !== code) {
      onModify(editedCode)
    } else {
      onExecute()
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="px-4 py-3 border-b border-slate-200 flex items-center justify-between bg-slate-50 rounded-t-xl">
          <div className="flex items-center gap-3">
            <span className="font-mono text-sm text-slate-600 bg-slate-200 px-2 py-1 rounded">
              {filePath}
            </span>
            <span className="text-xs text-slate-500 bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
              {language}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="p-1.5 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors"
              title="Copy code"
            >
              {copied ? <CheckCheck size={16} className="text-green-500" /> : <Copy size={16} />}
            </button>
            <button
              onClick={() => setEditMode(!editMode)}
              className={`
                p-1.5 rounded transition-colors flex items-center gap-1 text-sm
                ${editMode
                  ? 'bg-blue-100 text-blue-600'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'}
              `}
              title={editMode ? "Preview mode" : "Edit mode"}
            >
              {editMode ? <Check size={16} /> : <Edit3 size={16} />}
              <span className="hidden sm:inline">{editMode ? "Preview" : "Edit"}</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors"
              title="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Code area - dark background with light text */}
        <div
          className="flex-1 overflow-auto"
          style={{ backgroundColor: '#1e293b', minHeight: '400px' }}
        >
          {editMode ? (
            <textarea
              value={editedCode}
              onChange={(e) => setEditedCode(e.target.value)}
              className="w-full h-full p-4 font-mono text-sm resize-none focus:outline-none"
              style={{
                backgroundColor: '#1e293b',
                color: '#e2e8f0',
                minHeight: '400px',
                border: 'none'
              }}
              spellCheck={false}
            />
          ) : (
            <pre
              className="p-4 font-mono text-sm overflow-x-auto whitespace-pre-wrap"
              style={{
                backgroundColor: '#1e293b',
                color: '#e2e8f0',
                margin: 0
              }}
            >
              <code style={{ color: '#e2e8f0' }}>{code}</code>
            </pre>
          )}
        </div>

        {/* Footer with actions */}
        <div className="px-4 py-3 border-t border-slate-200 flex items-center justify-between bg-slate-50 rounded-b-xl">
          <p className="text-sm text-slate-500">
            Review the generated code and choose an action
          </p>

          <div className="flex items-center gap-2">
            <button
              onClick={onSkip}
              className="
                px-4 py-2 text-sm font-medium text-slate-600
                bg-white border border-slate-300 rounded-lg
                hover:bg-slate-50 transition-colors
                flex items-center gap-2
              "
            >
              <SkipForward size={16} />
              Skip Verification
            </button>

            <button
              onClick={handleExecute}
              className="
                px-4 py-2 text-sm font-medium text-white
                bg-blue-600 rounded-lg
                hover:bg-blue-700 transition-colors
                flex items-center gap-2
              "
            >
              <Play size={16} />
              {editMode && editedCode !== code ? "Save & Execute" : "Execute in Sandbox"}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CodePreview
