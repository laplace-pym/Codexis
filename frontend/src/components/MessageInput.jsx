import React, { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'

/**
 * Message input component with auto-resize textarea.
 */
export function MessageInput({ onSend, isLoading, disabled, placeholder }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [value])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (value.trim() && !isLoading && !disabled) {
      onSend(value)
      setValue('')
    }
  }

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="border-t border-slate-200 p-4 bg-white">
      <div className="flex items-end gap-2 max-w-4xl mx-auto">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder || "Type your message... (Shift+Enter for new line)"}
            disabled={isLoading || disabled}
            rows={1}
            className="
              w-full resize-none border border-slate-200 rounded-xl px-4 py-3
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
              disabled:bg-slate-50 disabled:cursor-not-allowed
              text-slate-800 placeholder-slate-400
            "
          />
        </div>

        <button
          type="submit"
          disabled={!value.trim() || isLoading || disabled}
          className="
            flex-shrink-0 w-12 h-12 rounded-xl
            bg-blue-500 text-white
            flex items-center justify-center
            hover:bg-blue-600 transition-colors
            disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed
          "
        >
          {isLoading ? (
            <Loader2 size={20} className="animate-spin" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </div>
    </form>
  )
}

export default MessageInput
