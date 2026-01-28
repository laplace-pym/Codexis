import React, { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { User, Bot, AlertCircle, Loader2 } from 'lucide-react'

/**
 * Single message component.
 */
function Message({ message }) {
  const isUser = message.role === 'user'
  const isStreaming = message.isStreaming
  const hasError = message.error

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
          <Bot size={18} className="text-blue-600" />
        </div>
      )}

      <div className={`flex flex-col max-w-[80%] ${isUser ? 'items-end' : ''}`}>
        {/* Iteration indicator */}
        {message.iteration && isStreaming && (
          <div className="text-xs text-blue-500 mb-1 flex items-center gap-1 bg-blue-50 px-2 py-0.5 rounded">
            <Loader2 size={10} className="animate-spin" />
            Iteration {message.iteration.iteration}/{message.iteration.max_iterations}
          </div>
        )}

        {/* Thinking indicator */}
        {message.thinking && (
          <div className="text-sm text-slate-500 mb-1 flex items-center gap-1">
            <Loader2 size={12} className="animate-spin" />
            {message.thinking}
          </div>
        )}

        {/* Status indicator */}
        {message.status && (
          <div className="text-xs text-green-600 mb-1">
            ✓ {message.status}
          </div>
        )}

        {/* Message content */}
        <div
          className={`
            rounded-2xl px-4 py-2.5
            ${isUser
              ? 'bg-blue-500 text-white'
              : hasError
                ? 'bg-red-50 border border-red-200 text-red-800'
                : 'bg-slate-100 text-slate-800'
            }
          `}
        >
          {hasError ? (
            <div className="flex items-center gap-2">
              <AlertCircle size={16} className="text-red-500" />
              <span>{message.error}</span>
            </div>
          ) : isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown>{message.content || ''}</ReactMarkdown>
            </div>
          )}

          {/* Streaming cursor */}
          {isStreaming && !message.content && !message.thinking && (
            <span className="typing-indicator">...</span>
          )}
        </div>

        {/* Timestamp */}
        <span className="text-xs text-slate-400 mt-1 px-1">
          {message.timestamp?.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </span>
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center">
          <User size={18} className="text-slate-600" />
        </div>
      )}
    </div>
  )
}

/**
 * Message list component displaying conversation history.
 */
export function MessageList({ messages }) {
  const bottomRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400">
        <div className="text-center">
          <Bot size={48} className="mx-auto mb-4 opacity-50" />
          <p className="text-lg">Start a conversation</p>
          <p className="text-sm mt-1">Ask me anything or give me a coding task</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((message, index) => (
        <Message key={index} message={message} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

export default MessageList
