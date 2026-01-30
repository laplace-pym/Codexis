import { useState, useCallback, useRef } from 'react'
import api from '../api'
import { useWebSocket } from './useWebSocket'

/**
 * Chat hook for managing conversation state and API calls.
 */
export function useChat() {
  const [messages, setMessages] = useState([])
  const [mode, setMode] = useState('agent') // 'chat' or 'agent'
  const [sessionId, setSessionId] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [documents, setDocuments] = useState([])
  const [toolCalls, setToolCalls] = useState([])
  // Interactive mode state
  const [executionMode, setExecutionMode] = useState('auto') // 'auto' or 'interactive'
  const [codePreview, setCodePreview] = useState(null)

  const currentResponseRef = useRef('')
  const ws = useWebSocket(sessionId || 'new')

  // Register WebSocket event handlers
  ws.onEvent('session', (data) => {
    setSessionId(data.session_id)
  })

  ws.onEvent('iteration', (data) => {
    setMessages(prev => {
      const lastMsg = prev[prev.length - 1]
      if (lastMsg?.role === 'assistant' && lastMsg.isStreaming) {
        return prev.map((msg, i) =>
          i === prev.length - 1
            ? { ...msg, iteration: data.metadata }
            : msg
        )
      }
      return prev
    })
  })

  ws.onEvent('thinking', (data) => {
    setMessages(prev => {
      const lastMsg = prev[prev.length - 1]
      if (lastMsg?.role === 'assistant' && lastMsg.isStreaming) {
        return prev.map((msg, i) =>
          i === prev.length - 1
            ? { ...msg, thinking: data.content }
            : msg
        )
      }
      return prev
    })
  })

  ws.onEvent('status', (data) => {
    setMessages(prev => {
      const lastMsg = prev[prev.length - 1]
      if (lastMsg?.role === 'assistant' && lastMsg.isStreaming) {
        return prev.map((msg, i) =>
          i === prev.length - 1
            ? { ...msg, status: data.content }
            : msg
        )
      }
      return prev
    })
  })

  ws.onEvent('tool_call', (data) => {
    setToolCalls(prev => [...prev, {
      type: 'call',
      tool: data.metadata?.tool,
      args: data.metadata?.args,
      content: data.content,
    }])
  })

  ws.onEvent('tool_result', (data) => {
    setToolCalls(prev => [...prev, {
      type: 'result',
      content: data.content,
      success: data.metadata?.success,
    }])
  })

  // Handle code_preview event for interactive mode
  ws.onEvent('code_preview', (data) => {
    if (executionMode === 'interactive') {
      setCodePreview({
        code: data.content,
        filePath: data.metadata?.file_path,
        language: data.metadata?.language || 'python',
        requestId: data.metadata?.request_id,
      })
    } else {
      // Auto mode - send execute action immediately
      ws.sendJson({
        type: 'user_action',
        action: 'execute',
        request_id: data.metadata?.request_id,
      })
    }
  })

  ws.onEvent('content', (data) => {
    currentResponseRef.current += data.content
    setMessages(prev => {
      const lastMsg = prev[prev.length - 1]
      if (lastMsg?.role === 'assistant' && lastMsg.isStreaming) {
        return prev.map((msg, i) =>
          i === prev.length - 1
            ? { ...msg, content: currentResponseRef.current }
            : msg
        )
      }
      return prev
    })
  })

  ws.onEvent('complete', (data) => {
    setMessages(prev => prev.map((msg, i) =>
      i === prev.length - 1
        ? { ...msg, isStreaming: false, content: data.content || currentResponseRef.current }
        : msg
    ))
    setIsLoading(false)
    currentResponseRef.current = ''
  })

  ws.onEvent('error', (data) => {
    setError(data.content)
    setIsLoading(false)
    setMessages(prev => prev.map((msg, i) =>
      i === prev.length - 1
        ? { ...msg, isStreaming: false, error: data.content }
        : msg
    ))
  })

  // Send message via WebSocket (streaming)
  const sendMessageStream = useCallback((content) => {
    if (!content.trim()) return

    setError(null)
    setToolCalls([])
    currentResponseRef.current = ''

    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }])

    // Add placeholder for assistant response
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: new Date(),
    }])

    setIsLoading(true)

    // Send via WebSocket
    const documentIds = documents.map(d => d.id)
    ws.sendMessage(content.trim(), mode, documentIds, executionMode)
  }, [mode, documents, ws, executionMode])

  // Send message via REST API (non-streaming)
  const sendMessageRest = useCallback(async (content) => {
    if (!content.trim()) return

    setError(null)
    setIsLoading(true)

    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }])

    try {
      const response = await api.post('/api/chat', {
        message: content.trim(),
        mode,
        session_id: sessionId,
        documents: documents.map(d => d.id),
      })

      setSessionId(response.data.session_id)

      // Add assistant response
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.data.content,
        toolCalls: response.data.tool_calls,
        timestamp: new Date(),
      }])

      if (response.data.tool_calls) {
        setToolCalls(response.data.tool_calls)
      }
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message
      setError(errorMsg)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '',
        error: errorMsg,
        timestamp: new Date(),
      }])
    } finally {
      setIsLoading(false)
    }
  }, [mode, sessionId, documents])

  // Default to streaming
  const sendMessage = sendMessageStream

  // Clear conversation
  const clearMessages = useCallback(() => {
    setMessages([])
    setToolCalls([])
    setSessionId(null)
    setError(null)
  }, [])

  // Add document
  const addDocument = useCallback((doc) => {
    setDocuments(prev => [...prev, doc])
  }, [])

  // Remove document
  const removeDocument = useCallback((docId) => {
    setDocuments(prev => prev.filter(d => d.id !== docId))
  }, [])

  // Send user action for interactive mode code preview
  const sendUserAction = useCallback((action, modifiedCode = null) => {
    if (!codePreview) return

    ws.sendJson({
      type: 'user_action',
      action: action,
      code: modifiedCode,
      request_id: codePreview.requestId,
    })
    setCodePreview(null)
  }, [codePreview, ws])

  // Dismiss code preview (close without action)
  const dismissCodePreview = useCallback(() => {
    setCodePreview(null)
  }, [])

  return {
    messages,
    mode,
    setMode,
    sessionId,
    isLoading,
    isStreaming: ws.isStreaming,
    isConnected: ws.isConnected,
    error,
    documents,
    toolCalls,
    sendMessage,
    sendMessageRest,
    clearMessages,
    addDocument,
    removeDocument,
    // Interactive mode
    executionMode,
    setExecutionMode,
    codePreview,
    sendUserAction,
    dismissCodePreview,
  }
}

export default useChat
