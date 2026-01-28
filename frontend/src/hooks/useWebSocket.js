import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * WebSocket hook for streaming chat responses.
 */
export function useWebSocket(sessionId = 'new') {
  const [isConnected, setIsConnected] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const wsRef = useRef(null)
  const eventHandlersRef = useRef({})
  const sessionIdRef = useRef(sessionId)

  // Keep sessionId ref up to date
  sessionIdRef.current = sessionId

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    // In development, connect directly to backend; in production, use same host
    const isDev = import.meta.env.DEV
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = isDev ? 'localhost:8000' : window.location.host
    const wsUrl = `${protocol}//${host}/api/ws/${sessionIdRef.current}`

    console.log('Connecting to WebSocket:', wsUrl)
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    ws.onclose = () => {
      setIsConnected(false)
      setIsStreaming(false)
    }

    ws.onerror = (e) => {
      setError('WebSocket connection error')
      setIsConnected(false)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        // Handle different event types
        if (data.type === 'complete') {
          setIsStreaming(false)
        }

        // Call registered handler
        const handler = eventHandlersRef.current[data.type] || eventHandlersRef.current['*']
        if (handler) {
          handler(data)
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }
  }, [sessionId])

  // Disconnect
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  // Send a message
  const sendMessage = useCallback((message, mode = 'agent', documents = []) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      setError('WebSocket not connected')
      return false
    }

    setIsStreaming(true)
    setError(null)

    wsRef.current.send(JSON.stringify({
      message,
      mode,
      documents,
    }))

    return true
  }, [])

  // Register event handler
  const onEvent = useCallback((type, handler) => {
    eventHandlersRef.current[type] = handler
  }, [])

  // Auto-connect on mount only
  useEffect(() => {
    connect()
    return () => disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])  // Only run on mount/unmount

  return {
    isConnected,
    isStreaming,
    error,
    connect,
    disconnect,
    sendMessage,
    onEvent,
  }
}

export default useWebSocket
