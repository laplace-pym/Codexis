import React, { useState } from 'react'
import { Trash2, PanelRightOpen, PanelRightClose, Wifi, WifiOff, FileUp } from 'lucide-react'
import { ModeSelector } from './ModeSelector'
import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'
import { DocumentUpload } from './DocumentUpload'
import { DocumentList } from './DocumentList'
import { ToolPanel } from './ToolPanel'
import { useChat } from '../hooks/useChat'

/**
 * Main chat container component.
 */
export function ChatContainer() {
  const {
    messages,
    mode,
    setMode,
    isLoading,
    isStreaming,
    isConnected,
    error,
    documents,
    toolCalls,
    sendMessage,
    clearMessages,
    addDocument,
    removeDocument,
  } = useChat()

  const [showToolPanel, setShowToolPanel] = useState(true)
  const [showUpload, setShowUpload] = useState(false)

  return (
    <div className="flex h-screen bg-white">
      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="border-b border-slate-200 px-4 py-3">
          <div className="flex items-center justify-between max-w-4xl mx-auto">
            <div className="flex items-center gap-4">
              <h1 className="text-xl font-semibold text-slate-900">Codexis</h1>
              <ModeSelector
                mode={mode}
                onModeChange={setMode}
                disabled={isLoading}
              />
            </div>

            <div className="flex items-center gap-2">
              {/* Connection status */}
              <div className={`flex items-center gap-1 text-sm ${isConnected ? 'text-green-600' : 'text-slate-400'}`}>
                {isConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
                <span className="hidden sm:inline">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              {/* Upload toggle */}
              <button
                onClick={() => setShowUpload(!showUpload)}
                className={`
                  p-2 rounded-lg transition-colors
                  ${showUpload ? 'bg-blue-100 text-blue-600' : 'text-slate-500 hover:bg-slate-100'}
                `}
                title="Upload document"
              >
                <FileUp size={20} />
              </button>

              {/* Tool panel toggle (only in agent mode) */}
              {mode === 'agent' && (
                <button
                  onClick={() => setShowToolPanel(!showToolPanel)}
                  className="p-2 text-slate-500 hover:bg-slate-100 rounded-lg transition-colors"
                  title={showToolPanel ? 'Hide tool panel' : 'Show tool panel'}
                >
                  {showToolPanel ? <PanelRightClose size={20} /> : <PanelRightOpen size={20} />}
                </button>
              )}

              {/* Clear chat */}
              <button
                onClick={clearMessages}
                disabled={messages.length === 0}
                className="
                  p-2 text-slate-500 hover:bg-slate-100 rounded-lg transition-colors
                  disabled:opacity-50 disabled:cursor-not-allowed
                "
                title="Clear conversation"
              >
                <Trash2 size={20} />
              </button>
            </div>
          </div>
        </header>

        {/* Upload area (collapsible) */}
        {showUpload && (
          <div className="border-b border-slate-200 p-4 bg-slate-50">
            <div className="max-w-4xl mx-auto space-y-3">
              <DocumentUpload onUpload={addDocument} disabled={isLoading} />
              <DocumentList documents={documents} onRemove={removeDocument} />
            </div>
          </div>
        )}

        {/* Document pills (always visible if documents present) */}
        {!showUpload && documents.length > 0 && (
          <div className="border-b border-slate-200 px-4 py-2">
            <div className="max-w-4xl mx-auto">
              <DocumentList documents={documents} onRemove={removeDocument} />
            </div>
          </div>
        )}

        {/* Error banner */}
        {error && (
          <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-red-700 text-sm">
            {error}
          </div>
        )}

        {/* Messages */}
        <MessageList messages={messages} />

        {/* Input */}
        <MessageInput
          onSend={sendMessage}
          isLoading={isLoading || isStreaming}
          disabled={!isConnected}
          placeholder={
            mode === 'chat'
              ? "Ask me anything..."
              : "Give me a coding task..."
          }
        />
      </div>

      {/* Tool panel (agent mode only) */}
      {mode === 'agent' && (
        <ToolPanel
          toolCalls={toolCalls}
          isVisible={showToolPanel}
        />
      )}
    </div>
  )
}

export default ChatContainer
