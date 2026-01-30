import React, { useState } from 'react'
import { Trash2, PanelRightOpen, PanelRightClose, Wifi, WifiOff, FileUp, Zap, MousePointer } from 'lucide-react'
import { ModeSelector } from './ModeSelector'
import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'
import { DocumentUpload } from './DocumentUpload'
import { DocumentList } from './DocumentList'
import { ToolPanel } from './ToolPanel'
import { CodePreview } from './CodePreview'
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
    // Interactive mode
    executionMode,
    setExecutionMode,
    codePreview,
    sendUserAction,
    dismissCodePreview,
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
              {/* Execution mode toggle (agent mode only) */}
              {mode === 'agent' && (
                <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-1">
                  <button
                    onClick={() => setExecutionMode('auto')}
                    className={`
                      px-2 py-1 text-xs font-medium rounded flex items-center gap-1 transition-colors
                      ${executionMode === 'auto'
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-slate-500 hover:text-slate-700'}
                    `}
                    title="Auto mode - execute code automatically"
                  >
                    <Zap size={12} />
                    Auto
                  </button>
                  <button
                    onClick={() => setExecutionMode('interactive')}
                    className={`
                      px-2 py-1 text-xs font-medium rounded flex items-center gap-1 transition-colors
                      ${executionMode === 'interactive'
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-slate-500 hover:text-slate-700'}
                    `}
                    title="Interactive mode - review code before execution"
                  >
                    <MousePointer size={12} />
                    Interactive
                  </button>
                </div>
              )}

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

      {/* Code preview modal (interactive mode) */}
      {codePreview && (
        <CodePreview
          code={codePreview.code}
          language={codePreview.language}
          filePath={codePreview.filePath}
          requestId={codePreview.requestId}
          onExecute={() => sendUserAction('execute')}
          onSkip={() => sendUserAction('skip')}
          onModify={(modifiedCode) => sendUserAction('modify', modifiedCode)}
          onClose={dismissCodePreview}
        />
      )}
    </div>
  )
}

export default ChatContainer
