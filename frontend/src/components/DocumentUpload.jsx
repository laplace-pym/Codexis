import React, { useState, useRef } from 'react'
import { Upload, File, X, Loader2 } from 'lucide-react'
import api from '../api'

/**
 * Document upload component with drag-and-drop support.
 */
export function DocumentUpload({ onUpload, disabled }) {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    if (!disabled) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = async (e) => {
    e.preventDefault()
    setIsDragging(false)

    if (disabled) return

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      await uploadFile(files[0])
    }
  }

  const handleFileSelect = async (e) => {
    const files = e.target.files
    if (files && files.length > 0) {
      await uploadFile(files[0])
    }
    // Reset input
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  const uploadFile = async (file) => {
    setIsUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      onUpload({
        ...response.data,
        name: file.name,
      })
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Upload failed'
      setError(errorMsg)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="relative">
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        onChange={handleFileSelect}
        accept=".pdf,.docx,.pptx,.ppt,.txt,.md"
        disabled={disabled || isUploading}
        className="hidden"
      />

      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors
          ${isDragging
            ? 'border-blue-500 bg-blue-50'
            : 'border-slate-200 hover:border-slate-300'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        {isUploading ? (
          <div className="flex items-center justify-center gap-2 text-slate-500">
            <Loader2 size={20} className="animate-spin" />
            <span>Uploading...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-slate-500">
            <Upload size={24} />
            <span className="text-sm">
              Drop a file here or click to upload
            </span>
            <span className="text-xs text-slate-400">
              PDF, DOCX, PPTX, TXT
            </span>
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="mt-2 text-sm text-red-600 flex items-center gap-1">
          <X size={14} />
          {error}
        </div>
      )}
    </div>
  )
}

export default DocumentUpload
