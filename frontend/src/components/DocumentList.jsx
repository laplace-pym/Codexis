import React from 'react'
import { FileText, X } from 'lucide-react'

/**
 * Displays list of uploaded documents.
 */
export function DocumentList({ documents, onRemove }) {
  if (documents.length === 0) {
    return null
  }

  return (
    <div className="flex flex-wrap gap-2">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="
            flex items-center gap-2 bg-slate-100 rounded-full px-3 py-1.5
            text-sm text-slate-700
          "
        >
          <FileText size={14} className="text-slate-500" />
          <span className="max-w-[150px] truncate">{doc.filename || doc.name}</span>
          <button
            onClick={() => onRemove(doc.id)}
            className="text-slate-400 hover:text-slate-600 transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  )
}

export default DocumentList
