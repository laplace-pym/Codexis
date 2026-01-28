"""
Document Service - Handles document upload and processing.
"""

import uuid
import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from tools.doc_tools import ReadPDFTool, ReadDocxTool, ReadPPTTool


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Represents an uploaded document."""
    id: str
    filename: str
    doc_type: DocumentType
    file_path: str
    content: Optional[str] = None
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "filename": self.filename,
            "type": self.doc_type.value,
            "uploaded_at": self.uploaded_at.isoformat(),
            "size_bytes": self.size_bytes,
            "has_content": self.content is not None,
        }


class DocumentService:
    """
    Service for handling document uploads and extraction.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".pptx": DocumentType.PPTX,
        ".ppt": DocumentType.PPTX,
        ".txt": DocumentType.TXT,
        ".md": DocumentType.TXT,
    }

    def __init__(self, upload_dir: str = "./uploads"):
        """
        Initialize document service.

        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self._documents: Dict[str, Document] = {}

        # Initialize document tools
        self._pdf_tool = ReadPDFTool()
        self._docx_tool = ReadDocxTool()
        self._ppt_tool = ReadPPTTool()

    def _get_doc_type(self, filename: str) -> DocumentType:
        """Determine document type from filename."""
        ext = Path(filename).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, DocumentType.UNKNOWN)

    async def upload_document(
        self,
        filename: str,
        content: bytes,
        extract_text: bool = True,
    ) -> Document:
        """
        Upload and process a document.

        Args:
            filename: Original filename
            content: File content bytes
            extract_text: Whether to extract text content

        Returns:
            Document instance
        """
        doc_id = str(uuid.uuid4())
        doc_type = self._get_doc_type(filename)

        if doc_type == DocumentType.UNKNOWN:
            raise ValueError(f"Unsupported file type: {Path(filename).suffix}")

        # Save file
        safe_filename = f"{doc_id}_{Path(filename).name}"
        file_path = self.upload_dir / safe_filename

        with open(file_path, "wb") as f:
            f.write(content)

        # Create document record
        document = Document(
            id=doc_id,
            filename=filename,
            doc_type=doc_type,
            file_path=str(file_path),
            size_bytes=len(content),
        )

        # Extract text if requested
        if extract_text:
            document.content = self._extract_text(document)

        self._documents[doc_id] = document
        return document

    def _extract_text(self, document: Document) -> Optional[str]:
        """Extract text from document."""
        try:
            if document.doc_type == DocumentType.PDF:
                result = self._pdf_tool.execute(path=document.file_path)
                if result.success:
                    return result.output
            elif document.doc_type == DocumentType.DOCX:
                result = self._docx_tool.execute(path=document.file_path)
                if result.success:
                    return result.output
            elif document.doc_type == DocumentType.PPTX:
                result = self._ppt_tool.execute(path=document.file_path)
                if result.success:
                    return result.output
            elif document.doc_type == DocumentType.TXT:
                with open(document.file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            document.metadata["extraction_error"] = str(e)
        return None

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get extracted text content for a document."""
        doc = self._documents.get(doc_id)
        if doc:
            return doc.content
        return None

    def get_documents_content(self, doc_ids: List[str]) -> str:
        """
        Get combined content from multiple documents.

        Args:
            doc_ids: List of document IDs

        Returns:
            Combined content string
        """
        contents = []
        for doc_id in doc_ids:
            doc = self._documents.get(doc_id)
            if doc and doc.content:
                contents.append(f"=== {doc.filename} ===\n{doc.content}")
        return "\n\n".join(contents)

    def list_documents(self) -> List[Document]:
        """List all documents."""
        return list(self._documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        doc = self._documents.get(doc_id)
        if doc:
            # Delete file
            try:
                os.remove(doc.file_path)
            except OSError:
                pass

            del self._documents[doc_id]
            return True
        return False
