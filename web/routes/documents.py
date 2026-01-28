"""
Documents Routes - API endpoints for document upload and management.
"""

from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from ..services import DocumentService


router = APIRouter(prefix="/api", tags=["documents"])

# Service will be injected via dependency
_document_service: Optional[DocumentService] = None


def set_service(document_service: DocumentService):
    """Set service instance (called from server setup)."""
    global _document_service
    _document_service = document_service


class DocumentResponse(BaseModel):
    """Response body for document operations."""
    id: str
    filename: str
    type: str
    uploaded_at: str
    size_bytes: int
    has_content: bool


class DocumentListResponse(BaseModel):
    """Response body for document list."""
    documents: List[DocumentResponse]
    total: int


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document.

    Supported formats: PDF, DOCX, PPTX, TXT, MD
    The document content will be automatically extracted for use as context.
    """
    if _document_service is None:
        raise HTTPException(status_code=500, detail="Document service not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    # Read file content
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Upload and process
    try:
        document = await _document_service.upload_document(
            filename=file.filename,
            content=content,
            extract_text=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return DocumentResponse(**document.to_dict())


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    if _document_service is None:
        raise HTTPException(status_code=500, detail="Document service not initialized")

    documents = _document_service.list_documents()
    return DocumentListResponse(
        documents=[DocumentResponse(**d.to_dict()) for d in documents],
        total=len(documents),
    )


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get document details."""
    if _document_service is None:
        raise HTTPException(status_code=500, detail="Document service not initialized")

    document = _document_service.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(**document.to_dict())


@router.get("/documents/{doc_id}/content")
async def get_document_content(doc_id: str):
    """Get extracted text content from a document."""
    if _document_service is None:
        raise HTTPException(status_code=500, detail="Document service not initialized")

    document = _document_service.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc_id,
        "filename": document.filename,
        "content": document.content or "",
        "has_content": document.content is not None,
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    if _document_service is None:
        raise HTTPException(status_code=500, detail="Document service not initialized")

    if _document_service.delete_document(doc_id):
        return {"status": "deleted", "id": doc_id}
    else:
        raise HTTPException(status_code=404, detail="Document not found")
