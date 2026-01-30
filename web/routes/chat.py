"""
Chat Routes - API endpoints for chat and agent interactions.
"""

from typing import Optional, List, Literal
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
import json

from ..services import ChatService, DocumentService


router = APIRouter(prefix="/api", tags=["chat"])

# Services will be injected via dependency
_chat_service: Optional[ChatService] = None
_document_service: Optional[DocumentService] = None


def set_services(chat_service: ChatService, document_service: DocumentService):
    """Set service instances (called from server setup)."""
    global _chat_service, _document_service
    _chat_service = chat_service
    _document_service = document_service


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., description="User message")
    mode: Literal["chat", "agent"] = Field(default="agent", description="Operating mode")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    documents: Optional[List[str]] = Field(default=None, description="Document IDs to include as context")


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    session_id: str
    content: str
    mode: str
    tool_calls: Optional[List[dict]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message.

    In 'chat' mode, provides simple conversation without tools.
    In 'agent' mode, uses full tool-calling capabilities.
    """
    if _chat_service is None:
        raise HTTPException(status_code=500, detail="Chat service not initialized")

    # Build context from documents
    context = None
    if request.documents and _document_service:
        context = _document_service.get_documents_content(request.documents)

    # Process message
    result = await _chat_service.process_message(
        message=request.message,
        mode=request.mode,
        session_id=request.session_id,
        context=context,
    )

    return ChatResponse(**result)


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming responses.

    Message format:
    - Client sends: {"message": "...", "mode": "chat|agent", "documents": ["id1", ...]}
    - Client sends: {"type": "user_action", "action": "execute|skip|modify", "code": "...", "request_id": "..."}
    - Client sends: {"type": "set_execution_mode", "execution_mode": "auto|interactive"}
    - Server sends events:
      - {"type": "session", "session_id": "..."}
      - {"type": "thinking", "content": "..."}
      - {"type": "tool_call", "content": "...", "metadata": {...}}
      - {"type": "tool_result", "content": "...", "metadata": {...}}
      - {"type": "code_preview", "content": "...", "metadata": {"file_path": "...", "language": "...", "request_id": "..."}}
      - {"type": "content", "content": "..."}
      - {"type": "complete", "content": "..."}
      - {"type": "error", "content": "..."}
    """
    await websocket.accept()

    if _chat_service is None:
        await websocket.send_json({"type": "error", "content": "Chat service not initialized"})
        await websocket.close()
        return

    # Get or create session for action handling
    current_session = None

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                request = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON"})
                continue

            msg_type = request.get("type", "message")

            # Handle user action (response to code_preview)
            if msg_type == "user_action":
                if current_session:
                    await current_session.send_user_action({
                        "action": request.get("action", "execute"),
                        "code": request.get("code"),
                        "request_id": request.get("request_id"),
                    })
                continue

            # Handle execution mode change
            if msg_type == "set_execution_mode":
                execution_mode = request.get("execution_mode", "auto")
                if current_session:
                    current_session.set_execution_mode(execution_mode)
                await websocket.send_json({
                    "type": "status",
                    "content": f"Execution mode set to: {execution_mode}"
                })
                continue

            # Regular message handling
            message = request.get("message", "")
            mode = request.get("mode", "agent")
            documents = request.get("documents", [])
            execution_mode = request.get("execution_mode", "auto")

            if not message:
                await websocket.send_json({"type": "error", "content": "Message required"})
                continue

            # Build context from documents
            context = None
            if documents and _document_service:
                context = _document_service.get_documents_content(documents)

            # Stream response
            try:
                async for event in _chat_service.process_message_stream(
                    message=message,
                    mode=mode,
                    session_id=session_id if session_id != "new" else None,
                    context=context,
                    execution_mode=execution_mode,
                ):
                    # Update current session reference on session event
                    if event.get("type") == "session":
                        current_session = _chat_service.sessions.get_session(event.get("session_id"))
                        if current_session:
                            current_session.set_execution_mode(execution_mode)
                    await websocket.send_json(event)
            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})

    except WebSocketDisconnect:
        pass  # Client disconnected
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if _chat_service is None:
        raise HTTPException(status_code=500, detail="Chat service not initialized")

    if _chat_service.sessions.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
