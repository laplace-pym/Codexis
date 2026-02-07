"""
Chat Routes - API endpoints for chat and agent interactions.
"""

from typing import Optional, List, Literal
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
import json

from ..services import ChatService, DocumentService, TeamService


router = APIRouter(prefix="/api", tags=["chat"])

# Services will be injected via dependency
_chat_service: Optional[ChatService] = None
_document_service: Optional[DocumentService] = None
_team_service: Optional[TeamService] = None


def set_services(
    chat_service: ChatService,
    document_service: DocumentService,
    team_service: Optional[TeamService] = None,
):
    """Set service instances (called from server setup)."""
    global _chat_service, _document_service, _team_service
    _chat_service = chat_service
    _document_service = document_service
    _team_service = team_service


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


# ---------- Team Routes ----------


class TeamCreateRequest(BaseModel):
    """Request to create and execute a team task."""
    task: str = Field(..., description="High-level task for the team")
    members: Optional[List[dict]] = Field(
        default=None,
        description='Member configs: [{"name": "...", "role": "..."}]',
    )
    provider: Optional[str] = Field(
        default=None, description="Default LLM provider"
    )


@router.post("/team/create")
async def create_team(request: TeamCreateRequest):
    """
    Create a team and start executing a task.

    If members is not provided, a default team (architect, developer, tester)
    is created.

    Returns the team_id and initial member info. The task runs in a
    background thread; poll /api/team/{team_id}/status for progress.
    """
    if _team_service is None:
        raise HTTPException(
            status_code=500, detail="Team service not initialized"
        )

    import concurrent.futures

    # Create team via service
    from agent.team import TeamManager

    if request.members:
        team = _team_service.manager.create_team(
            members_config=request.members,
            provider=request.provider,
        )
    else:
        team = _team_service.manager.create_default_team(
            provider=request.provider,
        )

    # Execute in background thread
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(team.execute, request.task)

    return {
        "team_id": team.team_id,
        "status": "started",
        "members": [m.info.to_dict() for m in team.members.values()],
    }


@router.get("/team/{team_id}/status")
async def get_team_status(team_id: str):
    """Get current progress of a team."""
    if _team_service is None:
        raise HTTPException(
            status_code=500, detail="Team service not initialized"
        )

    progress = _team_service.get_team_progress(team_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Team not found")

    return progress.to_dict()
