"""
FastAPI Server - Main web application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .services import ChatService, SessionManager, DocumentService, TeamService
from .routes import chat_router, documents_router
from .routes.chat import set_services as set_chat_services
from .routes.documents import set_service as set_document_service


# Global service instances
_session_manager: SessionManager | None = None
_chat_service: ChatService | None = None
_document_service: DocumentService | None = None
_team_service: TeamService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _session_manager, _chat_service, _document_service, _team_service

    # Startup
    _session_manager = SessionManager()
    _chat_service = ChatService(_session_manager)
    _document_service = DocumentService()
    _team_service = TeamService()

    # Inject services into routes
    set_chat_services(_chat_service, _document_service, _team_service)
    set_document_service(_document_service)

    # Start cleanup loop
    await _session_manager.start_cleanup_loop()

    yield

    # Shutdown
    _session_manager.stop_cleanup_loop()


def create_app(
    title: str = "Codexis API",
    description: str = "AI Coding Agent with Chat and Agent modes",
    version: str = "1.0.0",
    cors_origins: list[str] | None = None,
    serve_frontend: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        description: API description
        version: API version
        cors_origins: Allowed CORS origins
        serve_frontend: Whether to serve the frontend static files

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
    )

    # CORS configuration
    if cors_origins is None:
        cors_origins = [
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",  # Alternative dev server
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat_router)
    app.include_router(documents_router)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": version}

    # Serve frontend if requested
    if serve_frontend:
        frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
        if frontend_path.exists():
            app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

    return app


# Default app instance for running with uvicorn
app = create_app()
