"""
Web Module - FastAPI backend for Codexis.

This module provides:
- REST API endpoints for chat and document operations
- WebSocket support for streaming responses
- Session management
"""

from .server import create_app

__all__ = ["create_app"]
