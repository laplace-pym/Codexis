"""
Web Routes - API endpoint definitions.
"""

from .chat import router as chat_router
from .documents import router as documents_router

__all__ = ["chat_router", "documents_router"]
