#!/usr/bin/env python3
"""
Web Server Entry Point - Launch the Codexis web API.

Usage:
    python run_web.py                    # Default: localhost:8000
    python run_web.py --port 8080        # Custom port
    python run_web.py --host 0.0.0.0     # Listen on all interfaces
    python run_web.py --reload           # Auto-reload for development
    python run_web.py --serve-frontend   # Also serve the frontend build
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Codexis web API server"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--serve-frontend",
        action="store_true",
        help="Serve the frontend build from /frontend/dist"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    args = parser.parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════╗
    ║           Codexis Web Server                  ║
    ╠═══════════════════════════════════════════════╣
    ║  API:      http://{args.host}:{args.port}            ║
    ║  Docs:     http://{args.host}:{args.port}/docs       ║
    ║  Health:   http://{args.host}:{args.port}/health     ║
    ╚═══════════════════════════════════════════════╝
    """)

    # Use app factory for reload mode, or direct import otherwise
    if args.reload:
        uvicorn.run(
            "web.server:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["web", "agent", "tools", "llm"],
        )
    else:
        from web.server import create_app
        app = create_app(serve_frontend=args.serve_frontend)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
