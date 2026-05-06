"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.config import get_settings
from app.db.connection import init_db, close_db
from app.logging_conf import setup_logging
from app.api.middleware.logging import LoggingMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.api.routes import (
    auth,
    users,
    scenarios,
    personas,
    sessions,
    assignments,
    analytics,
    websocket,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    await init_db()
    yield
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Sales Teaching Assistant API",
        description="Real-Time AI Sales Role-Play Training Platform",
        version="0.1.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["100 per minute"],
        storage_uri="memory://"
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    from slowapi.middleware import SlowAPIMiddleware
    app.add_middleware(SlowAPIMiddleware)

    app.add_middleware(LoggingMiddleware)

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        """Limit the size of the request body to 10MB."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Payload too large. Maximum allowed size is 10MB."},
            )
        
        if request.method in ("POST", "PUT", "PATCH"):
            body_size = 0
            async for chunk in request.stream():
                body_size += len(chunk)
                if body_size > 10 * 1024 * 1024:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": "Payload too large. Maximum allowed size is 10MB."},
                    )
        
        response = await call_next(request)
        return response

    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        frame_ancestors = " ".join(settings.allowed_origins) if settings.allowed_origins else "'self'"
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; "
            f"script-src 'self' 'unsafe-inline'; "
            f"style-src 'self' 'unsafe-inline'; "
            f"img-src 'self' data:; "
            f"connect-src 'self' wss: https:; "
            f"frame-ancestors 'self' {frame_ancestors}"
        )
        return response

    @app.middleware("http")
    async def validate_origin(request: Request, call_next):
        """Server-side validation of the Origin header against allowed origins."""
        origin = request.headers.get("origin")
        if origin and origin not in settings.allowed_origins:
            if not (settings.debug and "localhost" in origin):
                import structlog
                logger = structlog.get_logger("app.security")
                logger.warning("Blocked request from unauthorized origin", origin=origin, path=request.url.path)
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": f"Origin {origin} is not allowed."},
                )
        
        response = await call_next(request)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Accept",
            "Origin",
        ],
    )
    
    import logging
    root_logger = logging.getLogger("app")
    root_logger.info(f"Startup: APP_ENV={settings.app_env}, DEBUG={settings.debug}")
    root_logger.info(f"Allowed Origins: {settings.allowed_origins}")
    
    api_prefix = "/api"
    app.include_router(auth.router, prefix=f"{api_prefix}/auth", tags=["Authentication"])
    app.include_router(users.router, prefix=f"{api_prefix}/users", tags=["Users"])
    app.include_router(scenarios.router, prefix=f"{api_prefix}/scenarios", tags=["Scenarios"])
    app.include_router(personas.router, prefix=f"{api_prefix}/personas", tags=["Personas"])
    app.include_router(sessions.router, prefix=f"{api_prefix}/sessions", tags=["Sessions"])
    app.include_router(assignments.router, prefix=f"{api_prefix}/assignments", tags=["Training Assignments"])
    app.include_router(analytics.router, prefix=f"{api_prefix}/analytics", tags=["Analytics"])
    app.include_router(websocket.router, prefix="/ws", tags=["WebSocket Call"])

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.1.0"}

    return app


setup_logging()
app = create_app()