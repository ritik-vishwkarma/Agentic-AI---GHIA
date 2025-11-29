"""FastAPI Application Entry Point"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from app.config import get_settings
from app.db.database import init_db
from app.routes import intake, dashboard
from app.routes.feedback import router as feedback_router

try:
    from app.api.routes.intake import router as enhanced_intake_router

    ENHANCED_INTAKE_AVAILABLE = True
except ImportError:
    ENHANCED_INTAKE_AVAILABLE = False

from app.routes.livekit import router as livekit_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    # Initialize database
    init_db()
    print("✅ Database initialized")

    # Check LiveKit status
    from app.services.livekit import get_livekit_service

    livekit = get_livekit_service()
    if livekit.is_configured():
        print("✅ LiveKit configured for real-time audio")
    else:
        print("⚠️ LiveKit not configured (optional for demo)")

    yield
    print("👋 Shutting down GHIA")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI-powered multi-agent health intake system for rural India",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(intake.router, prefix="/api/intake", tags=["Intake"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(feedback_router)
app.include_router(livekit_router, prefix="/api/livekit", tags=["LiveKit"])

if ENHANCED_INTAKE_AVAILABLE:
    app.include_router(enhanced_intake_router, tags=["Enhanced Intake"])
    




@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "intake": "/api/intake",
            "dashboard": "/api/dashboard",
            "livekit": "/api/livekit",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
