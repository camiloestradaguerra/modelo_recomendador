"""Health check router."""

from fastapi import APIRouter
from entrypoint.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    }
