"""Pydantic schemas for API request/response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request schema for restaurant recommendations."""
    
    id_persona: float = Field(..., description="User ID")
    ciudad: str = Field(..., description="City where user is located")
    hora: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    k: int = Field(5, ge=1, le=20, description="Number of recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id_persona": 21096.0,
                "ciudad": "Quito",
                "hora": 14,
                "k": 5
            }
        }


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    
    establecimiento: str
    probability: float
    ciudad: str


class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""

    recommendations: List[RecommendationItem]
    filtered_by_location: bool
    filtered_by_time: bool
    used_real_user_data: bool = Field(..., description="True if real user data was used, False if cold start with averages")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    version: str
