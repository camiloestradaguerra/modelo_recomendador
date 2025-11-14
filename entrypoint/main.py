"""
FastAPI Application for Restaurant Recommendations

This API provides endpoints for:
- Health checks
- Personalized restaurant recommendations with location/time filtering

The API implements best practices:
- CORS middleware for cross-origin requests
- Structured logging
- Pydantic validation
- Router-based organization
- Proper error handling

Author: Equipo ADX
Date: 2025-11-13

AWS SageMaker Deployment:
To deploy on SageMaker Endpoint, create a custom inference handler:

from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
class ModelHandler(default_inference_handler.DefaultInferenceHandler):
    def default_model_fn(self, model_dir):
        # Load model from model_dir
        pass
    
    def default_input_fn(self, input_data, content_type):
        # Parse input
        pass
    
    def default_predict_fn(self, data, model):
        # Run inference
        pass
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from entrypoint.routers import health, recommendations

# Initialize FastAPI app
app = FastAPI(
    title="RecSys V3 - Restaurant Recommendation API",
    description="Personalized restaurant recommendations with location and time filtering",
    version="1.0.0",
    contact={
        "name": "Equipo ADX",
        "email": "contact@example.com"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(recommendations.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RecSys V3 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
