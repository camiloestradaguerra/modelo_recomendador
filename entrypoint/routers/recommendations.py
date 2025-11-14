"""Recommendations router."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException

sys.path.append(str(Path(__file__).parent.parent.parent / 'src/pipelines/3-training'))
from model import EstablishmentDNN, LocationTimeFilter
from entrypoint.schemas import RecommendationRequest, RecommendationResponse, RecommendationItem

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Global variables for model (loaded once at startup)
MODEL = None
ENCODER_DICT = None
FEATURE_ENGINEER = None
LOCATION_FILTER = None
FEATURE_COLS = None
DEVICE = None


def load_model():
    """Load model and artifacts at startup."""
    global MODEL, ENCODER_DICT, FEATURE_ENGINEER, LOCATION_FILTER, FEATURE_COLS, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = Path('models/dnn_model.pth')
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Load encoders
    encoders_path = Path('models/label_encoders.pkl')
    ENCODER_DICT = joblib.load(encoders_path)
    
    # Load feature columns
    feature_cols_path = Path('models/feature_columns.pkl')
    FEATURE_COLS = joblib.load(feature_cols_path)
    
    # Load location filter
    filter_path = Path('models/location_filter.pkl')
    LOCATION_FILTER = joblib.load(filter_path)
    
    # Load feature engineer
    import importlib.util
    feature_eng_path = Path(__file__).parent.parent.parent / 'src/pipelines/2-feature_engineering/main.py'
    spec = importlib.util.spec_from_file_location("feature_engineering_module", feature_eng_path)
    feature_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    FEATURE_ENGINEER = feature_module.FeatureEngineer.load_encoders(encoders_path)
    
    # Initialize model
    num_establishments = len(ENCODER_DICT['establishment_encoder'].classes_)
    MODEL = EstablishmentDNN(
        input_dim=len(FEATURE_COLS),
        num_establishments=num_establishments
    ).to(DEVICE)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.eval()


@router.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized restaurant recommendations."""

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Try to load real user data from sampled dataset
    user_data = None
    used_real_user_data = False
    sampled_data_path = Path('data/02-sampled/sampled_data.parquet')

    if sampled_data_path.exists():
        try:
            df_sampled = pd.read_parquet(sampled_data_path)
            # Normalize column names to lowercase
            df_sampled.columns = df_sampled.columns.str.lower()

            # Convert id_persona to int for comparison (via float to handle string format)
            df_sampled['id_persona'] = df_sampled['id_persona'].astype(float).astype(int)

            # Find user's most recent transaction
            user_transactions = df_sampled[df_sampled['id_persona'] == int(request.id_persona)]

            if len(user_transactions) > 0:
                # Use the most recent transaction as template
                user_data = user_transactions.iloc[-1].to_dict()
                # Update with request parameters
                user_data['ciudad'] = request.ciudad
                user_data['hora'] = request.hora
                user_data['hora_inicio'] = pd.Timestamp.now().replace(hour=request.hora)
                used_real_user_data = True
            else:
                # User not found - use dataset averages for cold start
                avg_data = {
                    'id_persona': request.id_persona,
                    'ciudad': request.ciudad,
                    'hora': request.hora,
                    'hora_inicio': pd.Timestamp.now().replace(hour=request.hora),
                    'monto': float(df_sampled['monto'].median()),
                    'edad': int(df_sampled['edad'].median()),
                    'antiguedad_socio_unico': float(df_sampled['antiguedad_socio_unico'].median()),
                    'especialidad': df_sampled['especialidad'].mode()[0] if len(df_sampled['especialidad'].mode()) > 0 else 'GENERAL',
                    'estado_civil': df_sampled['estado_civil'].mode()[0] if len(df_sampled['estado_civil'].mode()) > 0 else 'SOLTERO',
                    'genero': df_sampled['genero'].mode()[0] if len(df_sampled['genero'].mode()) > 0 else 'M',
                    'rol': df_sampled['rol'].mode()[0] if len(df_sampled['rol'].mode()) > 0 else 'SOCIO',
                    'segmento_comercial': df_sampled['segmento_comercial'].mode()[0] if len(df_sampled['segmento_comercial'].mode()) > 0 else 'MASIVO',
                    'zona': df_sampled['zona'].mode()[0] if len(df_sampled['zona'].mode()) > 0 else 'NORTE',
                    'region': df_sampled['region'].mode()[0] if len(df_sampled['region'].mode()) > 0 else 'SIERRA',
                    'cadena': 'INDEPENDIENTE',
                    'establecimiento': 'UNKNOWN_USER'
                }
                user_data = avg_data
        except Exception as e:
            # Error loading data - use minimal defaults
            pass

    # If still no user data (file doesn't exist or error), raise error
    if user_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"User {request.id_persona} not found and no default data available"
        )

    # Use real or averaged user data
    input_df = pd.DataFrame([user_data])
    
    # Apply feature engineering
    input_processed = FEATURE_ENGINEER.transform(input_df)

    # Extract features and convert to float32 to avoid NaN from nullable types
    X = input_processed[FEATURE_COLS].fillna(0.0).astype('float32').values
    
    # Get predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        outputs = MODEL(X_tensor)
        
        # Apply location/time filter
        establishment_names = list(ENCODER_DICT['establishment_encoder'].classes_)
        filtered_probs = LOCATION_FILTER.apply(
            predictions=outputs,
            user_ciudad=request.ciudad,
            hora=request.hora,
            establishment_names=establishment_names
        )

        # Replace any NaN or inf values with 0
        filtered_probs = torch.where(
            torch.isnan(filtered_probs) | torch.isinf(filtered_probs),
            torch.zeros_like(filtered_probs),
            filtered_probs
        )

        # Get top-k
        probs, indices = torch.topk(filtered_probs[0], k=request.k)
        
    # Build response
    recommendations = []
    for idx, prob in zip(indices.cpu().numpy(), probs.cpu().numpy()):
        est_name = establishment_names[idx]
        est_ciudad = LOCATION_FILTER.establishment_locations.get(est_name, "Unknown")
        
        recommendations.append(RecommendationItem(
            establecimiento=est_name,
            probability=float(prob),
            ciudad=est_ciudad
        ))
    
    return RecommendationResponse(
        recommendations=recommendations,
        filtered_by_location=True,
        filtered_by_time=True,
        used_real_user_data=used_real_user_data
    )
