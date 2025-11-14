"""
Model Evaluation Component

Evaluates the trained model and generates metrics reports including NDCG@k scores.

Author: Equipo ADX
Date: 2025-11-13
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent / '3-training'))
from model import EstablishmentDNN

logger.remove()
logger.add(sys.stderr, format="<green>{time}</green> | <level>{level}</level> | <level>{message}</level>", level="INFO")


def evaluate_model(model_path, data_path, encoders_path, output_path, k_values=[3, 5, 10]):
    """Evaluate model and generate metrics."""
    logger.info("Starting model evaluation...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    df = pd.read_parquet(data_path)
    feature_cols_path = Path(model_path).parent / 'feature_columns.pkl'
    feature_cols = joblib.load(feature_cols_path)

    # Convert nullable pandas types to standard numpy types to avoid object dtype
    X = df[feature_cols].astype('float32').values
    y = df['establecimiento_encoded'].astype('int64').values
    
    # Load encoders
    encoder_dict = joblib.load(encoders_path)
    num_establishments = len(encoder_dict['establishment_encoder'].classes_)
    
    # Initialize model
    model = EstablishmentDNN(
        input_dim=X.shape[1],
        num_establishments=num_establishments
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    logger.info("Computing predictions...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
    
    # Compute metrics
    metrics = {}
    
    # Accuracy (top-1)
    predictions = np.argmax(probs, axis=1)
    accuracy = (predictions == y).mean()
    metrics['accuracy'] = float(accuracy)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # NDCG@k scores
    for k in k_values:
        # Create relevance matrix (1 for true establishment, 0 for others)
        y_true = np.zeros_like(probs)
        y_true[np.arange(len(y)), y] = 1

        ndcg = ndcg_score(y_true, probs, k=k)
        # Use underscore instead of @ for MLflow compatibility
        metrics[f'ndcg_at_{k}'] = float(ndcg)
        logger.success(f"NDCG@{k}: {ndcg:.4f}")
    
    # Save metrics
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.success(f"Metrics saved to {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--encoders_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        encoders_path=args.encoders_path,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
