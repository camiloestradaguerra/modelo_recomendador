"""
Model Registration Component

Registers the trained model in MLflow with metrics and metadata.

Author: Equipo ADX
Date: 2025-11-13

AWS SageMaker Note:
For SageMaker Model Registry, replace MLflow calls with boto3:
    sm_client = boto3.client('sagemaker')
    model_package_arn = sm_client.create_model_package(...)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import torch
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent / '3-training'))
from model import EstablishmentDNN

logger.remove()
logger.add(sys.stderr, format="<green>{time}</green> | <level>{level}</level> | <level>{message}</level>", level="INFO")


def register_model(model_path, encoders_path, metrics_path, model_name):
    """Register model in MLflow with metadata and tags."""
    logger.info("Starting model registration...")

    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    encoder_dict = joblib.load(encoders_path)
    num_establishments = len(encoder_dict['establishment_encoder'].classes_)

    # Load feature columns to get input dimension
    feature_cols_path = Path(model_path).parent / 'feature_columns.pkl'
    feature_cols = joblib.load(feature_cols_path)
    input_dim = len(feature_cols)

    # Initialize model
    model = EstablishmentDNN(
        input_dim=input_dim,
        num_establishments=num_establishments
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Set MLflow tracking URI (local or remote)
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'mlruns')
    mlflow.set_tracking_uri(mlflow_uri)

    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_registration") as run:
        # Log parameters (best hyperparameters)
        mlflow.log_params({
            'batch_size': 32,
            'learning_rate': 0.0001967641848109,
            'hidden_dim1': 1024,
            'hidden_dim2': 256,
            'hidden_dim3': 256,
            'dropout_rate': 0.1429465700244763,
            'weight_decay': 0.00008261871088,
            'input_dim': input_dim,
            'num_establishments': num_establishments
        })

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model using PyTorch flavor (required for model registry)
        mlflow.pytorch.log_model(model, "model")

        # Log additional artifacts
        mlflow.log_artifact(encoders_path, "encoders")
        mlflow.log_artifact(metrics_path, "metrics")

        # Add tags (GitHub commit info if available)
        tags = {
            'model_type': 'deep_neural_network',
            'framework': 'pytorch',
            'problem_type': 'multi-class_classification',
            'use_case': 'restaurant_recommendation',
            'git_commit': os.environ.get('GITHUB_SHA', 'local'),
            'git_branch': os.environ.get('GITHUB_REF_NAME', 'local')
        }
        mlflow.set_tags(tags)

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.success(f"Model registered: {model_name} version {registered_model.version}")
        logger.info(f"Run ID: {run.info.run_id}")

    return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Register model in MLflow")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--encoders_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="recsys_dnn")
    args = parser.parse_args()
    
    register_model(
        model_path=args.model_path,
        encoders_path=args.encoders_path,
        metrics_path=args.metrics_path,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
