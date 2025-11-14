"""
Integration Tests for Model Artifacts

Validates that trained models and artifacts are correctly saved and loadable.

Author: Equipo ADX
Date: 2025-11-13
"""

import sys
from pathlib import Path
import pytest
import torch
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src/pipelines/3-training'))

from model import EstablishmentDNN, LocationTimeFilter


def test_model_file_exists():
    """Test that model file was created."""
    model_path = Path('models/dnn_model.pth')
    assert model_path.exists(), "Model file not found"
    assert model_path.stat().st_size > 0, "Model file is empty"


def test_encoders_file_exists():
    """Test that encoders file was created."""
    encoders_path = Path('models/label_encoders.pkl')
    assert encoders_path.exists(), "Encoders file not found"
    assert encoders_path.stat().st_size > 0, "Encoders file is empty"


def test_feature_columns_file_exists():
    """Test that feature columns file was created."""
    feature_cols_path = Path('models/feature_columns.pkl')
    assert feature_cols_path.exists(), "Feature columns file not found"
    assert feature_cols_path.stat().st_size > 0, "Feature columns file is empty"


def test_location_filter_file_exists():
    """Test that location filter file was created."""
    filter_path = Path('models/location_filter.pkl')
    assert filter_path.exists(), "Location filter file not found"
    assert filter_path.stat().st_size > 0, "Location filter file is empty"


def test_model_loads_successfully():
    """Test that model can be loaded."""
    model_path = Path('models/dnn_model.pth')

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        assert 'model_state_dict' in checkpoint, "Model state dict not in checkpoint"
        assert 'config' in checkpoint, "Config not in checkpoint"
        print(f"Model loaded successfully. Keys: {checkpoint.keys()}")
    except Exception as e:
        pytest.fail(f"Failed to load model: {str(e)}")


def test_encoders_load_successfully():
    """Test that encoders can be loaded."""
    encoders_path = Path('models/label_encoders.pkl')

    try:
        encoders = joblib.load(encoders_path)
        assert 'establishment_encoder' in encoders, "Establishment encoder not found"
        assert hasattr(encoders['establishment_encoder'], 'classes_'), "Invalid encoder"

        num_classes = len(encoders['establishment_encoder'].classes_)
        print(f"Encoders loaded successfully. Classes: {num_classes}")
        assert num_classes > 0, "No classes in encoder"
    except Exception as e:
        pytest.fail(f"Failed to load encoders: {str(e)}")


def test_feature_columns_load_successfully():
    """Test that feature columns can be loaded."""
    feature_cols_path = Path('models/feature_columns.pkl')

    try:
        feature_cols = joblib.load(feature_cols_path)
        assert isinstance(feature_cols, list), "Feature columns should be a list"
        assert len(feature_cols) > 0, "No feature columns found"
        print(f"Feature columns loaded successfully. Count: {len(feature_cols)}")
    except Exception as e:
        pytest.fail(f"Failed to load feature columns: {str(e)}")


def test_location_filter_loads_successfully():
    """Test that location filter can be loaded."""
    filter_path = Path('models/location_filter.pkl')

    try:
        location_filter = joblib.load(filter_path)
        assert isinstance(location_filter, LocationTimeFilter), "Invalid filter type"
        assert hasattr(location_filter, 'establishment_locations'), "Missing establishment_locations"

        num_establishments = len(location_filter.establishment_locations)
        print(f"Location filter loaded successfully. Establishments: {num_establishments}")
        assert num_establishments > 0, "No establishments in filter"
    except Exception as e:
        pytest.fail(f"Failed to load location filter: {str(e)}")


def test_model_architecture_matches():
    """Test that model architecture matches expected dimensions."""
    model_path = Path('models/dnn_model.pth')
    encoders_path = Path('models/label_encoders.pkl')
    feature_cols_path = Path('models/feature_columns.pkl')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    encoders = joblib.load(encoders_path)
    feature_cols = joblib.load(feature_cols_path)

    input_dim = len(feature_cols)
    num_establishments = len(encoders['establishment_encoder'].classes_)

    # Create model with same architecture
    model = EstablishmentDNN(
        input_dim=input_dim,
        num_establishments=num_establishments
    )

    # Try to load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model architecture matches: input={input_dim}, output={num_establishments}")
    except Exception as e:
        pytest.fail(f"Model architecture mismatch: {str(e)}")


def test_model_inference_works():
    """Test that model can make predictions."""
    model_path = Path('models/dnn_model.pth')
    encoders_path = Path('models/label_encoders.pkl')
    feature_cols_path = Path('models/feature_columns.pkl')

    # Load artifacts
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    encoders = joblib.load(encoders_path)
    feature_cols = joblib.load(feature_cols_path)

    input_dim = len(feature_cols)
    num_establishments = len(encoders['establishment_encoder'].classes_)

    # Create and load model
    model = EstablishmentDNN(
        input_dim=input_dim,
        num_establishments=num_establishments
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, input_dim)

    # Make prediction
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, num_establishments), f"Output shape mismatch: {output.shape}"
    print(f"Inference successful. Output shape: {output.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
