"""
Deep Neural Network Model for Restaurant Recommendations

This module implements a multi-layer perceptron for predicting which restaurant
a user will visit, based on their characteristics, temporal context, and historical
behavior. The model outputs probabilities for each establishment.

The key innovation is the LocationTimeFilter class, which post-processes predictions
to ensure recommendations respect geographic and temporal constraints.

Author: Equipo ADX
Date: 2025-11-13
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EstablishmentDNN(nn.Module):
    """
    Deep neural network for establishment recommendation.

    This is a classification model that takes user features, temporal features,
    and interaction features as input and outputs probability distributions over
    all possible establishments.

    The architecture uses:
    - Three hidden layers with decreasing dimensions (1024 → 256 → 256)
    - Batch normalization for training stability
    - Dropout for regularization
    - ReLU activation functions

    Parameters
    ----------
    input_dim : int
        Number of input features (determined from feature engineering step).
    num_establishments : int
        Number of unique establishments to predict (output dimension).
    hidden_dim1 : int, optional
        Size of first hidden layer. Default is 1024.
    hidden_dim2 : int, optional
        Size of second hidden layer. Default is 256.
    hidden_dim3 : int, optional
        Size of third hidden layer. Default is 256.
    dropout_rate : float, optional
        Dropout probability for regularization. Default is 0.143.

    Attributes
    ----------
    network : nn.Sequential
        The neural network layers organized as a sequential module.

    Examples
    --------
    >>> model = EstablishmentDNN(
    ...     input_dim=45,
    ...     num_establishments=5566,
    ...     hidden_dim1=1024,
    ...     dropout_rate=0.143
    ... )
    >>> output = model(input_tensor)
    >>> probabilities = F.softmax(output, dim=1)

    Notes
    -----
    The model uses cross-entropy loss during training, which internally applies
    softmax. During inference, remember to apply softmax to get probabilities.

    Batch normalization is applied after each linear layer to normalize activations,
    which accelerates training and improves generalization.
    """

    def __init__(
        self,
        input_dim: int,
        num_establishments: int,
        hidden_dim1: int = 1024,
        hidden_dim2: int = 256,
        hidden_dim3: int = 256,
        dropout_rate: float = 0.1429465700244763
    ):
        """Initialize the deep neural network architecture."""
        super(EstablishmentDNN, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: Input → Hidden1
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2: Hidden1 → Hidden2
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 3: Hidden2 → Hidden3
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Output Layer: Hidden3 → Establishments
            nn.Linear(hidden_dim3, num_establishments)
        )

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.

        Xavier initialization helps prevent vanishing/exploding gradients by
        scaling weights based on the number of inputs and outputs. This is
        particularly important for deep networks.

        Notes
        -----
        Biases are initialized to zero, which is standard practice.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (batch_size, num_establishments).
            Apply softmax to get probabilities.

        Notes
        -----
        The output is not normalized (no softmax) because PyTorch's
        CrossEntropyLoss expects raw logits and applies log-softmax internally
        for numerical stability.
        """
        return self.network(x)

    def get_feature_importance(
        self,
        X_sample: np.ndarray,
        y_sample: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute feature importance using gradient-based attribution.

        This method calculates how much each input feature contributes to the
        model's predictions by computing gradients with respect to inputs.

        Parameters
        ----------
        X_sample : np.ndarray
            Sample of input features, shape (n_samples, n_features).
        y_sample : np.ndarray
            True labels for the samples, shape (n_samples,).
        feature_names : List[str]
            Names of features corresponding to columns in X_sample.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to importance scores.

        Examples
        --------
        >>> importance = model.get_feature_importance(
        ...     X_sample=X_test[:1000],
        ...     y_sample=y_test[:1000],
        ...     feature_names=feature_cols
        ... )
        >>> sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        Notes
        -----
        Feature importance is computed as the mean absolute gradient across
        all samples. Higher values indicate features that have more influence
        on predictions.

        This method requires gradients, so it should only be called when the
        model is in eval mode but with gradients enabled.
        """
        self.eval()
        device = next(self.parameters()).device

        X_tensor = torch.FloatTensor(X_sample).to(device)
        X_tensor.requires_grad = True

        y_tensor = torch.LongTensor(y_sample).to(device)

        # Forward pass
        outputs = self(X_tensor)
        loss = F.cross_entropy(outputs, y_tensor)

        # Backward pass to get gradients
        loss.backward()

        # Compute importance as mean absolute gradient
        gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()

        feature_importance = {
            name: float(importance)
            for name, importance in zip(feature_names, gradients)
        }

        return feature_importance


class LocationTimeFilter:
    """
    Post-processing filter for location and time constraints.

    This class addresses the critical issue identified in the original model:
    recommendations must respect WHERE the user is (ciudad) and WHEN they're
    making the request (hora).

    The filter works by:
    1. Taking the model's raw probability predictions
    2. Identifying which establishments are valid for the given location/time
    3. Zeroing out probabilities for invalid establishments
    4. Re-normalizing the remaining probabilities

    This ensures that a user in Quito only sees Quito restaurants, and only
    those that are open at the requested time.

    Parameters
    ----------
    establishment_locations : Dict[str, str]
        Mapping from establishment name to its ciudad.
    establishment_hours : Dict[str, Tuple[int, int]], optional
        Mapping from establishment name to (open_hour, close_hour).
        If not provided, all hours are considered valid.

    Attributes
    ----------
    establishment_locations : Dict[str, str]
        Stored location mappings.
    establishment_hours : Dict[str, Tuple[int, int]]
        Stored operating hours.

    Examples
    --------
    >>> filter = LocationTimeFilter(
    ...     establishment_locations={'Restaurant A': 'Quito', 'Restaurant B': 'Guayaquil'},
    ...     establishment_hours={'Restaurant A': (10, 22), 'Restaurant B': (11, 23)}
    ... )
    >>> filtered_probs = filter.apply(
    ...     predictions=model_output,
    ...     user_ciudad='Quito',
    ...     hora=14,
    ...     establishment_names=all_establishments
    ... )

    Notes
    -----
    This is a crucial fix for the production system. Without this filter,
    the model can recommend establishments in the wrong city or that are closed,
    leading to poor user experience.

    The filter maintains the relative ranking from the model but constrains
    the search space to valid options.

    AWS SageMaker Note
    ------------------
    When deploying to SageMaker, this filter should be part of the inference
    pipeline. The establishment metadata (locations, hours) should be stored
    in S3 or DynamoDB and loaded during endpoint initialization.
    """

    def __init__(
        self,
        establishment_locations: Dict[str, str],
        establishment_hours: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """Initialize the filter with establishment metadata."""
        self.establishment_locations = establishment_locations
        self.establishment_hours = establishment_hours or {}

    def apply(
        self,
        predictions: torch.Tensor,
        user_ciudad: str,
        hora: int,
        establishment_names: List[str]
    ) -> torch.Tensor:
        """
        Apply location and time filtering to predictions.

        Parameters
        ----------
        predictions : torch.Tensor
            Raw model predictions (logits or probabilities), shape (batch_size, num_establishments).
        user_ciudad : str
            The city where the user is located.
        hora : int
            Hour of day (0-23) when recommendation is requested.
        establishment_names : List[str]
            Names of establishments corresponding to prediction indices.

        Returns
        -------
        torch.Tensor
            Filtered and re-normalized predictions.

        Notes
        -----
        The filtering process:
        1. Create a mask of valid establishments (right city + open hours)
        2. Zero out invalid establishments
        3. Apply softmax to renormalize probabilities over valid options

        If no establishments match the criteria, all probabilities are returned
        as zero (the caller should handle this edge case).
        """
        device = predictions.device

        # Create validity mask
        valid_mask = torch.zeros(len(establishment_names), dtype=torch.bool)

        for idx, est_name in enumerate(establishment_names):
            # Check location
            est_ciudad = self.establishment_locations.get(est_name)
            if est_ciudad != user_ciudad:
                continue

            # Check operating hours (if available)
            if est_name in self.establishment_hours:
                open_hour, close_hour = self.establishment_hours[est_name]
                if not (open_hour <= hora < close_hour):
                    continue

            valid_mask[idx] = True

        # Move mask to same device as predictions
        valid_mask = valid_mask.to(device)

        # Apply mask
        masked_predictions = predictions.clone()
        masked_predictions[:, ~valid_mask] = -float('inf')  # Will become 0 after softmax

        # Re-normalize with softmax
        filtered_probs = F.softmax(masked_predictions, dim=1)

        return filtered_probs

    @classmethod
    def from_dataframe(cls, df) -> 'LocationTimeFilter':
        """
        Create filter from a pandas DataFrame containing establishment data.

        This convenience method extracts establishment metadata from the
        training data to build the filter.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'establecimiento' and 'ciudad' columns.

        Returns
        -------
        LocationTimeFilter
            Initialized filter ready to use.

        Examples
        --------
        >>> filter = LocationTimeFilter.from_dataframe(df_train)

        Notes
        -----
        This method assumes each establishment has a single location. If an
        establishment appears in multiple cities (e.g., chains), it will use
        the most common city.

        For operating hours, this simple implementation doesn't extract them
        from data. In production, you'd query a database or API.
        """
        import pandas as pd

        # Get most common ciudad for each establishment
        establishment_locations = (
            df.groupby('establecimiento')['ciudad']
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .to_dict()
        )

        # TODO: In production, fetch operating hours from database
        establishment_hours = {}

        return cls(establishment_locations, establishment_hours)


# AWS SageMaker Training Job Configuration (commented for local development)
#
# To run this on SageMaker, modify the training script as follows:
#
# 1. Add SageMaker-specific imports:
#    import sagemaker
#    from sagemaker.pytorch import PyTorch
#
# 2. Update paths to use SageMaker environment variables:
#    import os
#    input_path = os.environ.get('SM_CHANNEL_TRAIN', 'data/03-features/features.parquet')
#    model_path = os.environ.get('SM_MODEL_DIR', 'models/')
#    output_path = os.environ.get('SM_OUTPUT_DATA_DIR', 'reports/')
#
# 3. Save model artifacts to SM_MODEL_DIR:
#    torch.save(model.state_dict(), f"{model_path}/model.pth")
#    joblib.dump(filter, f"{model_path}/location_filter.pkl")
#
# 4. Use SageMaker hyperparameters:
#    import argparse
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--batch_size', type=int, default=32)
#    parser.add_argument('--learning_rate', type=float, default=0.0001967641848109)
#    args, _ = parser.parse_known_args()
#
# 5. Create SageMaker training job:
#    estimator = PyTorch(
#        entry_point='src/3-training/main.py',
#        source_dir='.',
#        role=role,
#        instance_type='ml.p3.2xlarge',
#        instance_count=1,
#        framework_version='2.1.0',
#        py_version='py311',
#        hyperparameters={
#            'batch_size': 32,
#            'learning_rate': 0.0001967641848109,
#            'epochs': 50
#        }
#    )
#    estimator.fit({'train': 's3://bucket/recsys-v3/data/03-features/'})
