"""
Feature Engineering Component

This module transforms raw transactional data into engineered features suitable
for machine learning models. It creates temporal patterns, user characteristics,
location-based features, and interaction metrics that capture user behavior.

The feature engineering process is critical for recommendation quality, as it
extracts latent patterns from historical transactions and encodes them in a
format that neural networks can learn from effectively.

Author: Equipo ADX
Date: 2025-11-13
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple


import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


class FeatureEngineer:
    """
    Feature engineering pipeline for recommendation system.

    This class orchestrates the creation of multiple feature types from raw
    transactional data. It maintains internal state (label encoders) that must
    be saved and reused during inference to ensure consistent encoding.

    The feature engineering follows these principles:
    - Temporal features capture when users make purchases
    - User features encode demographics and spending patterns
    - Interaction features quantify user-item relationships
    - Location features enable geographic recommendations

    Attributes
    ----------
    label_encoders : dict
        Mapping from categorical feature names to fitted LabelEncoder objects.
        These must be persisted for inference.
    establishment_encoder : LabelEncoder
        Special encoder for the target variable (establishments). Kept separate
        to enable easy inverse transformation during prediction.

    Examples
    --------
    >>> engineer = FeatureEngineer()
    >>> df_transformed = engineer.fit_transform(df_raw)
    >>> engineer.save_encoders('models/encoders.pkl')

    Notes
    -----
    The class follows sklearn's transformer interface with fit_transform() and
    transform() methods, making it compatible with scikit-learn pipelines if
    needed in the future.
    """

    def __init__(self):
        """Initialize feature engineer with empty encoder dictionaries."""
        self.label_encoders = {}
        self.establishment_encoder = LabelEncoder()

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features capturing temporal purchase patterns.

        Temporal features are crucial for recommendations because user behavior
        varies dramatically by time of day, day of week, and season. For example,
        breakfast restaurants are preferred in morning hours, while bars are
        popular at night.

        This method creates both raw temporal features (hour, day, month) and
        cyclical encodings using sine/cosine transforms. The cyclical encoding
        prevents artificial boundaries at the edges of time periods (e.g., 23:59
        and 00:00 should be close together, not far apart).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'HORA_INICIO' column containing transaction
            timestamps.

        Returns
        -------
        pd.DataFrame
            Dataframe with added temporal features:
            - hora: hour of day (0-23)
            - dia_semana: day of week (0-6, Monday=0)
            - mes: month (1-12)
            - franja_horaria: categorical time period
            - hora_sin, hora_cos: cyclical hour encoding
            - dia_sin, dia_cos: cyclical day encoding
            - mes_sin, mes_cos: cyclical month encoding

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'HORA_INICIO': pd.date_range('2025-01-01', periods=24, freq='H')
        ... })
        >>> df_temporal = engineer.create_temporal_features(df)
        >>> df_temporal[['hora', 'franja_horaria']].head()

        Notes
        -----
        The cyclical encoding uses the formula:
            sin_feature = sin(2π * feature / period)
            cos_feature = cos(2π * feature / period)

        This ensures that:
        - Hour 23 and hour 0 are encoded as similar
        - Day 6 (Sunday) and day 0 (Monday) are close
        - December and January are adjacent months
        """
        df = df.copy()

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Extract basic temporal components
        df['hora_inicio'] = pd.to_datetime(df['hora_inicio'])
        df['hora'] = df['hora_inicio'].dt.hour
        df['dia_semana'] = df['hora_inicio'].dt.dayofweek
        df['mes'] = df['hora_inicio'].dt.month

        # Create categorical time bands
        df['franja_horaria'] = pd.cut(
            df['hora'],
            bins=[0, 6, 12, 18, 24],
            labels=['Madrugada', 'Manana', 'Tarde', 'Noche'],
            include_lowest=True
        )

        # Cyclical encoding for hour (0-23)
        df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)

        # Cyclical encoding for day of week (0-6)
        df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)

        # Cyclical encoding for month (1-12)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

        logger.info("Temporal features created: hora, dia_semana, mes, cyclical encodings")
        return df

    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-specific features capturing demographics and spending.

        User features encode individual characteristics that influence purchase
        decisions. These include both demographic attributes (age, gender) and
        behavioral metrics (spending patterns, membership tenure).

        The spending features use logarithmic transforms and polynomial terms to
        capture non-linear relationships between spending and purchase probability.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with user attributes and transaction amounts.

        Returns
        -------
        pd.DataFrame
            Dataframe with added user features:
            - log_monto: log-transformed transaction amount
            - monto_squared: squared transaction amount
            - edad_grupo: age group category
            - antiguedad_normalizada: normalized membership tenure

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'monto': [10.5, 50.0, 100.0],
        ...     'edad': [25, 35, 45],
        ...     'antiguedad_socio_unico': [1.0, 5.0, 10.0]
        ... })
        >>> df_user = engineer.create_user_features(df)

        Notes
        -----
        The log transform log1p(x) = log(1 + x) is used instead of log(x) to
        handle zero values gracefully. This is important for transactions with
        very small or zero amounts.

        Age groups are created to capture generational differences in spending
        patterns while reducing the feature space compared to raw age values.
        """
        df = df.copy()

        # Log-transform spending to reduce skewness
        df['log_monto'] = np.log1p(df['monto'])

        # Polynomial term captures non-linear spending effects
        df['monto_squared'] = df['monto'] ** 2

        # Age groupings based on life stages
        df['edad_grupo'] = pd.cut(
            df['edad'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Joven', 'Adulto_Joven', 'Adulto', 'Maduro', 'Senior']
        )

        # Normalize membership tenure to [0, 1] range
        max_antiguedad = df['antiguedad_socio_unico'].max()
        if max_antiguedad > 0:
            df['antiguedad_normalizada'] = df['antiguedad_socio_unico'] / max_antiguedad
        else:
            df['antiguedad_normalizada'] = 0.0

        logger.info("User features created: spending patterns, age groups, tenure")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features capturing user-item and user-location interactions.

        Interaction features are the most powerful for recommendation systems
        because they directly encode user preferences. They answer questions like:
        - How often does this user visit this establishment?
        - Does this user prefer this type of cuisine?
        - How frequently does this user visit this city?

        These features enable the model to learn collaborative filtering patterns,
        where users with similar interaction histories receive similar recommendations.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with user IDs, establishments, specialties, and cities.

        Returns
        -------
        pd.DataFrame
            Dataframe with added interaction features:
            - user_establishment_freq: how often user visits each establishment
            - user_specialty_freq: how often user visits each cuisine type
            - user_city_freq: how often user visits each city
            - establishment_popularity: overall popularity of each establishment
            - specialty_popularity: overall popularity of each cuisine
            - city_popularity: overall popularity of each city
            - hora_ciudad_interaction: temporal-spatial interaction term

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'id_persona': [1, 1, 2, 2],
        ...     'establecimiento': ['A', 'A', 'B', 'C'],
        ...     'especialidad': ['Pizza', 'Pizza', 'Burger', 'Sushi'],
        ...     'ciudad': ['Quito', 'Quito', 'Guayaquil', 'Cuenca']
        ... })
        >>> df_interact = engineer.create_interaction_features(df)

        Notes
        -----
        Interaction features are computed using groupby aggregations, which can
        be memory-intensive for large datasets. For production systems with
        millions of users, consider using sparse matrix representations or
        incremental computation.

        The hora_ciudad interaction captures that certain hours are popular in
        certain cities (e.g., lunch hour in business districts vs. residential
        areas).
        """
        df = df.copy()

        # User-establishment frequency (collaborative filtering signal)
        user_est_counts = df.groupby(['id_persona', 'establecimiento']).size().to_dict()
        df['user_establishment_freq'] = df.apply(
            lambda row: user_est_counts.get((row['id_persona'], row['establecimiento']), 0),
            axis=1
        )

        # User-specialty frequency (content-based signal)
        user_spec_counts = df.groupby(['id_persona', 'especialidad']).size().to_dict()
        df['user_specialty_freq'] = df.apply(
            lambda row: user_spec_counts.get((row['id_persona'], row['especialidad']), 0),
            axis=1
        )

        # User-city frequency (geographic preference)
        user_city_counts = df.groupby(['id_persona', 'ciudad']).size().to_dict()
        df['user_city_freq'] = df.apply(
            lambda row: user_city_counts.get((row['id_persona'], row['ciudad']), 0),
            axis=1
        )

        # Global popularity features (wisdom of the crowd)
        est_counts = df['establecimiento'].value_counts().to_dict()
        df['establishment_popularity'] = df['establecimiento'].map(est_counts)

        spec_counts = df['especialidad'].value_counts().to_dict()
        df['specialty_popularity'] = df['especialidad'].map(spec_counts)

        city_counts = df['ciudad'].value_counts().to_dict()
        df['city_popularity'] = df['ciudad'].map(city_counts)

        # Spatial-temporal interaction (hour × city encoding)
        df['hora_ciudad_interaction'] = df['hora'] * df['ciudad'].astype('category').cat.codes

        logger.info("Interaction features created: user preferences, popularity, spatial-temporal")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoders and transform the dataframe in one step.

        This method should be called only once during training. It fits label
        encoders on the categorical features and applies all transformations.
        The fitted encoders are stored internally and must be saved for inference.

        Parameters
        ----------
        df : pd.DataFrame
            Raw training data with all required columns.

        Returns
        -------
        pd.DataFrame
            Fully transformed dataframe ready for model training, with:
            - All engineered features
            - Encoded categorical variables
            - Encoded target (establecimiento_encoded)

        Examples
        --------
        >>> engineer = FeatureEngineer()
        >>> df_train = engineer.fit_transform(df_raw)
        >>> engineer.save_encoders('encoders.pkl')

        Notes
        -----
        The method normalizes column names to lowercase to handle inconsistent
        naming in raw data. All original columns are preserved alongside the
        engineered features.

        After calling this method, you must save the encoders using save_encoders()
        to enable consistent encoding during inference.
        """
        logger.info("Starting feature engineering pipeline...")

        # Normalize column names
        df.columns = df.columns.str.lower()

        # Create feature groups
        df = self.create_temporal_features(df)
        df = self.create_user_features(df)
        df = self.create_interaction_features(df)

        # Encode categorical features
        categorical_features = [
            'especialidad', 'estado_civil', 'genero', 'rol',
            'segmento_comercial', 'ciudad', 'zona', 'region',
            'cadena', 'franja_horaria', 'edad_grupo'
        ]

        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
                logger.info(f"Encoded {feature}: {len(le.classes_)} unique values")

        # Encode target variable (establishment)
        if 'establecimiento' in df.columns:
            df['establecimiento_encoded'] = self.establishment_encoder.fit_transform(
                df['establecimiento']
            )
            logger.info(f"Encoded target: {len(self.establishment_encoder.classes_)} establishments")

        logger.success(f"Feature engineering completed. Shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using previously fitted encoders.

        This method should be used during inference or validation. It applies
        the same transformations as fit_transform() but uses the already-fitted
        encoders instead of fitting new ones.

        Parameters
        ----------
        df : pd.DataFrame
            New data to transform (e.g., validation set, test set, or inference
            data).

        Returns
        -------
        pd.DataFrame
            Transformed dataframe with the same features as training data.

        Examples
        --------
        >>> engineer = FeatureEngineer.load_encoders('encoders.pkl')
        >>> df_test_transformed = engineer.transform(df_test)

        Notes
        -----
        If a categorical value appears in the new data but was not seen during
        training, it's encoded as 0 (unknown). This graceful handling prevents
        errors during inference with novel categories.

        The method assumes the same column structure as the training data. Missing
        columns will cause errors, so ensure consistent data schemas.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Apply same transformations
        df = self.create_temporal_features(df)
        df = self.create_user_features(df)
        df = self.create_interaction_features(df)

        # Apply fitted encoders
        for feature, le in self.label_encoders.items():
            if feature in df.columns:
                df[f'{feature}_encoded'] = df[feature].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
                )

        if 'establecimiento' in df.columns:
            df['establecimiento_encoded'] = df['establecimiento'].apply(
                lambda x: self.establishment_encoder.transform([x])[0]
                if x in self.establishment_encoder.classes_ else 0
            )

        return df

    def save_encoders(self, path: Path) -> None:
        """
        Save fitted encoders to disk for inference.

        Encoders must be persisted because they contain the mapping from
        categorical values to integer codes. Without these mappings, we cannot
        transform new data consistently with the training data.

        Parameters
        ----------
        path : Path
            File path where encoders will be saved (typically .pkl extension).

        Examples
        --------
        >>> engineer.save_encoders(Path('models/encoders.pkl'))

        Notes
        -----
        Uses joblib for serialization, which is efficient for sklearn objects
        and handles numpy arrays well.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        encoder_dict = {
            'label_encoders': self.label_encoders,
            'establishment_encoder': self.establishment_encoder
        }

        joblib.dump(encoder_dict, path)
        logger.success(f"Encoders saved to {path}")

    @classmethod
    def load_encoders(cls, path: Path) -> 'FeatureEngineer':
        """
        Load previously saved encoders from disk.

        This class method creates a new FeatureEngineer instance and populates
        it with the saved encoders, enabling transformation of new data.

        Parameters
        ----------
        path : Path
            File path where encoders were saved.

        Returns
        -------
        FeatureEngineer
            New instance with loaded encoders ready for transform().

        Examples
        --------
        >>> engineer = FeatureEngineer.load_encoders(Path('models/encoders.pkl'))
        >>> df_new = engineer.transform(df_inference)

        Notes
        -----
        The returned instance can only call transform(), not fit_transform(),
        because the encoders are already fitted.
        """
        engineer = cls()
        encoder_dict = joblib.load(path)

        engineer.label_encoders = encoder_dict['label_encoders']
        engineer.establishment_encoder = encoder_dict['establishment_encoder']

        logger.info(f"Encoders loaded from {path}")
        return engineer


def run_feature_engineering(
    input_path: str,
    output_path: str,
    encoders_path: str
) -> None:
    """
    Execute the feature engineering pipeline.

    This is the main entry point that orchestrates loading data, applying
    transformations, and saving results. It's designed to be called by
    MLflow or other orchestration tools.

    Parameters
    ----------
    input_path : str
        Path to the sampled input data (parquet format).
    output_path : str
        Path where transformed features will be saved (parquet format).
    encoders_path : str
        Path where label encoders will be saved (pickle format).

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist.
    Exception
        If any step in the pipeline fails.

    Examples
    --------
    >>> run_feature_engineering(
    ...     input_path='data/sampled/sample.parquet',
    ...     output_path='data/features/features.parquet',
    ...     encoders_path='models/encoders.pkl'
    ... )

    Notes
    -----
    The function automatically drops rows with missing values after feature
    engineering. In production, consider more sophisticated imputation strategies.
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} records")

    # Apply feature engineering
    engineer = FeatureEngineer()
    df_transformed = engineer.fit_transform(df)

    # Drop rows with any NaN values
    initial_rows = len(df_transformed)
    df_transformed = df_transformed.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df_transformed)

    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing values")

    # Save transformed data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_parquet(output_file, index=False)
    logger.success(f"Features saved to {output_path}")

    # Save encoders
    engineer.save_encoders(Path(encoders_path))

    logger.success("Feature engineering pipeline completed!")


def main():
    """
    Parse command-line arguments and run the pipeline.

    This function serves as the entry point when the module is run as a script.
    It defines the CLI interface and delegates to the main pipeline function.

    Command-line Arguments
    ----------------------
    --input_path : str, required
        Path to the sampled input parquet file.
    --output_path : str, required
        Path where engineered features will be saved.
    --encoders_path : str, required
        Path where label encoders will be saved.

    Examples
    --------
    $ python main.py \\
        --input_path data/sampled/sample.parquet \\
        --output_path data/features/features.parquet \\
        --encoders_path models/encoders.pkl
    """
    parser = argparse.ArgumentParser(
        description="Feature engineering for recommendation system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the sampled input data"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where engineered features will be saved"
    )

    parser.add_argument(
        "--encoders_path",
        type=str,
        required=True,
        help="Path where label encoders will be saved"
    )

    args = parser.parse_args()

    run_feature_engineering(
        input_path=args.input_path,
        output_path=args.output_path,
        encoders_path=args.encoders_path
    )


if __name__ == "__main__":
    main()
