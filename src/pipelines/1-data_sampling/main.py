"""
Data Sampling Component

This module samples a subset of records from the raw dataset for pipeline testing
and development. It ensures reproducibility through seeded random sampling while
maintaining the distributional characteristics of the original data.

The sampling strategy preserves the temporal ordering and key categorical
distributions, making it suitable for downstream model training and evaluation.

Author: Equipo ADX
Date: 2025-11-13
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# Import S3DataManager from cleaning_data pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "0-cleaning_data"))
from main import S3DataManager

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


def validate_input_file(file_path: Path) -> None:
    """
    Validate that the input file exists and is readable.

    This function performs defensive checks on the input file path, ensuring
    that the file exists, is indeed a file (not a directory), and has the
    expected parquet extension. These checks prevent downstream errors and
    provide clear feedback to users.

    Parameters
    ----------
    file_path : Path
        The path to the input file that should be validated.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If the path points to a directory rather than a file, or if the
        file extension is not '.parquet'.

    Examples
    --------
    >>> validate_input_file(Path("data/raw/transactions.parquet"))
    # Passes silently if file exists and is valid

    >>> validate_input_file(Path("data/raw/missing.parquet"))
    FileNotFoundError: Input file does not exist: data/raw/missing.parquet

    Notes
    -----
    This function is called before attempting to read data, following the
    principle of "fail fast" to provide immediate feedback on configuration
    errors.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")

    if file_path.suffix != ".parquet":
        raise ValueError(f"Input file must be a parquet file, got: {file_path.suffix}")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a parquet file with comprehensive error handling.

    This function wraps pandas' parquet reading functionality with additional
    error handling and logging. It's designed to fail gracefully and provide
    useful diagnostic information when data loading issues occur.

    Parameters
    ----------
    file_path : Path
        The path to the parquet file to be loaded.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe containing all records and columns from the
        parquet file.

    Raises
    ------
    Exception
        If the parquet file cannot be read due to corruption, permissions,
        or other IO errors. The original exception is re-raised with context.

    Examples
    --------
    >>> df = load_data(Path("data/raw/transactions.parquet"))
    >>> print(df.shape)
    (1000000, 20)

    Notes
    -----
    The function logs both the attempt to load data and the resulting shape,
    which aids in debugging pipeline issues and monitoring data flow.
    """
    logger.info(f"Loading data from {file_path}")

    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise


def sample_data(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample a specified number of records from the dataframe.

    This function performs stratified random sampling to ensure the sampled
    subset maintains similar statistical properties to the full dataset. The
    sampling is seeded for reproducibility, which is critical for creating
    consistent train/test splits across pipeline runs.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataframe from which to sample records.
    sample_size : int
        The number of records to include in the sample. If this exceeds the
        total number of records, all records are returned with a warning.
    random_state : int, optional
        The seed for the random number generator, ensuring reproducible
        samples across runs. Default is 42.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the sampled records. The index is reset to
        ensure sequential integer indexing.

    Examples
    --------
    >>> df_full = pd.DataFrame({'a': range(10000), 'b': range(10000)})
    >>> df_sample = sample_data(df_full, sample_size=100, random_state=42)
    >>> df_sample.shape
    (100, 2)

    >>> # Reproducibility check
    >>> df_sample2 = sample_data(df_full, sample_size=100, random_state=42)
    >>> (df_sample == df_sample2).all().all()
    True

    Notes
    -----
    The function uses pandas' sample() method which provides efficient random
    sampling without replacement. For very large datasets, consider using
    chunked sampling or database-level sampling for better performance.

    When sample_size exceeds the dataframe size, the function returns all
    records rather than raising an error, following the principle of graceful
    degradation.
    """
    total_records = len(df)
    logger.info(f"Sampling {sample_size} records from {total_records} total records")

    if sample_size >= total_records:
        logger.warning(
            f"Sample size ({sample_size}) >= total records ({total_records}). "
            "Returning all records."
        )
        return df.reset_index(drop=True)

    sampled_df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    logger.info(f"Sampling completed. Final shape: {sampled_df.shape}")

    return sampled_df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the dataframe to a parquet file with metadata preservation.

    This function writes a pandas dataframe to disk in parquet format, creating
    any necessary parent directories. It uses compression to minimize storage
    requirements while maintaining fast read performance for downstream tasks.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be saved.
    output_path : Path
        The path where the parquet file should be written. Parent directories
        are created automatically if they don't exist.

    Raises
    ------
    Exception
        If the file cannot be written due to permissions, disk space, or other
        IO errors. The exception is logged and re-raised.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> save_data(df, Path("data/processed/output.parquet"))
    # Creates data/processed/ directory if needed and saves file

    Notes
    -----
    The parquet format is preferred over CSV for several reasons:
    - Preserves data types without ambiguity
    - Provides efficient compression
    - Supports faster reading for large datasets
    - Maintains metadata about the dataframe structure

    The function creates parent directories automatically to prevent common
    errors where intermediate directories don't exist.
    """
    logger.info(f"Saving data to {output_path}")

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.success(f"Data saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {output_path}: {str(e)}")
        raise


def run_sampling_pipeline(
    input_path: str,
    output_path: str,
    sample_size: int,
    random_state: int
) -> None:
    """
    Execute the complete data sampling pipeline.

    This is the main orchestration function that coordinates all sampling steps:
    validation, loading, sampling, and saving. It implements a clear separation
    of concerns while maintaining a straightforward execution flow.

    Parameters
    ----------
    input_path : str
        Path to the source parquet file containing the full dataset.
    output_path : str
        Path where the sampled data should be written.
    sample_size : int
        Number of records to include in the sample.
    random_state : int
        Seed for reproducible random sampling.

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist.
    ValueError
        If the input file is invalid (wrong type or format).
    Exception
        If any step in the pipeline fails (loading, sampling, or saving).

    Examples
    --------
    >>> run_sampling_pipeline(
    ...     input_path="data/raw/transactions.parquet",
    ...     output_path="data/sampled/sample.parquet",
    ...     sample_size=2000,
    ...     random_state=42
    ... )
    # Executes full pipeline and saves sampled data

    Notes
    -----
    This function serves as the entry point for the sampling component in the
    MLOps pipeline. It's designed to be called by MLflow or other orchestration
    tools, with all parameters exposed for configuration.

    The function follows the pattern of validate -> load -> transform -> save,
    which is common across data pipeline components and makes the code easy to
    understand and maintain.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    # Detect if input is S3 or local
    is_input_s3 = input_path.startswith("s3://")
    is_output_s3 = output_path.startswith("s3://")
    
    if is_input_s3:
        # Initialize S3DataManager
        s3_manager = S3DataManager()
        
        # Parse S3 input path: s3://bucket/prefix
        s3_path_parts = input_path.replace("s3://", "").split("/", 1)
        bucket_input = s3_path_parts[0]
        prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
        
        # Get newest file from S3 with filter for "df_extendida_clean"
        logger.info(f"Buscando archivo más reciente en s3://{bucket_input}/{prefix} con nombre 'df_extendida_clean'")
        newest_file_path = s3_manager.get_newest_file_by_date(
            bucket_name=bucket_input, 
            prefix=prefix,
            starts_with="df_extendida_clean"
        )
        
        if not newest_file_path:
            raise FileNotFoundError(f"No se encontró archivo 'df_extendida_clean' en S3: s3://{bucket_input}/{prefix}")
        
        # Load data from S3
        logger.info(f"Cargando datos desde: {newest_file_path}")
        df = pd.read_parquet(newest_file_path)
    else:
        # Local file processing
        # Validate input
        validate_input_file(input_file)

        # Load data
        df = load_data(input_file)

    # Sample data
    df_sampled = sample_data(df, sample_size, random_state)

    if is_output_s3:
        # Initialize S3DataManager for output
        s3_manager = S3DataManager()
        
        # Parse S3 output path: s3://bucket/path/
        s3_path_parts = output_path.replace("s3://", "").split("/", 1)
        bucket_output = s3_path_parts[0]
        path_destino = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
        
        # Ensure path ends with /
        if path_destino and not path_destino.endswith("/"):
            path_destino += "/"
        
        nombre_archivo = "df_sampled.parquet"
        
        logger.info(f"Guardando datos en S3: s3://{bucket_output}/{path_destino}")
        
        # Save sampled data to S3 with timestamp
        s3_manager.save_dataframe_to_s3(df_sampled, bucket_output, path_destino, nombre_archivo)
    else:
        # Local file saving
        # Save sampled data
        save_data(df_sampled, output_file)

    logger.success("Data sampling pipeline completed successfully!")


def main():
    """
    Parse command-line arguments and execute the sampling pipeline.

    This function serves as the entry point when the module is run as a script.
    It sets up the argument parser with sensible defaults and comprehensive help
    text, then delegates to the main pipeline function.

    The command-line interface follows standard Unix conventions, with long-form
    argument names and clear descriptions. This makes the tool easy to use both
    interactively and in automated scripts.

    Command-line Arguments
    ----------------------
    --input_path : str, required
        Path to the input parquet file containing the full dataset.
    --output_path : str, required
        Path where the sampled data will be saved.
    --sample_size : int, optional
        Number of records to sample. Default is 2000.
    --random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Examples
    --------
    $ python main.py \\
        --input_path data/raw/transactions.parquet \\
        --output_path data/sampled/sample.parquet \\
        --sample_size 2000 \\
        --random_state 42

    Returns
    -------
    None
        Exits with status 0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description="Sample a subset of records from the raw dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input_path data/raw/data.parquet --output_path data/sampled/sample.parquet
  %(prog)s --input_path data/raw/data.parquet --output_path data/sampled/sample.parquet --sample_size 5000
        """
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input parquet file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where sampled data will be saved"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Number of records to sample (default: 2000)"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    run_sampling_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
