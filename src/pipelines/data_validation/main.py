"""
Data Validation Component

Validates data quality and generates distribution reports.
Checks for:
- Missing values
- Data types
- Distribution statistics
- Outliers
- Feature correlations

Author: Equipo ADX
Date: 2025-11-13
"""

import argparse
import sys
import io
from pathlib import Path
from typing import Dict
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import S3DataManager from sibling module
sys.path.insert(0, str(Path(__file__).parent.parent / "0-cleaning_data"))
from main import S3DataManager

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)


def validate_missing_values(df: pd.DataFrame) -> Dict:
    """
    Check for missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.

    Returns
    -------
    Dict
        Dictionary with missing value statistics.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'column': missing.index,
        'missing_count': missing.values,
        'missing_percentage': missing_pct.values
    })

    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
        'missing_percentage', ascending=False
    )

    logger.info(f"Found {len(missing_df)} columns with missing values")
    return missing_df.to_dict('records')


def validate_data_types(df: pd.DataFrame) -> Dict:
    """
    Check data types of all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.

    Returns
    -------
    Dict
        Dictionary with data type information.
    """
    dtypes_df = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.values,
        'unique_values': [df[col].nunique() for col in df.columns],
        'sample_values': [str(df[col].head(3).tolist()) for col in df.columns]
    })

    logger.info(f"Dataset has {len(dtypes_df)} columns")
    return dtypes_df.to_dict('records')


def validate_distributions(df: pd.DataFrame) -> Dict:
    """
    Analyze distributions of numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.

    Returns
    -------
    Dict
        Dictionary with distribution statistics.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'q25': float(df[col].quantile(0.25)),
            'median': float(df[col].median()),
            'q75': float(df[col].quantile(0.75)),
            'max': float(df[col].max()),
            'skewness': float(df[col].skew()),
            'kurtosis': float(df[col].kurtosis())
        }

    logger.info(f"Analyzed distributions for {len(numeric_cols)} numeric features")
    return stats


def detect_outliers(df: pd.DataFrame) -> Dict:
    """
    Detect outliers using IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.

    Returns
    -------
    Dict
        Dictionary with outlier statistics.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outlier_count / len(df)) * 100

        if outlier_count > 0:
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': float(outlier_pct),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }

    logger.info(f"Found outliers in {len(outliers)} columns")
    return outliers


def generate_report_html(
    df: pd.DataFrame,
    missing: Dict,
    dtypes: Dict,
    distributions: Dict,
    outliers: Dict
) -> str:
    """
    Generate HTML validation report content.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    missing : Dict
        Missing values statistics.
    dtypes : Dict
        Data types information.
    distributions : Dict
        Distribution statistics.
    outliers : Dict
        Outlier information.

    Returns
    -------
    str
        HTML report content.
    """
    # Create plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Feature Distributions', 'Missing Values',
                       'Outlier Percentages', 'Correlation Matrix'),
        specs=[[{'type': 'histogram'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'heatmap'}]]
    )

    # Distribution plot for first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        first_col = numeric_cols[0]
        fig.add_trace(
            go.Histogram(x=df[first_col], name=first_col),
            row=1, col=1
        )

    # Missing values plot
    if missing:
        missing_df = pd.DataFrame(missing)
        fig.add_trace(
            go.Bar(x=missing_df['column'], y=missing_df['missing_percentage'],
                   name='Missing %'),
            row=1, col=2
        )

    # Outliers plot
    if outliers:
        outlier_cols = list(outliers.keys())
        outlier_pcts = [outliers[col]['percentage'] for col in outlier_cols]
        fig.add_trace(
            go.Bar(x=outlier_cols, y=outlier_pcts, name='Outlier %'),
            row=2, col=1
        )

    # Correlation matrix
    if len(numeric_cols) > 1:
        corr = df[numeric_cols[:10]].corr()  # Limit to first 10 for visibility
        fig.add_trace(
            go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                      colorscale='RdBu', zmid=0),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=False, title_text="Data Validation Report")

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .warning {{ color: #d9534f; }}
            .success {{ color: #5cb85c; }}
        </style>
    </head>
    <body>
        <h1>Data Validation Report</h1>

        <div class="summary">
            <h2>Dataset Summary</h2>
            <p><strong>Total Rows:</strong> {len(df):,}</p>
            <p><strong>Total Columns:</strong> {len(df.columns)}</p>
            <p><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>
            <p class="{'warning' if missing else 'success'}">
                <strong>Missing Values:</strong> {len(missing) if missing else 0} columns affected
            </p>
            <p class="{'warning' if outliers else 'success'}">
                <strong>Outliers:</strong> Detected in {len(outliers) if outliers else 0} columns
            </p>
        </div>

        <h2>Visualizations</h2>
        {fig.to_html(full_html=False, include_plotlyjs='cdn')}

        <h2>Data Types</h2>
        <table>
            <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Unique Values</th>
                <th>Sample</th>
            </tr>
            {''.join([f"<tr><td>{dt['column']}</td><td>{dt['dtype']}</td><td>{dt['unique_values']}</td><td>{dt['sample_values']}</td></tr>" for dt in dtypes])}
        </table>

        {'<h2>Missing Values</h2><table><tr><th>Column</th><th>Count</th><th>Percentage</th></tr>' + ''.join([f"<tr><td>{m['column']}</td><td>{m['missing_count']}</td><td>{m['missing_percentage']:.2f}%</td></tr>" for m in missing]) + '</table>' if missing else '<h2>Missing Values</h2><p class="success">No missing values found!</p>'}

        <h2>Distribution Statistics</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Median</th>
                <th>Max</th>
                <th>Skewness</th>
            </tr>
            {''.join([f"<tr><td>{col}</td><td>{stats['mean']:.2f}</td><td>{stats['std']:.2f}</td><td>{stats['min']:.2f}</td><td>{stats['median']:.2f}</td><td>{stats['max']:.2f}</td><td>{stats['skewness']:.2f}</td></tr>" for col, stats in distributions.items()])}
        </table>

        {'<h2>Outliers</h2><table><tr><th>Column</th><th>Count</th><th>Percentage</th><th>Bounds</th></tr>' + ''.join([f"<tr><td>{col}</td><td>{info['count']}</td><td>{info['percentage']:.2f}%</td><td>[{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]</td></tr>" for col, info in outliers.items()]) + '</table>' if outliers else '<h2>Outliers</h2><p class="success">No significant outliers detected!</p>'}

    </body>
    </html>
    """
    return html_content


def generate_report(
    df: pd.DataFrame,
    missing: Dict,
    dtypes: Dict,
    distributions: Dict,
    outliers: Dict,
    output_path: str
) -> None:
    """
    Generate and save HTML validation report to local path.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    missing : Dict
        Missing values statistics.
    dtypes : Dict
        Data types information.
    distributions : Dict
        Distribution statistics.
    outliers : Dict
        Outlier information.
    output_path : str
        Path to save HTML report (local path only).
    """
    html_content = generate_report_html(df, missing, dtypes, distributions, outliers)
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.success(f"Validation report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate data quality and generate report")
    parser.add_argument("--input_path", type=str, required=True, help="Input S3 folder containing df_features")
    parser.add_argument("--output_path", type=str, required=True, help="Output report path (local or S3)")
    args = parser.parse_args()

    # Detect if paths are S3
    is_input_s3 = args.input_path.startswith("s3://")
    is_output_s3 = args.output_path.startswith("s3://")

    # --------------------------
    # LOAD INPUT FROM S3
    # --------------------------
    if is_input_s3:
        s3_manager = S3DataManager()

        # Parse s3://bucket/prefix/
        s3_path = args.input_path.replace("s3://", "")
        parts = s3_path.split("/", 1)

        bucket_input = parts[0]
        prefix_input = parts[1] if len(parts) > 1 else ""

        logger.info(f"Bucket detectado: {bucket_input}")
        logger.info(f"Prefix detectado: {prefix_input}")

        # Buscar archivo que empiece por df_features
        logger.info(f"Buscando archivo más reciente en s3://{bucket_input}/{prefix_input} con inicio 'df_features'")
        
        newest_file_path = s3_manager.get_newest_file_by_date(
            bucket_name=bucket_input,
            prefix=prefix_input,
            starts_with="df_features"
        )

        if not newest_file_path:
            raise FileNotFoundError(
                f"No se encontró ningún archivo que inicie con 'df_features' en s3://{bucket_input}/{prefix_input}"
            )

        logger.info(f"Archivo encontrado: {newest_file_path}")

        df = pd.read_parquet(
            newest_file_path,
            storage_options={"key": s3_manager.aws_access_key,
                             "secret": s3_manager.aws_secret_key}
        )

    else:
        # Carga local
        logger.info(f"Loading data from local path: {args.input_path}")
        df = pd.read_parquet(args.input_path)

    logger.info(f"Loaded {len(df)} records and {len(df.columns)} columns")

    # --------------------------
    # VALIDACIONES
    # --------------------------
    missing = validate_missing_values(df)
    dtypes = validate_data_types(df)
    distributions = validate_distributions(df)
    outliers = detect_outliers(df)

    # --------------------------
    # GUARDAR REPORTE
    # --------------------------
    html_content = generate_report_html(df, missing, dtypes, distributions, outliers)

    if is_output_s3:
        s3_manager = S3DataManager()

        out_path = args.output_path.replace("s3://", "")
        parts = out_path.split("/", 1)

        bucket_output = parts[0]
        prefix_output = parts[1] if len(parts) > 1 else ""

        if prefix_output and not prefix_output.endswith("/"):
            prefix_output += "/"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{timestamp}.html"

        destino = f"s3://{bucket_output}/{prefix_output}{filename}"
        logger.info(f"Guardando reporte en: {destino}")

        try:
            with s3_manager.fs.open(destino, "wb") as f:
                f.write(html_content.encode("utf-8"))
            logger.success(f"Reporte guardado correctamente en {destino}")

        except Exception as e:
            logger.error(f"Error guardando reporte en S3: {e}")
            raise

    else:
        # LOCAL
        generate_report(df, missing, dtypes, distributions, outliers, args.output_path)
        logger.success(f"Reporte guardado en {args.output_path}")

    logger.success("Data validation completed!")



if __name__ == "__main__":
    main()
