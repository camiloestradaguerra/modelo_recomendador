"""
Main MLOps Pipeline Orchestrator

Executes the complete recommendation system pipeline sequentially.

Steps:
1. Data Sampling
2. Feature Engineering
3. Data Validation
4. Model Training
5. Model Evaluation
6. Model Registration

Author: Equipo ADX
Date: 2025-11-13
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import importlib.util

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)


def load_module(module_path):
    """Load a module dynamically from path."""
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sampling(input_path, output_path, sample_size, random_state):
    """Execute data sampling step."""
    logger.info("Step 1/6: Data Sampling")

    module_path = Path(__file__).parent / '1-data_sampling' / 'main.py'
    sampling = load_module(module_path)

    sys.argv = [
        'sampling',
        '--input_path', input_path,
        '--output_path', output_path,
        '--sample_size', str(sample_size),
        '--random_state', str(random_state)
    ]

    sampling.main()
    logger.success("Sampling completed")


def run_feature_engineering(input_path, output_path, encoders_path):
    """Execute feature engineering step."""
    logger.info("Step 2/6: Feature Engineering")

    module_path = Path(__file__).parent / '2-feature_engineering' / 'main.py'
    features = load_module(module_path)

    sys.argv = [
        'features',
        '--input_path', input_path,
        '--output_path', output_path,
        '--encoders_path', encoders_path
    ]

    features.main()
    logger.success("Feature engineering completed")


def run_data_validation(input_path, output_path):
    """Execute data validation step."""
    logger.info("Step 3/6: Data Validation")

    module_path = Path(__file__).parent / 'data_validation' / 'main.py'
    validation = load_module(module_path)

    sys.argv = [
        'validation',
        '--input_path', input_path,
        '--output_path', output_path
    ]

    validation.main()
    logger.success("Data validation completed")


def run_training(input_path, model_path, encoders_path, config_path):
    """Execute model training step."""
    logger.info("Step 4/6: Model Training")

    module_path = Path(__file__).parent / '3-training' / 'main.py'
    training = load_module(module_path)

    sys.argv = [
        'training',
        '--input_path', input_path,
        '--model_path', model_path,
        '--encoders_path', encoders_path,
        '--config_path', config_path
    ]

    training.main()
    logger.success("Training completed")


def run_evaluation(model_path, data_path, encoders_path, output_path):
    """Execute model evaluation step."""
    logger.info("Step 5/6: Model Evaluation")

    module_path = Path(__file__).parent / '4-evaluation' / 'main.py'
    evaluation = load_module(module_path)

    sys.argv = [
        'evaluation',
        '--model_path', model_path,
        '--data_path', data_path,
        '--encoders_path', encoders_path,
        '--output_path', output_path
    ]

    evaluation.main()
    logger.success("Evaluation completed")


def run_registration(model_path, encoders_path, metrics_path, model_name):
    """Execute model registration step."""
    logger.info("Step 6/6: Model Registration")

    module_path = Path(__file__).parent / '5-model_registration' / 'main.py'
    registration = load_module(module_path)

    sys.argv = [
        'registration',
        '--model_path', model_path,
        '--encoders_path', encoders_path,
        '--metrics_path', metrics_path,
        '--model_name', model_name
    ]

    registration.main()
    logger.success("Registration completed")


def run_pipeline(config):
    """Execute complete pipeline."""
    logger.info("Starting MLOps Pipeline")
    logger.info("="*50)

    try:
        # Step 1: Sampling
        run_sampling(
            input_path=config['sampling']['input_path'],
            output_path=config['sampling']['output_path'],
            sample_size=config['sampling']['sample_size'],
            random_state=config['sampling']['random_state']
        )

        # Step 2: Feature Engineering
        run_feature_engineering(
            input_path=config['features']['input_path'],
            output_path=config['features']['output_path'],
            encoders_path=config['features']['encoders_path']
        )

        # Step 3: Data Validation
        run_data_validation(
            input_path=config['validation']['input_path'],
            output_path=config['validation']['output_path']
        )

        # Step 4: Training
        run_training(
            input_path=config['training']['input_path'],
            model_path=config['training']['model_path'],
            encoders_path=config['training']['encoders_path'],
            config_path=config['training']['config_path']
        )

        # Step 5: Evaluation
        run_evaluation(
            model_path=config['evaluation']['model_path'],
            data_path=config['evaluation']['data_path'],
            encoders_path=config['evaluation']['encoders_path'],
            output_path=config['evaluation']['output_path']
        )

        # Step 6: Registration
        run_registration(
            model_path=config['registration']['model_path'],
            encoders_path=config['registration']['encoders_path'],
            metrics_path=config['registration']['metrics_path'],
            model_name=config['registration']['model_name']
        )

        logger.success("="*50)
        logger.success("Pipeline completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False


def get_default_config():
    """Get default pipeline configuration."""
    return {
        'sampling': {
            'input_path': 'data/01-raw/df_extendida_clean.parquet',
            'output_path': 'data/02-sampled/sampled_data.parquet',
            'sample_size': 2000,
            'random_state': 42
        },
        'features': {
            'input_path': 'data/02-sampled/sampled_data.parquet',
            'output_path': 'data/03-features/features.parquet',
            'encoders_path': 'models/label_encoders.pkl'
        },
        'validation': {
            'input_path': 'data/03-features/features.parquet',
            'output_path': 'reports/data_validation.html'
        },
        'training': {
            'input_path': 'data/03-features/features.parquet',
            'model_path': 'models/dnn_model.pth',
            'encoders_path': 'models/label_encoders.pkl',
            'config_path': 'src/pipelines/3-training/config.yaml'
        },
        'evaluation': {
            'model_path': 'models/dnn_model.pth',
            'data_path': 'data/03-features/features.parquet',
            'encoders_path': 'models/label_encoders.pkl',
            'output_path': 'reports/metrics.json'
        },
        'registration': {
            'model_path': 'models/dnn_model.pth',
            'encoders_path': 'models/label_encoders.pkl',
            'metrics_path': 'reports/metrics.json',
            'model_name': 'recsys_dnn'
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run complete MLOps pipeline")
    parser.add_argument("--config", type=str, help="Path to pipeline config file")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()

    success = run_pipeline(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
