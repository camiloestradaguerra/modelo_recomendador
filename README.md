# RecSys V3 - Restaurant Recommendation System

Production-ready MLOps pipeline for personalized restaurant recommendations using Deep Neural Networks with location and time filtering.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Model Details](#model-details)
- [API Usage](#api-usage)
- [Testing](#testing)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Monitoring](#monitoring)
- [AWS Deployment](#aws-deployment)
- [Development Notes](#development-notes)
- [Troubleshooting](#troubleshooting)

## Overview

End-to-end MLOps pipeline that implements a Deep Neural Network recommendation system with critical location and time filtering capabilities. The system processes transaction data, engineers features, trains a model, and serves recommendations via FastAPI.

### Critical Fix Implemented

The original model had a fundamental flaw: it could recommend establishments outside the user's city or that were closed. This has been fixed with the `LocationTimeFilter` class (src/pipelines/3-training/model.py:296-361) that enforces:

- **Geographic constraints**: Only shows restaurants in the user's ciudad
- **Temporal constraints**: Only shows restaurants open at the requested hora

## Key Features

- Deep Neural Network (1024→256→256) with batch normalization and dropout
- 53 engineered features (temporal, user, location, interaction)
- Location and time-aware recommendations
- MLflow experiment tracking and model registry
- FastAPI REST API with Pydantic validation
- Streamlit monitoring dashboard with Evidently drift detection
- GitHub Actions CI/CD pipeline
- Comprehensive unit and integration tests
- AWS SageMaker deployment ready

## Quick Start

### Prerequisites

- Python 3.11
- Input data: `data/01-raw/df_extendida_clean.parquet`
- 8GB RAM minimum

### Installation

```bash
# Clone repository
git clone https://github.com/carlosjimenez88M/recsys_v3.git
cd recsys_v3

# Setup environment
make setup-env
source .venv/bin/activate

# Install dependencies
make install
```

### Run Complete Pipeline

```bash
# Execute all 6 steps: sampling → features → validation → training → evaluation → registration
make run-pipeline
```

### Start API Server

```bash
make run-api
# Visit http://localhost:8000/docs for interactive API documentation
```

### Start Monitoring Dashboard

```bash
streamlit run dashboard/app.py
# Visit http://localhost:8501
```

## Project Structure

```
recsys_v3/
├── .github/workflows/          # CI/CD pipelines
│   └── ml-pipeline.yml         # GitHub Actions workflow (8 jobs)
├── data/
│   ├── 01-raw/                 # Raw transaction data (7.5M rows)
│   ├── 02-sampled/             # Sampled data (2000 for dev/test)
│   └── 03-features/            # Engineered features (53 columns)
├── src/pipelines/
│   ├── 1-data_sampling/        # Sample records from full dataset
│   ├── 2-feature_engineering/  # Engineer 53 features
│   ├── data_validation/        # Validate data quality
│   ├── 3-training/             # Train DNN model
│   ├── 4-evaluation/           # Evaluate with NDCG metrics
│   ├── 5-model_registration/   # Register in MLflow
│   └── main_pipeline.py        # Pipeline orchestrator
├── entrypoint/                 # FastAPI application
│   ├── main.py                 # FastAPI app
│   ├── schemas.py              # Pydantic models
│   └── routers/
│       └── recommendations.py  # Recommendation endpoint
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── tests/
│   ├── test_sampling.py        # Unit tests
│   └── test_api.py             # API integration tests
├── models/                     # Trained models (not in git)
├── reports/                    # Metrics and validation reports
├── Makefile                    # Build automation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Pipeline Components

### 1. Data Sampling

Samples records from the full dataset for efficient development and testing.

```bash
make run-sampling
# Input:  data/01-raw/df_extendida_clean.parquet (7,527,130 rows)
# Output: data/02-sampled/sampled_data.parquet (2,000 rows)
```

**Important**: The 2000-sample size is configured for **development and testing only**. For production:
- Use the full dataset or larger samples
- Adjust `sample_size` in `src/pipelines/main_pipeline.py:219` or config YAML
- Consider stratified sampling for better representation

### 2. Feature Engineering

Engineers 53 features from raw transaction data.

```bash
make run-features
```

**Features Created:**
- **Temporal** (13): hour, day_of_week, month, cyclical encodings (sin/cos), time bands
- **User** (8): age, gender, marital status, membership tenure, spending patterns (log, squared)
- **Location** (4): ciudad, zona, region, hour-ciudad interaction
- **Interaction** (7): user-establishment frequency, specialty preference, popularity
- **Encoded** (21): label-encoded categorical variables

**Output**: `data/03-features/features.parquet`, `models/label_encoders.pkl`

### 3. Data Validation

Validates data quality and generates comprehensive HTML reports.

```bash
python src/pipelines/data_validation/main.py \
  --input_path data/03-features/features.parquet \
  --output_path reports/data_validation.html
```

**Checks Performed:**
- Missing value analysis
- Distribution statistics (mean, std, skewness, kurtosis)
- Outlier detection (IQR method)
- Correlation matrix
- Visual reports with Plotly

### 4. Model Training

Trains Deep Neural Network with optimized hyperparameters.

```bash
make run-training
```

**Architecture:**
- Input: 53 features
- Hidden Layer 1: 1024 neurons + BatchNorm + ReLU + Dropout
- Hidden Layer 2: 256 neurons + BatchNorm + ReLU + Dropout
- Hidden Layer 3: 256 neurons + BatchNorm + ReLU + Dropout
- Output: num_establishments (softmax)

**Best Hyperparameters** (from Bayesian Optimization):
```yaml
batch_size: 32
learning_rate: 0.0001967641848109
weight_decay: 0.00008261871088
hidden_dim1: 1024
hidden_dim2: 256
hidden_dim3: 256
dropout_rate: 0.1429465700244763
epochs: 50
```

**Training Features:**
- Xavier weight initialization
- Cross-entropy loss
- AdamW optimizer
- Early stopping
- MLflow experiment tracking
- Location/time filter creation

### 5. Model Evaluation

Evaluates model performance using ranking metrics.

```bash
make run-evaluation
```

**Metrics:**
- Accuracy (top-1)
- NDCG@5 (Normalized Discounted Cumulative Gain)
- NDCG@10
- Per-class performance analysis

### 6. Model Registration

Registers model in MLflow with comprehensive metadata.

```bash
make run-registration
```

**Registered Artifacts:**
- Model weights (dnn_model.pth)
- Label encoders (label_encoders.pkl)
- Feature columns (feature_columns.pkl)
- Location filter (location_filter.pkl)
- Evaluation metrics (metrics.json)

## Model Details

### LocationTimeFilter

Critical component for production deployment (src/pipelines/3-training/model.py:226-395).

**How it works:**
1. Takes user's ciudad and hora as input
2. Creates mask of valid establishments (correct city + open hours)
3. Zeros out probabilities for invalid establishments
4. Re-normalizes with softmax
5. Returns filtered top-k recommendations

**Example:**
```python
filtered_probs = location_filter.apply(
    predictions=model_output,      # Raw model predictions
    user_ciudad="Quito",            # User's city
    hora=14,                        # Time: 2 PM
    establishment_names=all_names   # List of all establishments
)
# Returns only Quito restaurants open at 2 PM
```

### Performance Expectations

**On 2000-sample dataset (development):**
- Accuracy: 70-80%
- NDCG@5: 0.75-0.85
- NDCG@10: 0.80-0.90
- Training time: 10-30 minutes (CPU)

**Production (full dataset, 7.5M rows):**
- Performance metrics may vary
- Requires GPU for training (recommended: ml.p3.2xlarge on AWS)
- Consider incremental training for model updates

## API Usage

### Start Server

```bash
make run-api
# Server runs on http://localhost:8000
```

### Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommendations/" \
  -H "Content-Type: application/json" \
  -d '{
    "id_persona": 21096.0,
    "ciudad": "Quito",
    "hora": 14,
    "k": 5
  }'
```

### Response Format

```json
{
  "recommendations": [
    {
      "establecimiento": "RESTAURANT A",
      "probability": 0.85,
      "ciudad": "Quito"
    },
    {
      "establecimiento": "RESTAURANT B",
      "probability": 0.72,
      "ciudad": "Quito"
    }
  ],
  "filtered_by_location": true,
  "filtered_by_time": true
}
```

### API Documentation

Interactive documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

### Unit Tests

```bash
# Run unit tests
make test

# Run with coverage report
make test-coverage
```

**Tests Included:**
- Data sampling validation
- Feature engineering correctness
- Model architecture verification
- LocationTimeFilter logic

### API Integration Tests

```bash
# Start API server first
make run-api

# In another terminal, run API tests
python tests/test_api.py
```

**Test Coverage:**
- Health check endpoint
- Recommendations for different cities (Quito, Guayaquil, Cuenca)
- Time-based filtering (morning vs evening)
- Location filtering verification
- Concurrent request handling
- Real database references

## CI/CD with GitHub Actions

### Workflow Overview

File: `.github/workflows/ml-pipeline.yml`

**8 Jobs (individualized with dependencies):**

```
Setup & Unit Tests (Job 0)
    ↓
Data Sampling (Job 1/6)
    ↓
Feature Engineering (Job 2/6)
    ↓
Data Validation (Job 3/6)
    ↓
Model Training (Job 4/6) - with Docker MLflow
    ↓
Model Evaluation (Job 5/6)
    ↓
Model Registration (Job 6/6) - only on tags v*.*.*
    ↓
Cleanup (Job 7) - always runs
```

### Triggers

- Push to `master` or `main` branches
- Pull requests to `master` or `main`
- Version tags (e.g., `v1.0.0`)
- Manual workflow dispatch

### Features

- Uses `actions/setup-python@v5` with pip caching
- Docker MLflow server for training with health checks
- Artifact retention: 1 day (intermediate), 30 days (reports), 90 days (production)
- Automatic cleanup: keeps last 10 runs, removes old artifacts
- Comprehensive AWS deployment examples in comments

### Create Release

```bash
# Tag version
git tag v1.0.0
git push origin v1.0.0

# This triggers model registration job
```

## Monitoring

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

**Three Pages:**

1. **Model Performance**
   - Real-time metrics (Accuracy, NDCG@5, NDCG@10)
   - Training history visualization
   - Feature importance charts
   - Loss curves

2. **Data Drift Detection**
   - Statistical drift tests using Evidently
   - Feature distribution comparisons
   - Automatic HTML report generation
   - Alerts when retraining needed

3. **System Health**
   - API status monitoring
   - Response time tracking
   - Recent predictions log
   - Error rate monitoring

## AWS Deployment

The pipeline is ready for AWS SageMaker deployment. Comprehensive examples are provided in `.github/workflows/ml-pipeline.yml` (commented sections starting line 472).

### Key Components

1. **Processing Jobs** (ml.t3.medium)
   - Data sampling, feature engineering, validation, evaluation

2. **Training Job** (ml.p3.2xlarge + Spot Instances)
   - PyTorch 2.1.0 container, GPU-accelerated, ~70% cost savings with spot

3. **Model Registry**
   - Versioned models with Git metadata, approval workflow, A/B testing support

4. **Endpoint Deployment** (ml.m5.large)
   - Real-time inference, auto-scaling (1-10 instances), CloudWatch monitoring

5. **Lambda Alternative** (Serverless, Lower Cost)
   - Event-driven, auto-scaling, pay-per-request pricing

### Deployment Example

```bash
# Tag and push
git tag v1.0.0
git push origin v1.0.0

# GitHub Actions automatically:
# 1. Runs tests
# 2. Executes SageMaker Pipeline
# 3. Registers model with metadata
# 4. Deploys to production endpoint (if approved)
```

## Development Notes

### Sample Size Configuration

**IMPORTANT**: The default configuration uses **2000 samples** for development and testing purposes only.

**For Development/Testing:**
```python
# src/pipelines/main_pipeline.py:219
'sample_size': 2000  # Fast iteration, 10-30 min training
```

**For Production:**
```python
# Option 1: Larger sample
'sample_size': 100000  # Representative sample

# Option 2: Full dataset
'sample_size': None  # Use all 7.5M rows (requires GPU)
```

**Considerations for production:**
- GPU strongly recommended (ml.p3.2xlarge on AWS)
- Stratified sampling for better class balance
- Incremental training for model updates
- Monitor data drift for retraining triggers

### Docstring Style

All functions use Hadley Wickham-style docstrings with clear parameter descriptions, types, return values, and usage examples.

### Code Quality

- No emojis in code or documentation
- Pydantic validation for API requests/responses
- Type hints throughout
- Professional appearance

## Troubleshooting

### Module not found errors

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

### CUDA out of memory

Edit `src/pipelines/3-training/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### API cannot load model

```bash
# Ensure all required model files exist
ls -la models/
# Required: dnn_model.pth, label_encoders.pkl, feature_columns.pkl, location_filter.pkl
```

### Training takes too long

```bash
# Use smaller sample for rapid development
# Edit src/pipelines/main_pipeline.py:219
'sample_size': 500  # Even faster for testing pipeline logic
```

### GitHub Actions failing

1. Check workflow logs in GitHub Actions tab
2. Verify all paths use `src/pipelines/` structure
3. Ensure data file exists in `data/01-raw/`
4. Check Python version is 3.11 in workflow

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following code style guidelines
4. Run tests (`make test`)
5. Commit with clear, descriptive messages
6. Push to branch
7. Open Pull Request

## License

This project is licensed under the MIT License.

## Author

Equipo ADX  
Date: 2025-11-13

## Acknowledgments

- MLOps best practices from School of DevOps
- Model architecture based on collaborative filtering research
- Feature engineering techniques from recommendation system literature
- Location filtering inspired by real production deployment challenges

---

**Important Note**: This project uses a **2000-record sample by default for development and testing**. For production deployment with the full dataset (7.5M rows), update the `sample_size` configuration in `src/pipelines/main_pipeline.py` and provision appropriate compute resources (GPU recommended for training).
