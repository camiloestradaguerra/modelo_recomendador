.PHONY: help install test test-coverage clean setup-env run-pipeline run-sampling run-features run-training run-evaluation run-registration run-api lint format

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

help: ## Show this help message
	@echo "$(BLUE)RecSys V3 - MLOps Pipeline$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup-env: ## Create virtual environment with UV
	@echo "$(BLUE)Creating virtual environment with UV...$(NC)"
	uv venv --python python3.11
	@echo "$(GREEN)✓ Virtual environment created!$(NC)"

install: ## Install dependencies with UV
	@echo "$(BLUE)Installing dependencies with UV...$(NC)"
	uv pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed!$(NC)"

test: ## Run all unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest src/pipelines/1-data_sampling/test_sampling.py -v
	@echo "$(GREEN)✓ Tests completed!$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term --cov-report=xml src/pipelines/1-data_sampling/test_sampling.py || true
	@echo "$(GREEN)✓ Coverage report generated!$(NC)"

run-sampling: ## Run data sampling component
	@echo "$(BLUE)Running data sampling...$(NC)"
	python src/pipelines/1-data_sampling/main.py \
		--input_path data/01-raw/df_extendida_clean.parquet \
		--output_path data/02-sampled/sampled_data.parquet \
		--sample_size 2000 \
		--random_state 42
	@echo "$(GREEN)✓ Sampling completed!$(NC)"

run-features: run-sampling ## Run feature engineering component
	@echo "$(BLUE)Running feature engineering...$(NC)"
	python src/pipelines/2-feature_engineering/main.py \
		--input_path data/02-sampled/sampled_data.parquet \
		--output_path data/03-features/features.parquet \
		--encoders_path models/label_encoders.pkl
	@echo "$(GREEN)✓ Feature engineering completed!$(NC)"

run-training: run-features ## Run model training
	@echo "$(BLUE)Running model training...$(NC)"
	python src/pipelines/3-training/main.py \
		--input_path data/03-features/features.parquet \
		--model_path models/dnn_model.pth \
		--encoders_path models/label_encoders.pkl \
		--config_path src/pipelines/3-training/config.yaml
	@echo "$(GREEN)✓ Training completed!$(NC)"

run-evaluation: run-training ## Run model evaluation
	@echo "$(BLUE)Running model evaluation...$(NC)"
	python src/pipelines/4-evaluation/main.py \
		--model_path models/dnn_model.pth \
		--data_path data/03-features/features.parquet \
		--encoders_path models/label_encoders.pkl \
		--output_path reports/metrics.json
	@echo "$(GREEN)✓ Evaluation completed!$(NC)"

run-registration: run-evaluation ## Register model in MLflow
	@echo "$(BLUE)Registering model...$(NC)"
	python src/pipelines/5-model_registration/main.py \
		--model_path models/dnn_model.pth \
		--encoders_path models/label_encoders.pkl \
		--metrics_path reports/metrics.json \
		--model_name recsys_dnn
	@echo "$(GREEN)✓ Model registered!$(NC)"

run-pipeline: ## Run complete pipeline
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Running Complete MLOps Pipeline$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@make run-sampling
	@make run-features
	@make run-training
	@make run-evaluation
	@make run-registration
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Pipeline Execution Completed!$(NC)"
	@echo "$(GREEN)========================================$(NC)"

run-api: ## Start FastAPI server
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
	uvicorn entrypoint.main:app --reload --host 0.0.0.0 --port 8001

run-dashboard: ## Start monitoring dashboard
	@echo "$(BLUE)Starting Streamlit dashboard...$(NC)"
	streamlit run dashboard/app.py

test-api: ## Run API integration tests
	@echo "$(BLUE)Testing API endpoints...$(NC)"
	python tests/test_api.py

clean: ## Clean generated files
	@echo "$(BLUE)Cleaning files...$(NC)"
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f *.bak
	@echo "$(GREEN)Cleanup completed$(NC)"

quickstart: setup-env install ## Quick setup
	@echo "$(GREEN)Quick start completed$(NC)"

