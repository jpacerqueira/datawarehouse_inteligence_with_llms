.PHONY: build run_streamlit run_api dev clean test lint help


# Variables
VENV = venv
AWS_ENV_FILE = aws/aws_env_vars
DOCKER_TAG = latest


# Streamlit app
PORT_STREAMLIT = 8501
DOCKER_FILE_STREAMLIT = streamlit_app.Dockerfile
DOCKER_IMAGE_STREAMLIT = bedrock-datamap-rag-streamlit
DOCKER_CONTAINER_STREAMLIT = bedrock-datamap-rag-streamlit-container


# API app
PORT_API = 8000
DOCKER_FILE_API = bedrock_datamap_api.Dockerfile
DOCKER_IMAGE_API = bedrock-datamap-rag-api
DOCKER_CONTAINER_API = bedrock-datamap-rag-api-container



help:
	@echo "Available commands:"
	@echo "  make build         - Build the Docker image"
	@echo "  make run_all       - Run the Docker containers for the Streamlit app and the API"
	@echo "  make run_streamlit - Run the Docker container for the Streamlit app"
	@echo "  make run_api       - Run the Docker container for the API"
	@echo "  make dev           - Set up development environment"
	@echo "  make clean         - Clean up Docker resources"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make help          - Show this help message"

build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE_STREAMLIT):$(DOCKER_TAG) -f $(DOCKER_FILE_STREAMLIT) .
	docker build -t $(DOCKER_IMAGE_API):$(DOCKER_TAG) -f $(DOCKER_FILE_API) .

# Load AWS credentials from file
AWS_ACCESS_KEY_ID := $(shell grep AWS_ACCESS_KEY_ID $(AWS_ENV_FILE) | cut -d'"' -f2)
AWS_SECRET_ACCESS_KEY := $(shell grep AWS_SECRET_ACCESS_KEY $(AWS_ENV_FILE) | cut -d'"' -f2)
AWS_SESSION_TOKEN := $(shell grep AWS_SESSION_TOKEN $(AWS_ENV_FILE) | cut -d'"' -f2)

run_all:
	@echo "Running all Docker containers..."
	@if [ ! -f "$(AWS_ENV_FILE)" ]; then \
		echo "Error: $(AWS_ENV_FILE) not found"; \
		exit 1; \
	fi
	@if [ -z "$(AWS_ACCESS_KEY_ID)" ] || [ -z "$(AWS_SECRET_ACCESS_KEY)" ]; then \
		echo "Error: AWS credentials not found in $(AWS_ENV_FILE)"; \
		exit 1; \
	fi
	docker compose -f docker-compose.yml down 2>/dev/null || true
	docker compose -f docker-compose.yml up --build || true
	@echo "All Docker containers have been started successfully."

run_streamlit:
	@echo "Running Docker container with AWS credentials..."
	@if [ ! -f "$(AWS_ENV_FILE)" ]; then \
		echo "Error: $(AWS_ENV_FILE) not found"; \
		exit 1; \
	fi
	@if [ -z "$(AWS_ACCESS_KEY_ID)" ] || [ -z "$(AWS_SECRET_ACCESS_KEY)" ]; then \
		echo "Error: AWS credentials not found in $(AWS_ENV_FILE)"; \
		exit 1; \
	fi
	docker compose -f docker-compose-rag.yml down 2>/dev/null || true
	docker compose -f docker-compose-rag.yml up --build || true
	@echo "All RAG Streamlit Docker containers have been started successfully."

run_api:
	@echo "Running Docker container with AWS credentials..."
	@if [ ! -f "$(AWS_ENV_FILE)" ]; then \
		echo "Error: $(AWS_ENV_FILE) not found"; \
		exit 1; \
	fi
	@if [ -z "$(AWS_ACCESS_KEY_ID)" ] || [ -z "$(AWS_SECRET_ACCESS_KEY)" ]; then \
		echo "Error: AWS credentials not found in $(AWS_ENV_FILE)"; \
		exit 1; \
	fi
	docker compose -f docker-compose-api.yml down 2>/dev/null || true
	docker compose -f docker-compose-api.yml up --build || true
	@echo "All API Docker containers have been started successfully."

dev:
	@echo "Setting up development environment..."
	python -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt
	. $(VENV)/bin/activate && pip install -r requirements-dev.txt

clean:
	@echo "Cleaning up Docker resources..."
	docker stop $$(docker ps -q --filter ancestor=$(DOCKER_IMAGE):$(DOCKER_TAG)) 2>/dev/null || true
	docker rm $$(docker ps -a -q --filter ancestor=$(DOCKER_IMAGE):$(DOCKER_TAG)) 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true

test:
	@echo "Running tests..."
	. $(VENV)/bin/activate && python -m pytest tests/

lint:
	@echo "Running linting..."
	. $(VENV)/bin/activate && flake8 . 
