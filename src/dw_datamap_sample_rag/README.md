# Bedrock RAG Metadata Schema Analyzer

This application provides a RAG-based schema metadata and data analysis system for S£/parquet files using AWS Bedrock and FAISS.

## Prerequisites

- Docker installed on your system
- Make installed on your system
- AWS credentials configured (for Bedrock access)
- DuckDB database file

## Quick Start

### Using Make

The project includes a Makefile for common operations:

1. Show available commands:
```bash
make help
```

2. Build the Docker image:
```bash
make build
```

3. Run the containers (requires AWS credentials) , either or both :
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_SESSION_TOKEN=your_session_token
make run_all
OR 
make run_api ; make run_streamlit
```

4. Clean up Docker resources:
```bash
make clean
```

### Manual Docker Commands

If you prefer not to use Make:

1. Build the Docker image:
```bash
docker build -t bedrock-rag-metadata .
```

2. Run the streamlit APP container:
```bash
make run_streamlit
```

## Development

### Using Make

1. Set up development environment:
```bash
make dev
```

2. Run tests:
```bash
make test
```

3. Run linting:
```bash
make lint
```

### Manual Development Setup

For local development without Docker:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Run the application:
```bash
streamlit run metadata_analyzer.py
```

## Environment Variables

The following environment variables can be set when running the container:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_SESSION_TOKEN`: Your AWS session token (if using temporary credentials)
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)

## Project Structure

- `bedrock_rag_metadata.py`: Core RAG implementation
- `metadata_analyzer.py`: Streamlit interface
- `requirements.txt`: Production Python dependencies
- `requirements-dev.txt`: Development Python dependencies
- `Dockerfile`: Container configuration
- `.dockerignore`: Files to exclude from Docker build
- `Makefile`: Build and development commands
- `duckdb_data/`: Directory for DuckDB database files

## Testing and Quality Assurance

The project includes:
- Unit tests (run with `make test`)
- Linting (run with `make lint`)
- Type checking with mypy
- Code formatting with black and isort

## License

This project is licensed under the MIT License - see the LICENSE file for details. 