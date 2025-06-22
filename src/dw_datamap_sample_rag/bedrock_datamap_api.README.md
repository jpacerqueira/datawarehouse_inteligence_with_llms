# Cashflow DataMap Metadata Analyzer and API

A tool for analyzing and querying database metadata using AWS Bedrock and RAG (Retrieval-Augmented Generation).

## Features

- Natural language querying of database schema
- Similarity-based metadata retrieval
- Table-level metadata analysis
- REST API interface
- Streamlit web interface

## API Documentation

The API provides the following endpoints:

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Root
- **GET /** - Returns API status
- **Response**: `{"message": "Cashflow DataMap Metadata API"}`

#### 2. Schema Analysis
- **POST /analyze** - Analyze database schema based on natural language query
- **Request Body**:
  ```json
  {
    "query": "What are the main tables in the database?",
    "k": 3
  }
  ```
- **Response**: Analysis results with query context

#### 3. Similar Metadata
- **POST /similar** - Get similar metadata entries
- **Request Body**:
  ```json
  {
    "query": "customer data",
    "k": 5
  }
  ```
- **Response**: Similar metadata entries with similarity scores

#### 4. Table Details
- **POST /table** - Get detailed information about a specific table
- **Request Body**:
  ```json
  {
    "table_name": "customers"
  }
  ```
- **Response**: Table metadata including columns and sample data

#### 5. Schema Summary
- **GET /schema** - Get summary of entire database schema
- **Response**: Summary of all tables with column counts and names

### Interactive API Documentation
Access the Swagger UI at:
```
http://localhost:8000/docs
```

Access the ReDoc documentation at:
```
http://localhost:8000/redoc
```

## Setup

### Prerequisites
- Python 3.9+
- AWS credentials
- Docker (optional)

### Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_SESSION_TOKEN=your_session_token  # Optional
export DATAMAP_CONFIG_PATH=config/datamap_config.json
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/project_dw_intel.git
cd project_dw_intel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running with Docker

1. Build the Docker image:
```bash
docker build -t cashflow-datamap .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_SESSION_TOKEN=your_session_token \
  cashflow-datamap
```

### Running Locally

1. Start the API server:
```bash
python -m datamap.cashflow_datamap_api
```

2. Start the Streamlit app:
```bash
streamlit run datamap/cashflow_datamap_schema_analyzer.py
```

## Configuration

The API uses a JSON configuration file (`config/datamap_config.json`) with the following structure to map parquet files in s3 buckets with filename patterns:

```json
{
    "bucket_name": "project-dw_intel",
    "prefix": "sbca/batch3/1299438/bronze/",
    "region_name": "us-east-1",
    "cache_size": 128,
    "pattern": ".*parquet*"
}
```

## License

MIT License
