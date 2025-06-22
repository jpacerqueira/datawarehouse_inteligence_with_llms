FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create AWS credentials directory
#RUN mkdir -p /root/.aws
# Copy the aws/credentials file
#COPY aws/credentials /root/.aws/credentials
#COPY $HOME/.aws /root/.aws
#RUN chmod 600 /root/.aws/credentials
COPY aws/aws_env_vars .

# Copy requirements first to leverage Docker cache
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the rest of the application
COPY . .
COPY datamap/ .

# Create necessary directories
RUN mkdir -p /app/config /app/duckdb_data

# Expose port for the API
EXPOSE 8000

# Set environment variables
#DATAMAP_CONFIG_PATH=/app/config/datamap_config.json

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Command to run the API
CMD ["python", "-m", "uvicorn", "cashflow_datamap_api:app", "--host", "0.0.0.0", "--port", "8000"] 