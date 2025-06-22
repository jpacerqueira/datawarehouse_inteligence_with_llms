# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    unzip \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create AWS credentials directory
#RUN mkdir -p /root/.aws
# Copy the aws/credentials file
#COPY aws/credentials /root/.aws/credentials
#COPY $HOMR/.aws /root/.aws
#RUN chmod 600 /root/.aws/credentials
COPY aws/aws_env_vars .

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .
COPY datamap/ .

# Make the start script executable
RUN chmod +x start.sh

# Create directory for DuckDB data
RUN mkdir -p duckdb_data

# Set environment variables
#ENV AWS_DEFAULT_REGION=us-east-1

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["./start.sh"] 