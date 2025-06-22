#!/bin/bash

# Set AWS region
export AWS_DEFAULT_REGION=us-east-1

# First try to get credentials from environment variables
if [ -f "/app/aws_env_vars" ]; then
    echo "Loading AWS credentials from aws_env_vars file..."
    source /app/aws_env_vars
fi

# If credentials are not set, try to get them from AWS CLI
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "AWS credentials not found in environment, checking AWS CLI config..."
    if [ -f "$HOME/.aws/credentials" ]; then
        echo "Found AWS credentials file, attempting to get session token..."
        aws sts get-session-token > /tmp/aws_credentials.json
        
        # Extract credentials from the response
        AWS_ACCESS_KEY_ID=$(jq -r '.Credentials.AccessKeyId' /tmp/aws_credentials.json)
        AWS_SECRET_ACCESS_KEY=$(jq -r '.Credentials.SecretAccessKey' /tmp/aws_credentials.json)
        AWS_SESSION_TOKEN=$(jq -r '.Credentials.SessionToken' /tmp/aws_credentials.json)
        
        # Export the credentials
        export AWS_ACCESS_KEY_ID
        export AWS_SECRET_ACCESS_KEY
        export AWS_SESSION_TOKEN
    else
        echo "Error: No AWS credentials found in environment or AWS CLI config"
        exit 1
    fi
fi

# Verify credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS credentials are not properly set"
    exit 1
fi

# Print the credentials (for debugging)
echo "Using AWS credentials:"
echo "AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID"
echo "AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY"
echo "AWS_SESSION_TOKEN: $AWS_SESSION_TOKEN"

# Run the Streamlit application
echo "Starting Streamlit application..."
streamlit run cashflow_datamap_schema_analyzer.py --server.port=8501 --server.address=0.0.0.0 