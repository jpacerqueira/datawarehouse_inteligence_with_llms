"""
AWS Bedrock Client for Claude Integration

This module provides a client for interacting with AWS Bedrock's Claude model.
It handles authentication, request formatting, and response parsing.
"""
import json
import os
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

class BedrockClient:
    """
    Client for interacting with AWS Bedrock's Claude model.
    """
    
    def __init__(self, 
                 model_id: str = "anthropic.claude-v2", 
                 region_name: str = None,
                 max_tokens: int = 1000,
                 temperature: float = 0.7):
        """
        Initialize the Bedrock client.
        
        Args:
            model_id (str): The model ID to use for inference
            region_name (str): AWS region name (defaults to AWS_REGION env var)
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling (0.0 to 1.0)
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Get region from environment variable if not provided
        self.region_name = region_name
        if not self.region_name:
            self.region_name = os.environ.get('AWS_REGION')
            
        # Initialize the Bedrock runtime client
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name if self.region_name else None
        )
    
    def invoke_model(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the model with a prompt.
        
        Args:
            prompt (str): The user prompt
            system_prompt (Optional[str]): Optional system prompt for Claude
            
        Returns:
            Dict[str, Any]: The model response
        """
        try:
            # For Claude v2, we need to use the older prompt format
            formatted_prompt = prompt
            
            # Add system prompt if provided (as part of the prompt for Claude v2)
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
            
            # Prepare the request body for Claude v2
            request_body = {
                "prompt": formatted_prompt,
                "max_tokens_to_sample": self.max_tokens,
                "temperature": self.temperature,
                "stop_sequences": ["\n\nHuman:"]
            }
            
            # Invoke the model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response.get('body').read())
            return response_body
            
        except ClientError as e:
            print(f"Error invoking Bedrock model: {e}")
            raise
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the content from the model response.
        
        Args:
            response (Dict[str, Any]): The model response
            
        Returns:
            str: The extracted content
        """
        # For Claude v2, the response is in the 'completion' field
        return response.get("completion", "")
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query to determine intent and extract parameters.
        
        Args:
            query (str): The user query
            
        Returns:
            Dict[str, Any]: Analysis results containing intent, time period, etc.
        """
        # System prompt instructing Claude how to analyze the query
        system_prompt = """
        You are a financial analysis assistant that helps analyze queries about cash flow.
        
        For the given user query, extract the following information in JSON format:
        
        1. Intent: What is the user trying to do? (e.g., forecast, analyze_inflow, analyze_outflow, etc.)
        2. Time period: Extract any time period information (month, year, start_date, end_date)
        3. Tables: Which tables should be queried (e.g., payments)
        4. Template: Which SQL template should be used (e.g., net_cash_flow.sql.j2, inflow.sql.j2, outflow.sql.j2, analyze_cashflow.sql.j2)
        5. Parameters: Any additional parameters needed for the query
        
        Return ONLY the JSON object with these fields, nothing else.
        """
        
        # Invoke the model with the query and system prompt
        response = self.invoke_model(query, system_prompt)
        
        # Extract the content from the response
        content = self.extract_content(response)
        
        # Extract the JSON from the content
        # Claude sometimes adds explanatory text before/after the JSON
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        
        # If no JSON was found, return an error
        return {"intent": "unknown", "error": "Could not extract JSON from response"}
