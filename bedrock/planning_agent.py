"""
Planning Agent for Cash Flow POC

This module contains the logic to parse user queries and determine the appropriate
SQL template and parameters to use. It can use either rule-based parsing or
AWS Bedrock's Claude model for more advanced query understanding.
"""
import re
import os
import json
from datetime import datetime
import calendar
from typing import Dict, Any, Optional

# Import the BedrockClient
from bedrock_client import BedrockClient

def extract_date_info(query):
    """
    Extract date information from a query string.
    
    Args:
        query (str): The user query string
        
    Returns:
        dict: A dictionary containing start_date and end_date
    """
    # Default to current month if no date is specified
    current_date = datetime.now()
    year = current_date.year
    month = current_date.month
    
    # Extract month and year if specified
    month_pattern = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    month_match = re.search(month_pattern, query, re.IGNORECASE)
    
    year_pattern = r"\b(20\d{2})\b"  # Match years like 2023, 2024, etc.
    year_match = re.search(year_pattern, query)
    
    if month_match:
        month_name = month_match.group(0).lower()
        month_mapping = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month = month_mapping[month_name]
    
    if year_match:
        year = int(year_match.group(0))
    
    # Calculate the first and last day of the month
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    
    return {
        "start_date": start_date,
        "end_date": end_date
    }

def plan_task_rule_based(query):
    """
    Analyze the user query using rule-based approach to determine the appropriate action plan.
    
    Args:
        query (str): The user query string
        
    Returns:
        dict: A dictionary containing the plan details
    """
    query = query.lower()
    
    # Extract date information
    date_info = extract_date_info(query)
    
    # Determine intent and template
    if "net cash flow" in query or "cash flow" in query:
        return {
            "intent": "forecast",
            "tables": ["payments"],
            "template": "net_cash_flow.sql.j2",
            "parameters": {
                "table": "payments",
                "start_date": date_info["start_date"],
                "end_date": date_info["end_date"]
            }
        }
    elif "inflow" in query or "income" in query:
        return {
            "intent": "analyze_inflow",
            "tables": ["payments"],
            "template": "inflow.sql.j2",
            "parameters": {
                "table": "payments",
                "start_date": date_info["start_date"],
                "end_date": date_info["end_date"]
            }
        }
    elif "outflow" in query or "expenses" in query:
        return {
            "intent": "analyze_outflow",
            "tables": ["payments"],
            "template": "outflow.sql.j2",
            "parameters": {
                "table": "payments",
                "start_date": date_info["start_date"],
                "end_date": date_info["end_date"]
            }
        }
    
    # Default response if no specific intent is identified
    return {
        "intent": "unknown",
        "message": "I couldn't understand your query. Please try asking about net cash flow, inflows, or outflows for a specific month."
    }

def plan_task_with_bedrock(query: str) -> Dict[str, Any]:
    """
    Analyze the user query using AWS Bedrock's Claude model.
    
    Args:
        query (str): The user query string
        
    Returns:
        dict: A dictionary containing the plan details
    """
    try:
        # Get region from environment variable if available
        region = os.environ.get('AWS_REGION')
        
        # Initialize the Bedrock client with the region
        bedrock_client = BedrockClient(region_name=region)
        
        # Analyze the query using Claude
        analysis = bedrock_client.analyze_query(query)
        
        # If Claude couldn't determine the intent, return unknown
        if analysis.get("intent") == "unknown" or "error" in analysis:
            return {
                "intent": "unknown",
                "message": "I couldn't understand your query. Please try asking about net cash flow, inflows, or outflows for a specific month."
            }
        
        # Extract time period information
        time_period = analysis.get("time_period", {})
        
        # Always use 'payments' as the table name since that's what we have in our data directory
        table_name = "payments"
        
        # Get template from analysis or use default
        template = analysis.get("template", "net_cash_flow.sql.j2")
        
        # Extract month and year from time_period
        month = time_period.get("month", datetime.now().month)
        year = time_period.get("year", 2025)  # Default to 2025 for sample data
        
        # If the year is the current year, use 2025 instead (for sample data)
        if year == datetime.now().year:
            year = 2025
            
        # Calculate start and end dates
        last_day = calendar.monthrange(year, month)[1]
        start_date = time_period.get("start_date", f"{year}-{month:02d}-01")
        end_date = time_period.get("end_date", f"{year}-{month:02d}-{last_day:02d}")
        
        # Get parameters from analysis or use defaults
        parameters = analysis.get("parameters", {})
        
        # Update parameters with table and date information
        parameters.update({
            "table": table_name,
            "start_date": start_date,
            "end_date": end_date
        })
        
        return {
            "intent": analysis.get("intent", "forecast"),
            "tables": [table_name],
            "template": template,
            "parameters": parameters,
            "ai_analysis": True
        }
        
    except Exception as e:
        print(f"Error using Bedrock for query analysis: {e}")
        # Fall back to rule-based approach
        return plan_task_rule_based(query)

def plan_task(query: str, use_bedrock: bool = False) -> Dict[str, Any]:
    """
    Analyze the user query and determine the appropriate action plan.
    Uses either rule-based approach or AWS Bedrock's Claude model based on the use_bedrock flag.
    
    Args:
        query (str): The user query string
        use_bedrock (bool): Whether to use AWS Bedrock for analysis
        
    Returns:
        dict: A dictionary containing the plan details
    """
    # Check if AWS credentials are available and use_bedrock is True
    if use_bedrock and os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        try:
            return plan_task_with_bedrock(query)
        except Exception as e:
            print(f"Error using Bedrock: {e}. Falling back to rule-based approach.")
            return plan_task_rule_based(query)
    else:
        # Use rule-based approach
        return plan_task_rule_based(query)
