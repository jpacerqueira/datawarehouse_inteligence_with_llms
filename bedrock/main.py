"""
Cash Flow POC - Main Application

This is the entry point for the Cash Flow POC application.
It handles user input, calls the planning agent, renders SQL templates,
executes queries using DuckDB, and displays results.
"""
import os
import json
import pandas as pd
import duckdb
import jinja2
from datetime import datetime
import argparse
from planning_agent import plan_task

def create_sample_data():
    """
    Create sample payment data and save it as a parquet file.
    This is used for demonstration purposes.
    """
    # Check if data file already exists
    if os.path.exists('data/payments.parquet'):
        print("Sample data already exists.")
        return
    
    # Create sample data
    data = {
        'transaction_id': list(range(1, 11)),
        'transaction_date': [
            '2025-02-01', '2025-02-05', '2025-02-10', '2025-02-15', '2025-02-20',
            '2025-02-22', '2025-02-24', '2025-02-25', '2025-02-27', '2025-02-28'
        ],
        'inflow': [5000, 3000, 0, 0, 10000, 0, 15000, 0, 0, 0],
        'outflow': [0, 0, 1000, 2000, 0, 3000, 0, 2000, 1000, 1000],
        'category': [
            'Sales', 'Sales', 'Utilities', 'Rent', 'Investment', 
            'Salaries', 'Sales', 'Supplies', 'Marketing', 'Miscellaneous'
        ],
        'description': [
            'Product Sales', 'Service Revenue', 'Electricity Bill', 'Office Rent', 'Investor Funding',
            'Employee Salaries', 'Product Sales', 'Office Supplies', 'Marketing Campaign', 'Miscellaneous Expenses'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save as parquet
    df.to_parquet('data/payments.parquet')
    print("Sample data created successfully.")

def execute_query(sql_query):
    """
    Execute a SQL query using DuckDB.
    
    Args:
        sql_query (str): The SQL query to execute
        
    Returns:
        pandas.DataFrame: The query results
    """
    try:
        result = duckdb.query(sql_query).to_df()
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def render_template(template_name, parameters):
    """
    Render a SQL template with the given parameters.
    
    Args:
        template_name (str): The name of the template file
        parameters (dict): The parameters to use in the template
        
    Returns:
        str: The rendered SQL query
    """
    try:
        env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
        template = env.get_template(template_name)
        
        # Adjust table parameter to use read_parquet
        if 'table' in parameters:
            parameters['table'] = f"read_parquet('data/{parameters['table']}.parquet')"
        
        rendered_sql = template.render(**parameters)
        return rendered_sql
    except Exception as e:
        print(f"Error rendering template: {e}")
        return None

def process_query(user_query, use_bedrock=False):
    """
    Process a user query and return the results.
    
    Args:
        user_query (str): The user's query string
        use_bedrock (bool): Whether to use AWS Bedrock for query analysis
        
    Returns:
        dict: The results of the query
    """
    # Get plan from planning agent
    plan = plan_task(user_query, use_bedrock=use_bedrock)
    
    if plan.get('intent') == 'unknown':
        return {
            'status': 'error',
            'message': plan.get('message', 'Unknown query')
        }
    
    # Render SQL template
    rendered_sql = render_template(plan['template'], plan['parameters'])
    
    if not rendered_sql:
        return {
            'status': 'error',
            'message': 'Failed to render SQL template'
        }
    
    # Execute query
    result = execute_query(rendered_sql)
    
    if result is None:
        return {
            'status': 'error',
            'message': 'Failed to execute query'
        }
    
    # Save results to output directory
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'output/result_{timestamp}.json'
    
    # Convert result to dictionary for JSON serialization
    result_dict = result.to_dict('records')
    
    # Save query and results
    output = {
        'query': user_query,
        'sql': rendered_sql,
        'results': result_dict,
        'ai_analysis': plan.get('ai_analysis', False)
    }
    
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    return {
        'status': 'success',
        'sql': rendered_sql,
        'results': result,
        'output_file': result_file,
        'ai_analysis': plan.get('ai_analysis', False)
    }

def main():
    """
    Main function to run the Cash Flow POC application.
    """
    parser = argparse.ArgumentParser(description='Cash Flow POC')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--setup', action='store_true', help='Set up sample data')
    parser.add_argument('--use-bedrock', action='store_true', help='Use AWS Bedrock for query analysis')
    parser.add_argument('--aws-profile', type=str, help='AWS profile to use for Bedrock')
    
    args = parser.parse_args()
    
    if args.setup:
        create_sample_data()
        return
    
    # Set AWS profile if provided
    if args.aws_profile:
        os.environ['AWS_PROFILE'] = args.aws_profile
    
    if not args.query:
        print("Please provide a query using the --query argument.")
        print("Example: python main.py --query \"Show net cash flow for February 2025\"")
        print("Or run with --setup to create sample data.")
        print("Add --use-bedrock to use AWS Bedrock for advanced query analysis.")
        return
    
    # Process the query
    result = process_query(args.query, use_bedrock=args.use_bedrock)
    
    if result['status'] == 'error':
        print(f"Error: {result['message']}")
        return
    
    # Display results
    print("\nGenerated SQL:")
    print(result['sql'])
    
    print("\nResults:")
    print(result['results'])
    
    # Display AI analysis info if available
    if 'ai_analysis' in result and result['ai_analysis']:
        print("\nQuery was analyzed using AWS Bedrock Claude AI.")
    
    print(f"\nResults saved to {result['output_file']}")

if __name__ == "__main__":
    main()
