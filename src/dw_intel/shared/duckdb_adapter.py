import duckdb
from pathlib import Path
from typing import List, Optional, Dict
import os
import boto3
from botocore.exceptions import ClientError
import logging
from dotenv import load_dotenv
# load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

#secret_query = """CREATE SECRET IF NOT EXISTS secret4 (
#TYPE S3,
#PROVIDER CREDENTIAL_CHAIN,
#CHAIN 'config',
#REGION 'us-east-1');"""

#def get_duckdb() -> duckdb:
#    return duckdb

def get_duckdb_s3_connection(region: str = 'us-east-1') -> duckdb.DuckDBPyConnection:
    """
    Get a DuckDB connection configured for S3 access using boto3 client.
    
    Args:
        region (str): AWS region for S3 access
    
    Returns:
        DuckDBPyConnection: Configured DuckDB connection
    """
    try:
        # session config from boto3
        session = boto3.Session()
        credentials = session.get_credentials()

        # Create a new connection
        conn = duckdb.connect()
        
        # install and enable the httpfs extension
        # Prerequisites: - https://duckdb.org/docs/stable/guides/network_cloud_storage/s3_import.html
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")

        # Issues workarround
        # 1. DuckDB SSO - https://github.com/duckdb/duckdb-aws/issues/14
        # 2. DuckDB S3 - https://github.com/duckdb/duckdb/issues/10409
      
        os.environ['AWS_PROFILE'] = os.getenv('AWS_PROFILE', 'dw_intel-copilot-dev')
        os.environ['AWS_REGION'] = region
        os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
        os.environ['AWS_SESSION_TOKEN'] = credentials.token

        credentials_result = conn.execute("call load_aws_credentials();")
        logger.info(f"credentials_result: {credentials_result}")

        # Configure S3 using boto3's credential chain with explicit settings
        connection_string = f"""
        CREATE OR REPLACE SECRET secret8 (
            TYPE s3,
            PROVIDER credential_chain,
            CHAIN 'sso',
            REGION '{region}',
            PROFILE '{os.getenv('AWS_PROFILE', 'dw_intel-copilot-dev')}'
        )
        """
        logger.info(f"Creating S3 connection with: {connection_string}")
        conn.execute(connection_string)
        
        # Test the S3 connection with a simple operation
        try:
            test_query = "SELECT count(*) FROM read_parquet('s3://project-quack/sbca/test/test.parquet') "
            conn.execute(test_query)
            logger.info("S3 connection test successful")
        except Exception as e:
            logger.warning(f"S3 connection test failed: {str(e)}")
        
        # From secreted list, get the secret
        secrets = conn.execute("FROM duckdb_secrets();").fetchall()
        
        # Log connection status and configuration
        logger.info(f"ducbdb s3 - connection object: {conn}")
        logger.info(f"ducbdb s3 - connection secrets: {secrets}")
        
        return conn
    except Exception as e:
        logger.error(f"Error creating S3 connection: {str(e)}")
        raise

def execute_query_select_start_parquet(
    query: str,
    parquet_path: str,
    table_name: Optional[str] = None,
    connection: Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb.DuckDBPyRelation:
    """
    Execute a SQL query against parquet files where each file represents a table with the same name.
    
    Args:
        query (str): The SQL query to execute
        parquet_path (str): Path to the parquet files (can be a directory or a specific file)
        table_name (str, optional): Name of the table to use in the query. If None, will use the file name
        connection (DuckDBPyConnection, optional): Existing DuckDB connection to use
    
    Returns:
        DuckDBPyRelation: The result of the query
    """
    # Create a new connection if none provided
    if connection is None:
        connection = get_duckdb_s3_connection()
    
    # Get the table name from the file path if not provided
    if table_name is None:
        table_name = Path(parquet_path).stem
    
    # Register the parquet file(s) as a table
    connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')")
    
    # Execute the query
    result = connection.execute(query)
    
    return result

def execute_s3_pattern_query(
    query: str,
    s3_path: str,
    pattern: str,
    connection: Optional[duckdb.DuckDBPyConnection] = None,
    region: str = 'us-east-1'
) -> Dict[str, duckdb.DuckDBPyRelation]:
    """
    Execute a SQL query against multiple tables found in an S3 folder based on a pattern.
    
    Args:
        query (str): The SQL query to execute (can use {table_name} placeholder)
        s3_path (str): S3 path to the folder containing parquet files (e.g., 's3://my-bucket/data/')
        pattern (str): Pattern to match table names (e.g., '*.parquet' or 'table_*.parquet')
        connection (DuckDBPyConnection, optional): Existing DuckDB connection to use
    
    Returns:
        Dict[str, DuckDBPyRelation]: Dictionary mapping table names to their query results
    """
    if connection is None:
        connection = get_duckdb_s3_connection(region)

    # replace $ with % in the pattern
    local_pattern = pattern.replace('.*', '%').replace('$', '').replace('\\', '')
    
    # List all parquet files matching the pattern in the S3 path
    list_query = f"""
    SELECT DISTINCT filename as file_path
    FROM read_parquet('{s3_path}*.parquet', filename=true)
    WHERE filename LIKE '{local_pattern}'
    """
    logger.info(f"Listing files: {list_query}")
    files = connection.execute(list_query).fetchall()
    logger.info(f"Files: {files}")
    
    results = {}
    for file_path, in files:
        # Extract table name from the file path
        table_name = Path(file_path).stem
        
        # Create the table from the parquet file
        table_query = f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM read_parquet('{file_path}')
        """
        logger.info(f"Creating table: {table_query}")
        try:
            connection.execute(table_query)
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            raise
    # Execute the query
    logger.info(f"Executing query: {query}")
    try:
        # Extract table name from SQL statement
        query_lower = query.lower()
        table_name = None
        
        # Try to get table name from WITH clause
        if "with" in query_lower:
            with_parts = query_lower.split("with", 1)[1].strip()
            table_name = with_parts.split()[0]
        
        # If no WITH clause or empty table name, try FROM clause
        if not table_name and "from" in query_lower:
            from_parts = query_lower.split("from", 1)[1].strip()
            table_name = from_parts.split()[0]
            
        if not table_name:
            raise ValueError("Could not determine table name from query")
            
        results[table_name] = connection.execute(query)
        logger.info(f"Results: {results[table_name]}")
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise
    
    return results