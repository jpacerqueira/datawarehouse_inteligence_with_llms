import boto3
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from typing import List, Dict, Any, Optional, Set, Pattern
import logging
from botocore.exceptions import ClientError
import os
import re
from functools import lru_cache
from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class S3DataSource:
    def __init__(self, bucket_name: str, prefix: str = "", cache_size: int = 128):
        """Initialize S3 data source handler for parquet files.
        
        Args:
            bucket_name (str): Name of the S3 bucket
            prefix (str): Prefix for the parquet files in the bucket
            cache_size (int): Maximum number of files to cache in memory
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
        self._cache_size = cache_size
        
    @lru_cache(maxsize=128)
    def list_parquet_files(self, pattern: Optional[str] = None) -> List[str]:
        """List all parquet files in the specified S3 location.
        
        Args:
            pattern (Optional[str]): Regex pattern to filter filenames
            
        Returns:
            List[str]: List of S3 keys for parquet files
        """
        try:
            all_files = []
            continuation_token = None
            
            while True:
                kwargs = {
                    'Bucket': self.bucket_name,
                    'Prefix': self.prefix
                }
                if continuation_token:
                    kwargs['ContinuationToken'] = continuation_token
                    
                response = self.s3_client.list_objects_v2(**kwargs)
                
                files = [
                    obj['Key'] for obj in response.get('Contents', [])
                    if obj['Key'].endswith('.parquet')
                ]
                
                if pattern:
                    regex = re.compile(pattern)
                    files = [f for f in files if regex.search(os.path.basename(f))]
                
                all_files.extend(files)
                
                if not response.get('IsTruncated'):
                    break
                    
                continuation_token = response.get('NextContinuationToken')
            
            return all_files
        except ClientError as e:
            logger.error(f"Error listing parquet files: {str(e)}")
            raise
    
    @lru_cache(maxsize=128)
    def read_parquet(self, key: str) -> pd.DataFrame:
        """Read a parquet file from S3 into a pandas DataFrame using PyArrow.
        
        Args:
            key (str): S3 key of the parquet file
            
        Returns:
            pd.DataFrame: The loaded parquet data
        """
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            # Read the entire file into memory using BytesIO
            buffer = BytesIO(obj['Body'].read())
            # Read parquet file using PyArrow
            table = pq.read_table(buffer)
            # Convert to pandas DataFrame
            return table.to_pandas()
        except Exception as e:
            logger.error(f"Error reading parquet file {key}: {str(e)}")
            raise
    
    def get_column_info(self, key: str, column_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific column.
        
        Args:
            key (str): S3 key of the parquet file
            column_name (str): Name of the column to analyze
            
        Returns:
            Dict[str, Any]: Column information including:
                - name: Column name
                - type: Data type
                - nullable: Whether column contains nulls
                - unique_values: Number of unique values
                - null_count: Number of null values
                - sample_values: Sample of unique values
        """
        try:
            df = self.read_parquet(key)
            if column_name not in df.columns:
                raise ValueError(f"Column {column_name} not found in table")
                
            col = df[column_name]
            return {
                "name": column_name,
                "type": str(col.dtype),
                "nullable": col.isna().any(),
                "unique_values": col.nunique(),
                "null_count": col.isna().sum(),
                "sample_values": col.dropna().unique().tolist()[:5]
            }
        except Exception as e:
            logger.error(f"Error getting column info for {column_name} in {key}: {str(e)}")
            raise
    
    def get_table_schema(self, key: str) -> Dict[str, Any]:
        """Get schema information for a parquet file.
        
        Args:
            key (str): S3 key of the parquet file
            
        Returns:
            Dict[str, Any]: Schema information including:
                - table_name: Name derived from the file
                - columns: List of column information (name, type, nullable)
                - row_count: Number of rows in the table
                - last_modified: Last modified timestamp
                - size_bytes: File size in bytes
        """
        try:
            df = self.read_parquet(key)
            
            # Get file metadata
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            
            return {
                "table_name": os.path.basename(key).replace('.parquet', ''),
                "columns": [
                    {
                        "name": col,
                        "type": str(df[col].dtype),
                        "nullable": df[col].isna().any()
                    }
                    for col in df.columns
                ],
                "row_count": len(df),
                "last_modified": response['LastModified'].isoformat(),
                "size_bytes": response['ContentLength']
            }
        except Exception as e:
            logger.error(f"Error getting schema for {key}: {str(e)}")
            raise
    
    def get_all_schemas(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get schema information for all parquet files in the S3 location.
        
        Args:
            pattern (Optional[str]): Regex pattern to filter filenames
            
        Returns:
            List[Dict[str, Any]]: List of schema information for each table
        """
        try:
            parquet_files = self.list_parquet_files(pattern)
            return [self.get_table_schema(file) for file in parquet_files]
        except Exception as e:
            logger.error(f"Error getting all schemas: {str(e)}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the LRU cache for both file listing and parquet reading."""
        self.list_parquet_files.cache_clear()
        self.read_parquet.cache_clear() 