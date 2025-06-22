import os
import logging
import json
from typing import Dict, Any, List, Tuple, Optional

from fastapi import HTTPException
import pandas as pd
import numpy as np

from quack.datamap_api.configuration.api import ApiConfiguration
from quack.shared.rag import BedrockDatamapRAG
from quack.shared.produce_tables_erd import TableRelationshipAnalyzer

from quack.shared.duckdb_adapter import get_duckdb_s3_connection, execute_s3_pattern_query
from quack.shared.sql_plan_validator import SQLPlanValidator
from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class DataMapSchemaAnalyzer:
    config: ApiConfiguration

    def __init__(self, configuration: ApiConfiguration):
        self.config = configuration

        # Get S3 configuration with defaults
        bucket_name = self.config.bucket_name
        prefix = self.config.prefix
        region_name = self.config.aws_region
        cache_size = self.config.cache_size

        bedrock_region = self.config.bedrock_region
        bedrock_embeddings_model = self.config.bedrock_embeddings_model
        bedrock_inference_model = self.config.bedrock_inference_model

        logger.info("Initializing RAG with S3 configuration:")
        logger.info("Bucket: %s", bucket_name)
        logger.info("Prefix: %s", prefix)
        logger.info("Region: %s", region_name)

        # Initialize RAG with S3 configuration
        try:
            self.rag = BedrockDatamapRAG(
                bucket_name=bucket_name,
                prefix=prefix,
                region_name=bedrock_region,
                embeddings_model=bedrock_embeddings_model,
                inference_model=bedrock_inference_model,
                cache_size=cache_size,
            )
            self._initialize_rag()
         
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize RAG: {str(e)}"
            ) from e

        # Initialize SQL plan validator with schema cache
        try:
            self.sql_validator = SQLPlanValidator(schema_cache = self.rag.schema_cache)
            self._initialize_sql_validator()
        except Exception as e:
            logger.error(f"Failed to initialize SQL plan validator: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize SQL plan validator: {str(e)}"
            ) from e
        
        # Initialize ERD analyzer
        self.erd_analyzer = TableRelationshipAnalyzer(
            rag_instance=self.rag,
            bucket_name=bucket_name,
            prefix=prefix,
            region_name=region_name
        )

    def _initialize_rag(self):
        """Initialize the RAG system with the S3 schema."""
        try:
            pattern = self.config.pattern
            logger.info("Building RAG index with pattern: %s", pattern)

            self.rag.build_rag_index(pattern=pattern)
            logger.info("RAG index - text embeddings - built successfully")
            self.rag.build_rag_index_v2(pattern=pattern)
            logger.info("RAG index v2 - tables as documents embeddings - built successfully")
            self.rag.build_rag_index_v3(pattern=pattern)
            logger.info("RAG index v3 - tables as documents embeddings - built successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error initializing RAG: {str(e)}"
            ) from e

    def _initialize_sql_validator(self):
        """Initialize the SQL plan validator with the schema cache."""
        try:
            schema_cache = self.rag.schema_cache
            #logger.info("Schema cache: %s", schema_cache)
            self.sql_validator = SQLPlanValidator(schema_cache=schema_cache)
        except Exception as e:
            logger.error(f"Failed to initialize SQL plan validator: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize SQL plan validator: {str(e)}"
            ) from e

    def analyze_schema(self, query: str, context: str = "cashflow", format_type: str = "sql") -> Dict[str, Any]:
        """Analyze the database schema based on a natural language query."""
        try:
            analysis = self.rag.get_detailed_schema_analysis(query, context, format_type)
            return {"analysis": analysis, "query": query}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error analyzing schema: {str(e)}"
            ) from e

    def get_sql_in_context(self, query: str, context: str = "cashflow", format_type: str = "sql") -> Dict[str, Any]:
        """Get SQL in context"""
        try:
            sql_in_context = self.rag.get_detailed_sql_in_context(query, context, format_type)

            # Clean the SQL plan
            sql_in_context = self.sql_validator.clean_sql_plan(sql_in_context)
            # log the sql_in_context
            logger.info(f"SQL in context - cleaned: {sql_in_context}")
            # Validate the SQL plan
            is_valid, validation_message = self.sql_validator.validate_sql_plan(sql_in_context, context)
            if not is_valid:
                logger.warning(f"SQL validation failed: {validation_message}")
                return {
                    "sql_in_context": sql_in_context,
                    "query": query,
                    "validation_status": "failed",
                    "validation_message": validation_message
                }
            
            return {
                "sql_in_context": sql_in_context,
                "query": query,
                "validation_status": "success",
                "validation_message": "SQL plan validated successfully"
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting SQL in context: {str(e)}"
            ) from e
        
    def get_sql_in_context_v2(self, query: str, context: str = "cashflow", format_type: str = "sql") -> Dict[str, Any]:
        """Get SQL in context"""
        try:
            sql_in_context = self.rag.get_detailed_sql_in_context_v2(query, context, format_type)

            # Clean the SQL plan
            sql_in_context = self.sql_validator.clean_sql_plan(sql_in_context)
            # log the sql_in_context
            logger.info(f"SQL in context V2 - cleaned: {sql_in_context}")
            # Validate the SQL plan
            is_valid, validation_message = self.sql_validator.validate_sql_plan(sql_in_context, context)
            if not is_valid:
                logger.warning(f"SQL validation failed: {validation_message}")
                return {
                    "sql_in_context": sql_in_context,
                    "query": query,
                    "validation_status": "failed",
                    "validation_message": validation_message
                }
            
            return {
                "sql_in_context": sql_in_context,
                "query": query,
                "validation_status": "success",
                "validation_message": "SQL plan validated successfully"
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting SQL in context v2: {str(e)}"
            ) from e

    def get_similar_schema(
        self, query: str, k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """Get similar schema entries with similarity scores."""
        try:
            return self.rag.query_schema(query, k)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting similar schema: {str(e)}"
            ) from e

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get detailed schema information for a specific table."""
        try:
            response = self.rag.get_table_schema(table_name)
            
            # Convert NumPy types to Python native types in columns
            if 'columns' in response:
                for column in response['columns']:
                    # Convert numpy.bool_ to Python bool
                    if 'nullable' in column:
                        column['nullable'] = bool(column['nullable'])
                    # Convert numpy.int64 to Python int
                    if 'unique_values' in column:
                        column['unique_values'] = int(column['unique_values'])
                    if 'null_count' in column:
                        column['null_count'] = int(column['null_count'])
                    # Convert sample values to strings and handle datetime values
                    if 'sample_values' in column:
                        column['sample_values'] = [
                            str(val) if not isinstance(val, (pd.Timestamp, np.datetime64)) 
                            else val.isoformat() if hasattr(val, 'isoformat') 
                            else str(val) 
                            for val in column['sample_values']
                        ]
            
            # Convert other NumPy types in the response
            if 'row_count' in response:
                response['row_count'] = int(response['row_count'])
            if 'size_bytes' in response:
                response['size_bytes'] = int(response['size_bytes'])
            
            # Convert last_modified to ISO format string if it's a datetime
            if 'last_modified' in response:
                if isinstance(response['last_modified'], (pd.Timestamp, np.datetime64)):
                    response['last_modified'] = response['last_modified'].isoformat()
                elif isinstance(response['last_modified'], str):
                    # If it's already a string, ensure it's in ISO format
                    try:
                        dt = pd.to_datetime(response['last_modified'])
                        response['last_modified'] = dt.isoformat()
                    except:
                        pass
            
            # Rename table_name to name to match the response model
            if 'table_name' in response:
                response['name'] = response.pop('table_name')
                
            return response
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error getting table schema: {str(e)}"
            ) from e

    def get_column_info(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific column."""
        try:
            return self.rag.get_column_info(table_name, column_name)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting column info: {str(e)}"
            ) from e

    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire database schema."""
        try:
            all_tables = self.rag.schema_cache
            summary = {"total_tables": len(all_tables), "tables": []}

            for table_name, schema in all_tables.items():
                table_info = {
                    "name": table_name,
                    "column_count": len(schema["columns"]),
                    "columns": [col["name"] for col in schema["columns"]],
                    "row_count": schema["row_count"],
                    "last_modified": schema["last_modified"],
                    "size_bytes": schema["size_bytes"],
                }
                summary["tables"].append(table_info)

            return summary
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting schema summary: {str(e)}"
            ) from e

    def execute_sql(self, sql: str, conn: DuckDBPyConnection = None) -> Dict[str, Any]:
        """Execute a SQL query and return the results."""
        try:
            # Validate the SQL plan before execution
            is_valid, validation_message = self.sql_validator.validate_sql_plan(sql)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid SQL plan: {validation_message}"
                )
            
            local_region = self.config.aws_region if self.config.aws_region else "us-east-1"
            if conn is None:
                conn = get_duckdb_s3_connection(region=local_region)
            results = execute_s3_pattern_query(
                query=sql,
                s3_path=f"s3://{self.config.bucket_name}/{self.config.prefix}",
                pattern=self.config.pattern,
                connection=conn,
                region=local_region
            )
            return results
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error executing SQL: {str(e)}"
            ) from e
    
    def get_detailed_erd_schema(self, query: str = "Analyze all tables and their relationships", context: str = "database_schema", format_type: str = "json") -> Dict[str, Any]:
        """Get a detailed analysis of table relationships and schema structure using LLM."""
        try:
            return self.rag.get_detailed_erd_schema(query, context, format_type)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting detailed ERD schema: {str(e)}"
            ) from e