from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
from bedrock_datamap_rag import BedrockDatamapRAG
from dotenv import load_dotenv
import logging
import numpy as np

# Load environment variables
load_dotenv()

# OpenAPI/Swagger documentation
description = """
Cashflow DataMap Schema API helps you analyze and query database schema using AWS Bedrock and RAG.

## Features

* Natural language querying of database schema
* Similarity-based schema retrieval
* Table and column-level schema analysis
* Interactive API documentation
* S3-based data source with parquet files
"""

tags_metadata = [
    {
        "name": "root",
        "description": "Basic API information and health check",
    },
    {
        "name": "analysis",
        "description": "Schema analysis and operations",
    },
    {
        "name": "tables",
        "description": "Table-specific operations and schema",
    },
    {
        "name": "columns",
        "description": "Column-specific operations and analysis",
    }
]

logger = logging.getLogger(__name__)

class DataMapSchemaAnalyzer:
    def __init__(self, config_path: str):
        """Initialize the DataMapSchemaAnalyzer with a configuration file path."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Validate AWS credentials
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key_id or not aws_secret_access_key:
            raise HTTPException(
                status_code=500,
                detail="AWS credentials not found in environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )
        
        # Get AWS credentials from environment variables
        aws_credentials = {
            'AWS_ACCESS_KEY_ID': aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': aws_secret_access_key,
            'AWS_SESSION_TOKEN': os.getenv('AWS_SESSION_TOKEN')
        }
        
        # Get S3 configuration with defaults
        bucket_name = self.config.get('bucket_name', 'project-dw_intel')
        prefix = self.config.get('prefix', 'sbca/batch3/1299438/bronze/')
        region_name = self.config.get('region_name', 'us-east-1')
        cache_size = self.config.get('cache_size', 128)
        
        logger.info(f"Initializing RAG with S3 configuration:")
        logger.info(f"Bucket: {bucket_name}")
        logger.info(f"Prefix: {prefix}")
        logger.info(f"Region: {region_name}")
        
        # Initialize RAG with S3 configuration
        try:
            self.rag = BedrockDatamapRAG(
                bucket_name=bucket_name,
                prefix=prefix,
                region_name=region_name,
                aws_credentials=aws_credentials,
                cache_size=cache_size
            )
            self._initialize_rag()
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize RAG: {str(e)}"
            )
        
    def update_config(self, config: Dict[str, Any]):
        """Update the configuration with new values."""
        try:
            self.config.update(config)
            self._initialize_rag()
            logger.info("Updated configuration:")
            logger.info(f"S3 Configuration:")
            logger.info(f"  - Bucket: {self.config.bucket_name}")
            logger.info(f"  - Prefix: {self.config.prefix}")
            logger.info(f"  - Region: {self.config.region_name}")
            logger.info(f"  - Pattern: {self.config.pattern}")
            logger.info(f"  - Cache Size: {self.config.cache_size}")
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Configuration file not found")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON configuration file")
    
    def _initialize_rag(self):
        """Initialize the RAG system with the S3 schema."""
        try:
            if 'bucket_name' not in self.config:
                raise HTTPException(status_code=400, detail="S3 bucket name not found in configuration")
            
            pattern = self.config.get('pattern')
            logger.info(f"Building RAG index with files in pattern: {pattern}")
            
            self.rag.build_rag_index(pattern=pattern)
            logger.info("RAG index built successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error initializing RAG: {str(e)}"
            )
    
    def analyze_schema(self, query: str, context: str = "cashflow", format_type: str = "sql") -> Dict[str, Any]:
        """Analyze the database schema based on a natural language query."""
        try:
            analysis = self.rag.get_detailed_schema_analysis(query, context, format_type)
            return {
                "analysis": analysis,
                "query": query
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing schema: {str(e)}")
        
    def get_sql_in_context(self, query: str, context: str = "cashflow", format_type: str = "sql") -> Dict[str, Any]:
        """Get SQL in context"""
        try:
            sql_in_context = self.rag.get_detailed_sql_in_context(query, context, format_type)
            return {
                "sql_in_context": sql_in_context,
                "query": query
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting SQL in context: {str(e)}")
    
    def get_similar_schema(self, query: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """Get similar schema entries with similarity scores."""
        try:
            return self.rag.query_schema(query, k)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting similar schema: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get detailed schema information about a specific table."""
        try:
            schema = self.rag.get_table_schema(table_name)
            
            # Convert NumPy types to Python native types in columns
            converted_columns = []
            for col in schema.get("columns", []):
                converted_col = col.copy()
                # Convert numpy.bool_ to Python bool
                if isinstance(converted_col.get("nullable"), (bool, np.bool_)):
                    converted_col["nullable"] = bool(converted_col["nullable"])
                converted_columns.append(converted_col)
            
            # Transform the response to match TableResponse model
            return {
                "name": table_name,
                "columns": converted_columns,
                "row_count": int(schema.get("row_count", 0)),  # Convert to Python int
                "last_modified": str(schema.get("last_modified", "")),  # Convert to string
                "size_bytes": int(schema.get("size_bytes", 0))  # Convert to Python int
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting table schema: {str(e)}")
    
    def get_column_info(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific column."""
        try:
            return self.rag.get_column_info(table_name, column_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting column info: {str(e)}")
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire database schema."""
        try:
            all_tables = self.rag.schema_cache
            summary = {
                "total_tables": len(all_tables),
                "tables": []
            }
            
            for table_name, schema in all_tables.items():
                table_info = {
                    "name": table_name,
                    "column_count": len(schema["columns"]),
                    "columns": [col["name"] for col in schema["columns"]],
                    "row_count": schema["row_count"],
                    "last_modified": schema["last_modified"],
                    "size_bytes": schema["size_bytes"]
                }
                summary["tables"].append(table_info)
            
            return summary
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting schema summary: {str(e)}")

# FastAPI models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about the database schema")
    k: int = Field(default=3, description="Number of results to return", ge=1, le=10)
    context: str = Field(default="cashflow", description="Context for the query")
    format_type: str = Field(default="sql", description="Format type for the response")

class TableRequest(BaseModel):
    table_name: str = Field(..., description="Name of the table to get schema for")

class ColumnRequest(BaseModel):
    table_name: str = Field(..., description="Name of the table")
    column_name: str = Field(..., description="Name of the column")

class AnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Analysis results")
    query: str = Field(..., description="Original query")

class SQLInContextResponse(BaseModel):
    sql_in_context: str = Field(..., description="SQL in context")
    query: str = Field(..., description="Original query")

class SimilarResponse(BaseModel):
    similar_schema: List[str] = Field(..., description="List of similar schema entries")
    scores: List[float] = Field(..., description="Similarity scores for each entry")

class TableResponse(BaseModel):
    name: str = Field(..., description="Table name")
    columns: List[Dict[str, Any]] = Field(..., description="Table columns schema")
    row_count: int = Field(..., description="Number of rows in the table")
    last_modified: str = Field(..., description="Last modified timestamp")
    size_bytes: int = Field(..., description="File size in bytes")

class ColumnResponse(BaseModel):
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    nullable: bool = Field(..., description="Whether column contains nulls")
    unique_values: int = Field(..., description="Number of unique values")
    null_count: int = Field(..., description="Number of null values")
    sample_values: List[Any] = Field(..., description="Sample of unique values")

class SchemaSummaryResponse(BaseModel):
    total_tables: int = Field(..., description="Total number of tables")
    tables: List[Dict[str, Any]] = Field(..., description="List of tables with schema")

class ConfigRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration data for the analyzer")

# Initialize FastAPI app with enhanced documentation
app = FastAPI(
    title="Cashflow DataMap Schema API",
    description=description,
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Initialize analyzer
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    config_path = os.getenv('DATAMAP_CONFIG_PATH', 'config/datamap_config.json')
    try:
        if os.path.exists(config_path):
            analyzer = DataMapSchemaAnalyzer(config_path)
    except Exception as e:
        print(f"Error initializing analyzer: {str(e)}")
        raise

@app.post("/initialize", tags=["root"])
async def initialize_analyzer(request: ConfigRequest):
    """Initialize the analyzer with provided configuration"""
    global analyzer
    try:
        # Get current configuration        
        # Update configuration with new values
        analyzer.update_config(request.config)
        
        # Log the updated configuration
        logger.info("Updated configuration:")
        logger.info(f"S3 Configuration:")
        logger.info(f"  - Bucket: {analyzer.config.bucket_name}")
        logger.info(f"  - Prefix: {analyzer.config.prefix}")
        logger.info(f"  - Region: {config.aws_region}")
        logger.info(f"  - Pattern: {config.pattern}")
        logger.info(f"  - Cache Size: {config.cache_size}")
        
        logger.info(f"Bedrock Configuration:")
        logger.info(f"  - Region: {config.bedrock_region}")
        logger.info(f"  - Embeddings Model: {config.bedrock_embeddings_model}")
        logger.info(f"  - Inference Model: {config.bedrock_inference_model}")
        
        # Initialize or reinitialize analyzer with new configuration
        if analyzer:
            logger.info("Reinitializing analyzer with new configuration")
            analyzer = None
        
        analyzer = DataMapSchemaAnalyzer(request.config)
        
        return {
            "message": "Analyzer initialized successfully",
            "configuration": {
                "s3": {
                    "bucket_name": config.bucket_name,
                    "prefix": config.prefix,
                    "region": config.aws_region,
                    "pattern": config.pattern,
                    "cache_size": config.cache_size
                },
                "bedrock": {
                    "region": config.bedrock_region,
                    "embeddings_model": config.bedrock_embeddings_model,
                    "inference_model": config.bedrock_inference_model
                }
            }
        }
    except Exception as e:
        logger.error(f"Error initializing analyzer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing analyzer: {str(e)}"
        )

@app.get("/", tags=["root"])
async def root():
    """Root endpoint returning API status"""
    return {"message": "Cashflow DataMap Schema API"}

@app.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
async def analyze_schema(request: QueryRequest):
    """Analyze database schema based on natural language query"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        return analyzer.analyze_schema(request.query, request.context, request.format_type)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/sql_in_context", response_model=SQLInContextResponse, tags=["analysis"])
async def get_sql_in_context(request: QueryRequest):
    """Get SQL in context"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        return analyzer.get_sql_in_context(request.query, request.context, request.format_type)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/similar", response_model=SimilarResponse, tags=["analysis"])
async def get_similar(request: QueryRequest):
    """Get similar schema entries with similarity scores"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    similar_schema, scores = analyzer.get_similar_schema(request.query, request.k)
    return {
        "similar_schema": similar_schema,
        "scores": scores
    }

@app.post("/table", response_model=TableResponse, tags=["tables"])
async def get_table(request: TableRequest):
    """Get detailed schema information about a specific table"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    return analyzer.get_table_schema(request.table_name)

@app.post("/column", response_model=ColumnResponse, tags=["columns"])
async def get_column(request: ColumnRequest):
    """Get detailed information about a specific column"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    return analyzer.get_column_info(request.table_name, request.column_name)

@app.get("/schema", response_model=SchemaSummaryResponse, tags=["analysis"])
async def get_schema():
    """Get summary of entire database schema"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    return analyzer.get_schema_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 