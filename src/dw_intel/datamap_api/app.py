import logging
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from dw_intel.datamap_api.routes import router as main_router
from dw_intel.shared.analyser import DataMapSchemaAnalyzer
from dw_intel.datamap_api.deps.analyzer_lock import set_analyzer
from dw_intel.datamap_api.configuration.api import ApiConfiguration
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
    },
]

logger = logging.getLogger(__name__)


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
    },
)


@app.on_event("startup")
async def startup_event():
    try:
        configuration = ApiConfiguration()
        set_analyzer(DataMapSchemaAnalyzer(configuration))
    except Exception as e:
        logger.error(f"Error initializing analyzer: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup if needed


app.include_router(
    router=main_router,
    prefix="/api/v1",
    responses={404: {"description": "Not found"}},
)
