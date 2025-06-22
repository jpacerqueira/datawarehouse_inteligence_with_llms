from fastapi import APIRouter, Depends, HTTPException
from dw_intel.datamap_api.models.sql_request import SQLRequest
from dw_intel.datamap_api.models.sql_response import SQLResponse
from dw_intel.datamap_api.deps.analyzer_lock import get_analyzer

router = APIRouter()

@router.post("/execute_sql", response_model=SQLResponse, tags=["execute_sql"])
async def execute_sql(request: SQLRequest):
    """Execute a SQL query via duckdb to the S3 bucket in the configuration prefix and pattern"""
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="SQL execution failed")
    try:
        return analyzer.execute_sql(request.sql)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error executing SQL: {str(e)}"
        ) from e
