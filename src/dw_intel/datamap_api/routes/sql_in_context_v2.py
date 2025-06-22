from fastapi import APIRouter, Depends, HTTPException
from quack.datamap_api.models.sql_in_context import SQLInContextResponse
from quack.datamap_api.models.query_request import QueryRequest
from quack.datamap_api.deps.analyzer_lock import get_analyzer

router = APIRouter()

@router.post("/sql_in_context_v2", response_model=SQLInContextResponse, tags=["analysis"])
async def get_sql_in_context_v2(request: QueryRequest):
    """Get SQL in context"""
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        return analyzer.get_sql_in_context_v2(request.query, request.context, request.format_type)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        ) from e
    