from fastapi import APIRouter, Depends, HTTPException

from quack.datamap_api.models.column_request import ColumnRequest
from quack.datamap_api.models.column_response import ColumnResponse
from quack.datamap_api.deps.analyzer_lock import get_analyzer

router = APIRouter()


@router.post("/column", response_model=ColumnResponse, tags=["columns"])
async def get_column(request: ColumnRequest):
    """Get detailed information about a specific column"""
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        return analyzer.get_column_info(request.table_name, request.column_name)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting column: {str(e)}"
        ) from e
