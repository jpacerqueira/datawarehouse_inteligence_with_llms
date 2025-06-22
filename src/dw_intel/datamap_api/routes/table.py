from fastapi import APIRouter, Depends, HTTPException
from quack.datamap_api.models.table_request import TableRequest
from quack.datamap_api.models.table_response import TableResponse
from quack.datamap_api.deps.analyzer_lock import get_analyzer

router = APIRouter()

@router.post("/table", response_model=TableResponse, tags=["tables"])
async def get_table(request: TableRequest):
    """Get detailed schema information about a specific table"""
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        return analyzer.get_table_schema(request.table_name)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting table: {str(e)}"
        ) from e