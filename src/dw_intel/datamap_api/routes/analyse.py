from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from dw_intel.datamap_api.models.query_request import QueryRequest
from dw_intel.datamap_api.models.analysis_response import AnalysisResponse
from dw_intel.datamap_api.deps.analyzer_lock import get_analyzer

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
async def analyze_schema(request: QueryRequest):
    """Analyze database schema based on natural language query"""
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        return analyzer.analyze_schema(request.query)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        ) from e
