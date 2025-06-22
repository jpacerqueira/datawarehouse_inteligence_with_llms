from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
from dw_intel.datamap_api.models.query_request import QueryRequest
from dw_intel.datamap_api.models.similar_response import SimilarResponse
from dw_intel.datamap_api.deps.analyzer_lock import get_analyzer

router = APIRouter()

@router.post("/similar", response_model=SimilarResponse, tags=["analysis"])
async def get_similar(request: QueryRequest):
    """Get similar schema entries with similarity scores"""
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        similar_schema, scores = analyzer.get_similar_schema(request.query, request.k)
        return {
            "similar_schema": similar_schema,
            "scores": scores
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting similar schema: {str(e)}"
        ) from e