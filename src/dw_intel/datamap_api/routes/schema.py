from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from dw_intel.datamap_api.models.schema_summary_response import SchemaSummaryResponse
from dw_intel.datamap_api.deps.analyzer_lock import get_analyzer

router = APIRouter()

@router.get("/schema", response_model=SchemaSummaryResponse, tags=["analysis"])
async def get_schema():
    """Get summary of entire database schema"""
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        return  analyzer.get_schema_summary()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing schema: {str(e)}"
        ) from e
    