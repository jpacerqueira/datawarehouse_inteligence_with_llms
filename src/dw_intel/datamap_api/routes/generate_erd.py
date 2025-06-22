from fastapi import APIRouter, Depends, HTTPException
from quack.datamap_api.models.erd_request import ERDRequest
from quack.datamap_api.models.erd_response import ERDResponse
from quack.datamap_api.deps.analyzer_lock import get_analyzer
from quack.shared.analyser import DataMapSchemaAnalyzer
from typing import Dict, Optional

router = APIRouter()

@router.post("/generate_erd/analyze", response_model=ERDResponse, tags=["generate_erd"])
async def analyze_schema_for_erd(
    request: ERDRequest
) -> ERDResponse:
    """
    Analyze the schema and generate ERD based on the provided query.
    This endpoint performs the same operation as the UI's ERD generation.
    """
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        # Get the enriched schema analysis
        enriched_schema = analyzer.get_detailed_erd_schema(
            query=request.query,
            context=request.context,
            format_type=request.format_type
        )
        
        # Generate and store ERD files
        erd_files = analyzer.erd_analyzer.generate_and_store_erd()
        
        return ERDResponse(
            enriched_schema=enriched_schema,
            erd_files=erd_files,
            query=request.query,
            context=request.context,
            format_type=request.format_type
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating ERD: {str(e)}"
        )

@router.get("/generate_erd/files", response_model=Dict[str, Optional[str]], tags=["generate_erd"])
async def get_latest_erd_files() -> Dict[str, Optional[str]]:
    """
    Get the latest generated ERD files.
    This endpoint returns the paths to the latest generated ERD files.
    Files that haven't been generated will return None.
    """
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        latest_files = analyzer.erd_analyzer.get_latest_erd_files()
        return latest_files
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting latest ERD files: {str(e)}"
        )
