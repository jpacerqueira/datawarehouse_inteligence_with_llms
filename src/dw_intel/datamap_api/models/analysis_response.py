from pydantic import BaseModel, Field

class AnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Analysis results")
    query: str = Field(..., description="Original query")