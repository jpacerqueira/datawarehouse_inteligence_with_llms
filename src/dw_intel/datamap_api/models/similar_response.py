from typing import List
from pydantic import BaseModel, Field

class SimilarResponse(BaseModel):
    similar_schema: List[str] = Field(..., description="List of similar schema entries")
    scores: List[float] = Field(..., description="Similarity scores for each entry")
