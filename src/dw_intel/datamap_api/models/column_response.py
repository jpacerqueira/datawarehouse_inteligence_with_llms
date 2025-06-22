from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ColumnResponse(BaseModel):
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    nullable: bool = Field(..., description="Whether column contains nulls")
    unique_values: int = Field(..., description="Number of unique values")
    null_count: int = Field(..., description="Number of null values")
    sample_values: List[Any] = Field(..., description="Sample of unique values")
