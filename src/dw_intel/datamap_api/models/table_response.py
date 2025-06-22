from typing import List, Dict, Any
from pydantic import BaseModel, Field

class TableResponse(BaseModel):
    name: str = Field(..., description="Table name")
    columns: List[Dict[str, Any]] = Field(..., description="Table columns schema")
    row_count: int = Field(..., description="Number of rows in the table")
    last_modified: str = Field(..., description="Last modified timestamp")
    size_bytes: int = Field(..., description="File size in bytes")
