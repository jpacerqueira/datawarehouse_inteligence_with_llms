from typing import Any, Dict, List
from pydantic import BaseModel, Field

class SchemaSummaryResponse(BaseModel):
    total_tables: int = Field(..., description="Total number of tables")
    tables: List[Dict[str, Any]] = Field(..., description="List of tables with schema")
