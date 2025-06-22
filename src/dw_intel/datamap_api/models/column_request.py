from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ColumnRequest(BaseModel):
    table_name: str = Field(..., description="Name of the table")
    column_name: str = Field(..., description="Name of the column")
