from typing import List, Dict, Any
from pydantic import BaseModel, Field

class SQLResponse(BaseModel):
    sql_data: List[Dict[str, Any]] = Field(..., description="Data from the table")
#    table_name: str = Field(..., description="Table name")