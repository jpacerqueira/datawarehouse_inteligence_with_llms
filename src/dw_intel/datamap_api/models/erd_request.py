from pydantic import BaseModel
from typing import Optional

class ERDRequest(BaseModel):
    """Model for ERD generation requests."""
    query: str
    context: Optional[str] = "database_schema"
    format_type: Optional[str] = "json" 