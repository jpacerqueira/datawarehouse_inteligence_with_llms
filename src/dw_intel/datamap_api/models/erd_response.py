from pydantic import BaseModel
from typing import Dict, Any, Optional

class ERDResponse(BaseModel):
    """Model for ERD generation responses."""
    enriched_schema: Dict[str, Any]
    erd_files: Dict[str, str]  # Dictionary containing paths to generated files (svg, png, json)
    query: str
    context: str
    format_type: str 