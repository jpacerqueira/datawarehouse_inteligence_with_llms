from pydantic import BaseModel, Field
from typing import Dict, Any

class ConfigRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration data for the analyzer")