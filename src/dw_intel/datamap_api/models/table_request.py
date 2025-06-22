from pydantic import BaseModel, Field

class TableRequest(BaseModel):
    table_name: str = Field(..., description="Name of the table to get schema for")
