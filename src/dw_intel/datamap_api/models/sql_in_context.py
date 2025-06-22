from pydantic import BaseModel, Field

class SQLInContextResponse(BaseModel):
    sql_in_context: str = Field(..., description="SQL in context")
    query: str = Field(..., description="Original query")
    validation_status: str = Field(..., description="Validation status")
    validation_message: str = Field(..., description="Validation message")