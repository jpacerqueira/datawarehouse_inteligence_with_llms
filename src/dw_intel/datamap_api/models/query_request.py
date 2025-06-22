from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about the database schema")
    k: int = Field(default=3, description="Number of results to return", ge=1, le=10)
    context: str = Field(default="cashflow", description="Context for the query")
    format_type: str = Field(default="sql", description="Format type for the response")
