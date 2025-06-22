from pydantic import BaseModel, Field

class SQLRequest(BaseModel):
    sql: str = Field(..., description="SQL query to execute")
#    table_name: str = Field(..., description="Table name")