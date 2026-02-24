from pydantic import BaseModel, Field

class ProcessRequest(BaseModel):
    json_data: dict
    query: str = Field(min_length=2, max_length=500)