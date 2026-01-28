from pydantic import BaseModel, Field


class LabellingRequest(BaseModel):
    api_key: str = Field(..., description="API key for authentication")

