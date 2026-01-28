from typing import List, Optional
from pydantic import BaseModel, Field

class LabellingRequest(BaseModel):
    api_key: str = Field(..., description="API key for authentication")
    services: Optional[List[str]] = Field(None, description="List of services to classify")
    activities: Optional[List[str]] = Field(None, description="List of activities to classify")

