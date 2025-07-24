"""Pydantic schemas for labelling endpoints."""
from pydantic import BaseModel, Field


class LabellingRequest(BaseModel):
    """Request payload for labelling endpoints."""
    api_key: str = Field(..., description="API key for authentication")

