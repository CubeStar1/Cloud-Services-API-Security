from pydantic import BaseModel, Field
from typing import Optional


class RfcTrainRequest(BaseModel):
    input_file: str = Field(..., description="CSV filename located in data/output/codebert/predictions directory")


class RfcInferenceRequest(BaseModel):

    headers_Host: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    requestHeaders_Origin: Optional[str] = None
    requestHeaders_Content_Type: Optional[str] = None
    responseHeaders_Content_Type: Optional[str] = None
    requestHeaders_Referer: Optional[str] = None
    requestHeaders_Accept: Optional[str] = None
