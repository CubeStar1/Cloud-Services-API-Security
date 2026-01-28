from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..utils.labelling_utils import run_full_labelling
from ..types.labelling import LabellingRequest

router = APIRouter()

@router.post("/label", summary="Run full labelling workflow")
async def label(request: LabellingRequest) -> dict[str, str | list[str]]:
    """Trigger the complete labelling process (conversion ➜ split ➜ classification)."""
    try:
        return run_full_labelling(api_key=request.api_key)
    except Exception as exc:
        return {"success": False, "message": "Labelling failed", "error": str(exc)}


