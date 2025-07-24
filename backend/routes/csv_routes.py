from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..utils.csv_utils import convert_all_raw_json_to_csv

router = APIRouter()


@router.post("/json-to-csv", summary="Convert all raw JSON logs to CSV")
async def json_to_csv() -> dict[str, str | list[str]]:
    try:
        created = convert_all_raw_json_to_csv()
        return {
            "success": True,
            "message": "CSV conversion completed successfully",
            "output": "\n".join(str(p) for p in created),
        }
    except Exception as exc:
        return {
            "success": False,
            "message": "CSV conversion failed",
            "error": str(exc),
        }
