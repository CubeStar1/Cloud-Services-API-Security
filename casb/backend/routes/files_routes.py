from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query

from ..utils.file_listing import list_data_files, read_data_file

router = APIRouter()


@router.get("/files", summary="List or read files inside data folder")
async def list_or_read_files(
    subdir: Optional[str] = Query(
        None, description="Subdirectory within data folder to list (relative path)"
    ),
    ext: Optional[str] = Query(
        None, description="Comma-separated list of file extensions to filter, e.g. csv,joblib"
    ),
    file: Optional[str] = Query(
        None,
        description="Relative path to a single file inside data folder to read (takes precedence)",
    ),
):
    """List files or return the content of a single file in the data directory."""
    from pathlib import Path

    if file:
        try:
            return read_data_file(file)
        except ValueError:
            return {"error": "Invalid file path"}
        except FileNotFoundError:
            return {"error": "File not found"}

    extensions = [e.strip() for e in ext.split(",")] if ext else None
    return list_data_files(subdir, extensions)
