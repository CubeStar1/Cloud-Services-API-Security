from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

from .path_config import PATHS


def list_data_files(
    subdir: str | None = None,
    extensions: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Return metadata for files in data folder.

    Parameters
    ----------
    subdir: str | None
        Relative path inside the data folder to list. If None, list entire data dir.
    extensions: list[str] | None
        If provided, only include files whose name ends with one of the extensions.
    """
    base_dir: Path = PATHS["data_folder"]
    target_dir: Path = base_dir / subdir if subdir else base_dir

    if not target_dir.exists():
        return []

    files: List[Dict[str, object]] = []
    for path in target_dir.rglob("*"):
        if path.is_file():
            if extensions and not any(path.name.endswith(ext) for ext in extensions):
                continue
            stat = path.stat()
            files.append({
                "name": path.name,
                "path": str(path),
                "timestamp": int(stat.st_ctime * 1000),
            })
    # newest first
    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return files


def read_data_file(rel_path: str) -> Dict[str, object]:
    """Return metadata and UTF-8 content for a single file under the data directory.

    Parameters
    ----------
    rel_path: str
        Path relative to the data folder.
    """
    base_dir: Path = PATHS["data_folder"].resolve()
    target: Path = (base_dir / rel_path).resolve()

    if base_dir not in target.parents and target != base_dir:
        raise ValueError("Invalid file path")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(rel_path)

    stat = target.stat()
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception:
        content = ""  

    return {
        "name": target.name,
        "path": str(target),
        "timestamp": int(stat.st_ctime * 1000),
        "content": content,
    }
