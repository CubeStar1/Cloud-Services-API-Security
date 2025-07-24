from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict, Any

from .path_config import PATHS


__all__ = [
    "convert_all_raw_json_to_csv",
]

def read_logs(log_file: Path) -> List[Dict[str, Any]]:
    with log_file.open("r", encoding="utf-8") as f:
        logs_lines = f.readlines()
    return [json.loads(line.strip(',\n')) for line in logs_lines if line.strip(',\n')]

def process_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    processed_logs = []
    for log in logs:
        processed_logs.append({
            'headers_Host': log.get('headers_Host') or 'none',
            'url': log.get('url') or 'none',
            'method': log.get('method') or 'UNKNOWN',
            'requestHeaders_Origin': log.get('requestHeaders_Origin') or 'none',
            'requestHeaders_Content_Type': log.get('requestHeaders_Content_Type') or 'none',
            'responseHeaders_Content_Type': log.get('responseHeaders_Content_Type') or 'none',
            'requestHeaders_Referer': log.get('requestHeaders_Referer') or 'none',
            'requestHeaders_Accept': log.get('requestHeaders_Accept') or 'none',
        })
    return processed_logs

def write_csv(processed_logs: List[Dict[str, str]], output_file: Path) -> None:
    headers = [
        'headers_Host', 'url', 'method', 'requestHeaders_Origin',
        'requestHeaders_Content_Type', 'responseHeaders_Content_Type',
        'requestHeaders_Referer', 'requestHeaders_Accept',
    ]

    if output_file.exists():
        output_file.unlink()

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for log in processed_logs:
            row = {key: log.get(key, '') for key in headers}
            writer.writerow(row)

def convert_all_raw_json_to_csv() -> List[Path]:
    """Convert every JSON file under PATHS['raw_json_folder'] to CSV.

    Returns a list of created CSV `Path`s.
    """
    raw_dir = Path(PATHS['raw_json_folder'])
    csv_dir = Path(PATHS['csv_folder'])
    csv_dir.mkdir(parents=True, exist_ok=True)

    created_csv_files: List[Path] = []

    for json_path in raw_dir.glob("*.json"):
        base_name = json_path.stem
        output_csv = csv_dir / f"{base_name}.csv"
        try:
            logs = read_logs(json_path)
            processed = process_logs(logs)
            write_csv(processed, output_csv)
            created_csv_files.append(output_csv)
        except Exception as e:
            # Log error but continue processing others
            print(f"[csv_utils] Error processing {json_path}: {e}")
    return created_csv_files
