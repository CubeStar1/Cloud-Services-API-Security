
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import subprocess
import sys
import time

from ..path_config import PATHS



def _status(msg: str, sink: List[str] | None = None) -> None:  
    if sink is not None:
        sink.append(msg)
    print(msg, flush=True)



OUTPUT_DIR = Path(PATHS["rfc_codegen_output_folder"]).resolve()
EXECUTABLE_WIN = OUTPUT_DIR / "api_classifier.exe"
EXECUTABLE_NIX = OUTPUT_DIR / "api_classifier" 
LABEL_MAP_FILE = OUTPUT_DIR / "label_mappings.txt"

_service_map: Dict[int, str] | None = None
_activity_map: Dict[int, str] | None = None

def _load_label_maps() -> None:
    global _service_map, _activity_map
    if _service_map is not None and _activity_map is not None:
        return  

    _service_map, _activity_map = {}, {}
    if not LABEL_MAP_FILE.exists():
        return

    current_kind: str | None = None
    with LABEL_MAP_FILE.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if "service" in lower and "mapping" in lower:
                current_kind = "service"
                continue
            if "activity" in lower and "mapping" in lower:
                current_kind = "activity"
                continue

            parts = line.split("\t")
            if len(parts) == 3:
                kind, idx_str, label = parts
            elif len(parts) == 2 and current_kind is not None:  
                if ":" in parts[0]:
                    idx_str, label = parts[0].split(":", 1)
                else:
                    idx_str, label = parts[0], parts[1]
                kind = current_kind
            elif ":" in line and current_kind is not None: 
                idx_str, label = line.split(":", 1)
                kind = current_kind
            else:
                continue 

            try:
                idx = int(idx_str.strip())
            except ValueError:
                continue
            label = label.strip()
            if kind == "service":
                _service_map[idx] = label
            elif kind == "activity":
                _activity_map[idx] = label

def _get_executable() -> Path:
    """Return path to classifier executable, raising helpful error if not present."""
    if sys.platform.startswith("win"):
        exe = EXECUTABLE_WIN
    else:
        exe = EXECUTABLE_NIX
    if not exe.exists():
        raise FileNotFoundError(
            f"Classifier executable not found at {exe}. Please compile the C code first." )
    return exe


def _build_args_from_request(request: Dict[str, Any]) -> List[str]:
    """Convert request dict into CLI argument list expected by C binary.

    Ensures each value is a *string* (not None, NaN, etc.) so that the executed
    command always supplies exactly 8 arguments.
    """
    import math 

    def _safe(val: Any) -> str:
        """Sanitize a single field for safe CLI passing."""
        if val is None:
            return "-"
        if isinstance(val, float) and math.isnan(val):
            return "-"
        s = str(val)
        if not s:
            return "-"
        s = s.replace("\\", "\\\\")  
        s = s.replace('"', r'\"')
        s = s.replace("charset=UTF-8", 'charset=\\"UTF-8\\"')
        s = s.replace(" ", "%20")
        s = s.replace(" ", "%20")
        return s

    return [
        _safe(request.get("headers_Host")),
        _safe(request.get("url")),
        _safe(request.get("method")),
        _safe(request.get("requestHeaders_Origin")),
        _safe(request.get("requestHeaders_Content_Type")),
        _safe(request.get("responseHeaders_Content_Type")),
        _safe(request.get("requestHeaders_Referer")),
        _safe(request.get("requestHeaders_Accept")),
    ]


def predict_rfc_c(request_data: Dict[str, Any], logs: List[str] | None = None) -> Dict[str, Any]:
    """Run a single inference through the compiled C classifier.

    Returns a dict containing the *raw* ids and, if mapping available, the
    decoded labels.
    """
    exe = _get_executable()
    args_payload = _build_args_from_request(request_data)
    if len(args_payload) != 8:
        raise RuntimeError(f"Internal error: expected 8 args, got {len(args_payload)} – {args_payload}")
    quoted_payload = [f'"{a}"' for a in args_payload] if sys.platform.startswith("win") else args_payload
    args = [str(exe), *quoted_payload]

    _status(f"Args payload: {args_payload}", logs)
    _status(f"Running C classifier (argc={len(args)}): {' '.join(args[:2])} …", logs)
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"C classifier failed with code {e.returncode}: {e.stderr.strip()}" ) from e

    stdout_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    parsed: dict[str, Any] | None = None
    for ln in stdout_lines:
        try:
            parsed = json.loads(ln)
            _status(f"C output (parsed): {ln}", logs)
            break
        except json.JSONDecodeError:
            _status(f"C output (ignored): {ln}", logs)
            continue

    if parsed is None:
        raise ValueError(f"Could not parse classifier output: {stdout_lines[-1] if stdout_lines else '<empty>'}")
    out = parsed

    service_id = int(out.get("service_id", -1))
    activity_id = int(out.get("activity_id", -1))

    _load_label_maps()
    service_label = _service_map.get(service_id) if _service_map else None
    activity_label = _activity_map.get(activity_id) if _activity_map else None

    result = {
        "service_id": service_id,
        "activity_id": activity_id,
        "service": service_label,
        "activity": activity_label,
    }
    return result


def batch_predict_rfc_c(requests_data: List[Dict[str, Any]], logs: List[str] | None = None) -> Tuple[List[Dict[str, Any]], float]:
    """Run inference for a list of requests."""
    results: List[Dict[str, Any]] = []
    total_requests = len(requests_data)
    start_time = time.perf_counter()
    
    for i, req in enumerate(requests_data, 1):
        iter_start = time.perf_counter()
        res = predict_rfc_c(req, logs)
        results.append({**req, **res})
        
        iter_time = (time.perf_counter() - iter_start) * 1000  
        
        service = res.get('service', 'Unknown Service')
        activity = res.get('activity', 'Unknown Activity')
        
        status_msg = (
            f"Processing request {i}/{total_requests} "
            f"[{iter_time:.2f} ms] "
            f"Service: {service}, "
            f"Activity: {activity}"
        )
        _status(status_msg, logs)
    
    total_time = time.perf_counter() - start_time
    _status(f"Processed {total_requests} requests in {total_time:.2f} seconds", logs)
    
    return results, round(total_time, 2)


def batch_predict_rfc_c_file(filename: str, logs: List[str] | None = None) -> Tuple[List[Dict[str, Any]], float]:
    """Load CSV file located under test directory and perform batch inference."""
    import pandas as pd

    test_dir = Path(PATHS["rfc_python_train_test"]).resolve()
    csv_path = (test_dir / filename).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(filename)

    df = pd.read_csv(csv_path)
    requests = df.to_dict(orient="records")
    return batch_predict_rfc_c(requests, logs)
