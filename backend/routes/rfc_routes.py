from __future__ import annotations

from fastapi import APIRouter, Query

from pathlib import Path
import pandas as pd
from ..utils.path_config import PATHS

from ..types.rfc import RfcTrainRequest, RfcInferenceRequest
from ..utils.rfc.python_train import train_rfc_python
from ..utils.rfc.codegen_manual import train_rfc_c_manual
from ..utils.rfc.codegen_emlearn import train_rfc_c_emlearn
from ..utils.rfc.python_inference import predict_rfc_python
from ..utils.rfc.python_inference import batch_predict_rfc_python_file
from ..utils.rfc.c_inference import predict_rfc_c, batch_predict_rfc_c_file

router = APIRouter()


@router.post("/train/python", summary="Train RFC models using Python")
async def train_python(request: RfcTrainRequest) -> dict[str, object]:
    logs: list[str] = []
    try:
        metrics = train_rfc_python(request.input_file, logs)
        return {"success": True, "output": logs, "metrics": metrics}
    except Exception as exc:
        return {"success": False, "error": str(exc), "output": logs}


@router.post("/train/c/manual", summary="Generate C code manually from RFC models")
async def train_c_manual() -> dict[str, object]:
    logs: list[str] = []
    try:
        result = train_rfc_c_manual(logs=logs)
        return {"success": True, "output": logs, **result}
    except Exception as exc:
        return {"success": False, "error": str(exc), "output": logs}


@router.post("/train/c/emlearn", summary="Generate C code using emlearn")
async def train_c_emlearn() -> dict[str, object]:
    logs: list[str] = []
    try:
        result = train_rfc_c_emlearn(logs=logs)
        return {"success": True, "output": logs, **result}
    except Exception as exc:
        return {"success": False, "error": str(exc), "output": logs}



@router.post("/inference/c", summary="Run inference using compiled RFC C classifier")
async def inference_c(
    request: RfcInferenceRequest | None = None,
    file: str | None = Query(None, description="Relative filename under data/output/rfc/test directory"),
) -> dict[str, object]:
    logs: list[str] = []
    try:
        if file:
            try:
                results, time = batch_predict_rfc_c_file(file, logs)
            except FileNotFoundError:
                return {"success": False, "error": "File not found", "output": logs}
            return {"success": True, "output": logs, "results": results, "time": time}
        else:
            if request is None:
                return {"success": False, "error": "Request body missing", "output": logs}
            result = predict_rfc_c(request.dict(), logs)
            return {"success": True, "output": logs, **result}
    except Exception as exc:
        return {"success": False, "error": str(exc), "output": logs}


@router.post("/inference/python", summary="Run inference using trained RFC Python models")
async def inference_python(
    request: RfcInferenceRequest | None = None,
    file: str | None = Query(None, description="Relative filename under data/output/rfc/test directory"),
) -> dict[str, object]:
    logs: list[str] = []
    try:
        if file:
           
            try:
                results, time = batch_predict_rfc_python_file(file, logs)
            except FileNotFoundError:
                return {"success": False, "error": "File not found", "output": logs}
            return {"success": True, "output": logs, "results": results, "time": time}
        else:
            if request is None:
                return {"success": False, "error": "Request body missing", "output": logs}
            result = predict_rfc_python(request.dict(), logs)
            return {"success": True, "output": logs, **result}
    except Exception as exc:
        return {"success": False, "error": str(exc), "output": logs}

