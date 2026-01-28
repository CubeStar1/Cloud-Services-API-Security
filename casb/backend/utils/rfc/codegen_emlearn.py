from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import joblib

try:
    from emlearn import convert
except ImportError:  
    convert = None 

from ..path_config import DATA_DIR


def _status(msg: str, sink: List[str] | None):
    if sink is not None:
        sink.append(msg)
    print(msg, flush=True)


def train_rfc_c_emlearn(model_dir: str | Path | None = None, logs: List[str] | None = None) -> Dict[str, Any]:
    """Convert trained RandomForest to C using emlearn (if installed)."""
    if convert is None:
        raise RuntimeError("emlearn package not installed. Add 'emlearn' to requirements.txt")

    model_dir = Path(model_dir) if model_dir else DATA_DIR / "models" / "rfc"
    out_dir = DATA_DIR / "output" / "rfc" / "em-codegen"
    include_dir = out_dir / "include"
    models_dir = out_dir / "models"
    for d in (out_dir, include_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    svc_model = joblib.load(model_dir / "service_classifier.joblib")
    svc_le = joblib.load(model_dir / "service_encoder.joblib")

    _status("Converting service classifier via emlearn", logs)
    portable_model = convert(svc_model)  

    c_path = out_dir / "rfc_em_inference.c"
    h_path = include_dir / "service_classifier_eml.h"
    c_path.write_text(portable_model.to_c())  
    h_path.write_text(portable_model.to_h())  

    joblib.dump(portable_model, models_dir / "service_classifier_eml.joblib")  

    label_map_path = out_dir / "label_mappings.json"
    import json
    json.dump({int(i): cls for i, cls in enumerate(svc_le.classes_)}, label_map_path.open("w", encoding="utf-8"), indent=2)

    return {
        "binary_path": str(c_path),
        "header": str(h_path),
        "label_map": str(label_map_path),
        "success": True,
    }
