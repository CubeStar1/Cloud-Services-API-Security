from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from ..path_config import PATHS


def _status(msg: str, sink: List[str] | None = None) -> None:
    if sink is not None:
        sink.append(msg)
    print(msg, flush=True)


def train_rfc_python(input_file: str, log: List[str] | None = None) -> Dict[str, Any]:
    """Train service & activity RFC models and persist them.

    Parameters
    ----------
    input_file: str
        CSV file name expected under data/output/codebert/predictions.
    log: list[str] | None
        Collects stdout-style messages.
    Returns
    -------
    dict with training metrics for frontend consumption.
    """
    paths = {
        "input": PATHS["rfc_python_train_input"],
        "models": PATHS["rfc_python_train_models"],
        "test": PATHS["rfc_python_train_test"],
    }
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    csv_path = paths["input"] / input_file
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    _status(f"Loading data from {csv_path}", log)

    df = pd.read_csv(csv_path)
    services = df["service"].astype(str).unique()
    activities = df["activityType"].astype(str).unique()

    _status(f"Found {len(services)} unique services and {len(activities)} unique activities", log)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    ts = datetime.now().strftime("%Y%m%d")
    test_file = paths["test"] / f"test_set_{ts}.csv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_file, index=False)
    _status(f"Test set saved to {test_file}", log)

    train_df["combined"] = (
        train_df[[
            "headers_Host",
            "url",
            "method",
            "requestHeaders_Origin",
            "requestHeaders_Content_Type",
            "responseHeaders_Content_Type",
            "requestHeaders_Referer",
            "requestHeaders_Accept",
        ]]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    svc_le = LabelEncoder().fit(services)
    act_le = LabelEncoder().fit(activities)
    train_df["service_encoded"] = svc_le.transform(train_df["service"].astype(str))
    train_df["activity_encoded"] = act_le.transform(train_df["activityType"].astype(str))

    X_train, X_val, y_train_svc, y_val_svc = train_test_split(
        train_df["combined"], train_df["service_encoded"], test_size=0.2, random_state=42
    )
    _, _, y_train_act, y_val_act = train_test_split(
        train_df["combined"], train_df["activity_encoded"], test_size=0.2, random_state=42
    )

    _status("Training service classifier...", log)
    svc_model = make_pipeline(
        TfidfVectorizer(max_features=5000),
        RandomForestClassifier(n_estimators=100, random_state=42),
    ).fit(X_train, y_train_svc)
    svc_acc = accuracy_score(y_val_svc, svc_model.predict(X_val))

    _status("Training activity classifier...", log)
    act_model = make_pipeline(
        TfidfVectorizer(max_features=5000),
        RandomForestClassifier(n_estimators=100, random_state=42),
    ).fit(X_train, y_train_act)
    act_acc = accuracy_score(y_val_act, act_model.predict(X_val))

    _status(f"Service Classification Accuracy: {svc_acc:.4f}", log)
    _status(f"Activity Classification Accuracy: {act_acc:.4f}", log)

    # Save artefacts
    joblib.dump(svc_model, Path(paths["models"]) / "service_classifier.joblib")
    joblib.dump(act_model, Path(paths["models"]) / "activity_classifier.joblib")
    joblib.dump(svc_le, Path(paths["models"]) / "service_encoder.joblib")
    joblib.dump(act_le, Path(paths["models"]) / "activity_encoder.joblib")
    _status("Models and encoders saved successfully", log)

    return {
        "service_accuracy": float(svc_acc),
        "activity_accuracy": float(act_acc),
        "unique_services": int(len(services)),
        "unique_activities": int(len(activities)),
    }
