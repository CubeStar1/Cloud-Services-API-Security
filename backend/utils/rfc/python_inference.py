from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import time
import pandas as pd
from sklearn.pipeline import Pipeline

from ..path_config import PATHS


def _status(msg: str, sink: List[str] | None = None) -> None:
    """Log status message to console and optional sink."""
    if sink is not None:
        sink.append(msg)
    print(msg, flush=True)


def load_rfc_models() -> Dict[str, Any]:
    """Load trained RFC models and encoders.
    
    Returns:
        Dictionary containing loaded models and encoders
        
    Raises:
        FileNotFoundError: If model files don't exist
    """
    models_dir = Path(PATHS["rfc_python_inference_models"]) 
    
    model_files = {
        "service_model": models_dir / "service_classifier.joblib",
        "activity_model": models_dir / "activity_classifier.joblib", 
        "service_encoder": models_dir / "service_encoder.joblib",
        "activity_encoder": models_dir / "activity_encoder.joblib"
    }
    
    # Check if all model files exist
    missing_files = [str(path) for path in model_files.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing model files: {missing_files}")
    
    # Load all models
    loaded_models = {}
    for key, path in model_files.items():
        loaded_models[key] = joblib.load(path)
    
    return loaded_models


def predict_rfc_python(
    request_data: Dict[str, Any], 
    logs: List[str] | None = None
) -> Dict[str, Any]:
    """Perform inference using trained RFC models.
    
    Args:
        request_data: Dictionary containing request features
        logs: Optional list to collect log messages
        
    Returns:
        Dictionary containing predictions and confidence scores
    """
    try:
        _status("Loading trained models...", logs)
        models = load_rfc_models()
        
        service_model: Pipeline = models["service_model"]
        activity_model: Pipeline = models["activity_model"]
        service_encoder = models["service_encoder"]
        activity_encoder = models["activity_encoder"]
        
        _status("Models loaded successfully", logs)
        
        # Extract and combine features from request data
        features = [
            request_data.get("headers_Host", ""),
            request_data.get("url", ""),
            request_data.get("method", ""),
            request_data.get("requestHeaders_Origin", ""),
            request_data.get("requestHeaders_Content_Type", ""),
            request_data.get("responseHeaders_Content_Type", ""),
            request_data.get("requestHeaders_Referer", ""),
            request_data.get("requestHeaders_Accept", "")
        ]
        
        # Combine features into a single string
        combined_features = " ".join(str(f) if f else "" for f in features)
        _status(f"Combined features: {combined_features[:100]}...", logs)
        
        # Make predictions
        _status("Making predictions...", logs)
        
        # Service prediction
        service_pred_encoded = service_model.predict([combined_features])[0]
        service_pred_proba = service_model.predict_proba([combined_features])[0]
        service_prediction = service_encoder.inverse_transform([service_pred_encoded])[0]
        service_confidence = float(max(service_pred_proba))
        
        # Activity prediction  
        activity_pred_encoded = activity_model.predict([combined_features])[0]
        activity_pred_proba = activity_model.predict_proba([combined_features])[0]
        activity_prediction = activity_encoder.inverse_transform([activity_pred_encoded])[0]
        activity_confidence = float(max(activity_pred_proba))
        
        _status(f"Service prediction: {service_prediction} (confidence: {service_confidence:.4f})", logs)
        _status(f"Activity prediction: {activity_prediction} (confidence: {activity_confidence:.4f})", logs)
        
        # Get all class probabilities for detailed analysis
        service_classes = service_encoder.classes_
        activity_classes = activity_encoder.classes_
        
        service_probabilities = {
            str(cls): float(prob) 
            for cls, prob in zip(service_classes, service_pred_proba)
        }
        
        activity_probabilities = {
            str(cls): float(prob) 
            for cls, prob in zip(activity_classes, activity_pred_proba)
        }
        
        return {
            "service_prediction": str(service_prediction),
            "service_confidence": service_confidence,
            "service_probabilities": service_probabilities,
            "activity_prediction": str(activity_prediction), 
            "activity_confidence": activity_confidence,
            "activity_probabilities": activity_probabilities,
            "combined_features": combined_features,
            "feature_count": len([f for f in features if f])
        }
        
    except FileNotFoundError as e:
        error_msg = f"Model files not found. Please train the models first. Error: {e}"
        _status(error_msg, logs)
        raise
    except Exception as e:
        error_msg = f"Inference error: {e}"
        _status(error_msg, logs)
        raise


def batch_predict_rfc_python(
    requests_data: List[Dict[str, Any]], 
    logs: List[str] | None = None
) -> Tuple[List[Dict[str, Any]], float]:
    """Perform batch inference using trained RFC models.
    
    Args:
        requests_data: List of dictionaries containing request features
        logs: Optional list to collect log messages
        
    Returns:
        List of dictionaries containing predictions for each request
    """
    try:
        _status(f"Starting batch inference for {len(requests_data)} requests...", logs)
        models = load_rfc_models()
        
        service_model: Pipeline = models["service_model"]
        activity_model: Pipeline = models["activity_model"]
        service_encoder = models["service_encoder"]
        activity_encoder = models["activity_encoder"]
        start_time = time.perf_counter()
        
        results = []
        
        for i, request_data in enumerate(requests_data):
            # _status(f"Processing request {i+1}/{len(requests_data)}", logs)
            
            # Extract and combine features
            features = [
                request_data.get("headers_Host", ""),
                request_data.get("url", ""),
                request_data.get("method", ""),
                request_data.get("requestHeaders_Origin", ""),
                request_data.get("requestHeaders_Content_Type", ""),
                request_data.get("responseHeaders_Content_Type", ""),
                request_data.get("requestHeaders_Referer", ""),
                request_data.get("requestHeaders_Accept", "")
            ]
            
            combined_features = " ".join(str(f) if f else "" for f in features)
            
            # Measure inference time
            iter_start = time.perf_counter()
            # Make predictions
            service_pred_encoded = service_model.predict([combined_features])[0]
            service_pred_proba = service_model.predict_proba([combined_features])[0]
            service_prediction = service_encoder.inverse_transform([service_pred_encoded])[0]
            service_confidence = round(float(max(service_pred_proba)), 2)
            
            activity_pred_encoded = activity_model.predict([combined_features])[0]
            activity_pred_proba = activity_model.predict_proba([combined_features])[0]
            activity_prediction = activity_encoder.inverse_transform([activity_pred_encoded])[0]
            activity_confidence = round(float(max(activity_pred_proba)), 2)
            iter_end_time = time.perf_counter()
            elapsed_ms = (iter_end_time - iter_start) * 1000.0
            
            _status(f"Processing request {i+1}/{len(requests_data)} [{elapsed_ms:.2f} ms] Service: {service_prediction} ({service_confidence:.2f}), "
                f"Activity: {activity_prediction} ({activity_confidence:.2f})",
                logs,
            )

            enriched = {
                **request_data,
                "service_prediction": str(service_prediction),
                "service_confidence": service_confidence,
                "activity_prediction": str(activity_prediction),
                "activity_confidence": activity_confidence,
            }
            results.append(enriched)

        total_time = time.perf_counter() - start_time 
        _status(f"Processed {len(results)} requests in {total_time:.2f} seconds", logs)
        return results, round(total_time, 2)
    except Exception as e:
        error_msg = f"Batch inference error: {e}"
        _status(error_msg, logs)
        raise



def batch_predict_rfc_python_file(
    filename: str,
    logs: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Load a CSV file located in the RFC test directory and run batch inference.

    Parameters
    ----------
    filename: str
        Filename relative to PATHS["rfc_python_train_test"].
    logs: list[str] | None
        Optional sink for status messages.
    """
    from pathlib import Path
    import pandas as pd

    test_dir = Path(PATHS["rfc_python_train_test"])
    csv_path = (test_dir / filename).resolve()

    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(filename)

    _status(f"Reading CSV file {csv_path}", logs)
    df = pd.read_csv(csv_path)
    requests = df.to_dict(orient="records")
    return batch_predict_rfc_python(requests, logs)
