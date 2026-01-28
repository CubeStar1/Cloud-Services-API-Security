"""
Real-time Fusion Layer for OT Security System
Integrates LogBERT (logs) + Autoencoder (sensors)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
# REMOVED: from tensorflow.keras.models import load_model as keras_load_model
from datetime import datetime

# Assuming logbert_ics_ot is in your environment
from logbert_ics_ot import LogBERTAnomalyDetector

# fusion_layer.py (Lines 11-20)
# No need to import detector module for inference
# Keras/TensorFlow will load the saved model architecture from .h5 file
CUSTOM_OBJECTS = {}
class ProductionFusionDetector:
    """Real-time fusion detector for scanner integration"""

    def __init__(self,
                 logbert_model_path: str,
                 ae_model_path: str,
                 ae_theta_path: Optional[str] = None,
                 fusion_weights: Tuple[float, float] = (0.6, 0.4)):

        print("Loading models...")

        # 🚨 FINAL FIX: Move Keras import inside __init__ to resolve module conflict 🚨
        keras_load_model = None
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError:
            print("— WARNING: TensorFlow not found. Autoencoder will be disabled.")

        # 1. Load LogBERT with error handling
        self.logbert = None
        if logbert_model_path:
            try:
                self.logbert = LogBERTAnomalyDetector()
                self.logbert.load_model(logbert_model_path)
                print("✓ LogBERT loaded successfully")
            except FileNotFoundError as e:
                print(f"— LogBERT skipped (file not found): {e}")
            except Exception as e:
                print(f"— LogBERT loading failed: {e}")
        else:
            print("— LogBERT skipped (Path is None)")

        # 2. Load Autoencoder with error handling
        self.autoencoder = None
        if ae_model_path and keras_load_model:
            try:
                # Load Autoencoder (compile=False for inference)
                self.autoencoder = keras_load_model(ae_model_path, compile=False, custom_objects=None)
                print("✓ Autoencoder loaded successfully")
            except FileNotFoundError as e:
                print(f"— Autoencoder skipped (file not found): {e}")
            except Exception as e:
                print(f"— Autoencoder loading failed: {e}")
        elif ae_model_path and not keras_load_model:
            print("— Autoencoder skipped (Keras/TensorFlow not available)")
        else:
            print("— Autoencoder skipped (Path is None)")

        # Weights
        self.logbert_weight = fusion_weights[0]
        self.ae_weight = fusion_weights[1]

        # Normalize weights (handle case where both weights might be 0 due to skipping/None paths)
        total = self.logbert_weight + self.ae_weight
        if total > 0:
            self.logbert_weight /= total
            self.ae_weight /= total
        else:
            self.logbert_weight = 0.5 # Default to 0.5 if paths were invalid but models aren't crashing
            self.ae_weight = 0.5

        print(f"✓ Fusion weights: LogBERT={self.logbert_weight:.2f}, AE={self.ae_weight:.2f}")
        
        # Print initialization summary
        logbert_status = "✓ Ready" if (self.logbert and self.logbert.model) else "✗ Not loaded"
        ae_status = "✓ Ready" if self.autoencoder else "✗ Not loaded"
        print(f"=== Fusion Detector Status ===")
        print(f"  LogBERT: {logbert_status}")
        print(f"  Autoencoder: {ae_status}")
        print(f"==============================")

    def get_status(self) -> Dict:
        """Return the status of loaded models"""
        return {
            "logbert_loaded": self.logbert is not None and self.logbert.model is not None,
            "autoencoder_loaded": self.autoencoder is not None,
            "logbert_weight": self.logbert_weight,
            "ae_weight": self.ae_weight
        }

    # 🚨 FIX APPLIED EARLIER: MOVED METHOD OUTSIDE OF __init__ 🚨
    def convert_scan_to_log(self, scan_result: Dict) -> List[str]:
        """Convert scanner result to ICS log entries"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        ip = scan_result.get('ip', 'unknown')
        hostname_alias = scan_result.get('vendor', 'Unknown_Device')

        logs = []
        all_services = scan_result.get('all_services', {})

        for port, protocols_list in all_services.items():
            protocol = protocols_list[0].strip('?') if protocols_list else 'UNKNOWN'
            try:
                port_int = int(port)
            except ValueError:
                port_int = None

            if 'modbus' in protocol.lower() or port_int == 502:
                log = f"{timestamp} MODBUS 172.20.0.10 -> {ip} Function: READ_HOLDING_REGISTERS Port:{port} Device: {hostname_alias}"
            elif 'mqtt' in protocol.lower() or port_int == 1883:
                log = f"{timestamp} MQTT Scanner -> {ip} Topic: /status Message: ping Port:{port} Device: {hostname_alias}"
            elif 'http' in protocol.lower() or 'https' in protocol.lower():
                log = f"{timestamp} HTTP GET {ip}:{port} /index.html Status: 200 Device: {hostname_alias}"
            else:
                log = f"{timestamp} SCAN 172.20.0.10 -> {ip} Port: {port} Protocol: {protocol} Device: {hostname_alias}"

            logs.append(log)

        if not logs and scan_result.get('ports'):
            open_ports = scan_result.get('ports', [])
            log = f"{timestamp} SCAN Target: {ip} Hostname: {hostname_alias} Open_Ports: {len(open_ports)} Status: COMPLETE_PORT_SCAN"
            logs.append(log)

        return logs

    def get_logbert_score(self, log_line: str) -> float:
        """Get normalized anomaly score from LogBERT"""
        if not self.logbert or not self.logbert.model:
            return 0.0

        try:
            result = self.logbert.detect_anomaly(log_line)
            raw_score = result['anomaly_score']
            threshold = result['threshold']
            # Normalize: score/threshold so 1.0 = at threshold
            return raw_score / threshold if threshold > 0 else raw_score
        except Exception as e:
            print(f"LogBERT scoring error: {e}")
            return 0.0

    def get_ae_score(self, sensor_data: np.ndarray) -> float:
        """Get normalized anomaly score from Autoencoder"""
        if not self.autoencoder:
            return 0.0

        try:
            if sensor_data.ndim == 1:
                sensor_data = sensor_data.reshape(1, -1)

            # Check if sensor data has the right shape (BATADAL expects 43 features)
            expected_features = self.autoencoder.input_shape[-1]
            if sensor_data.shape[-1] != expected_features:
                # Pad or truncate to match expected features
                if sensor_data.shape[-1] < expected_features:
                    # Pad with zeros
                    padded = np.zeros((sensor_data.shape[0], expected_features))
                    padded[:, :sensor_data.shape[-1]] = sensor_data
                    sensor_data = padded
                else:
                    # Truncate
                    sensor_data = sensor_data[:, :expected_features]

            reconstruction = self.autoencoder.predict(sensor_data, verbose=0)
            mse = np.mean((sensor_data - reconstruction) ** 2)

            # Normalize MSE to reasonable scale
            # For normalized 0-1 data: MSE < 0.02 is normal, > 0.05 is anomalous
            # We use 0.02 as baseline so: score ~1.0 for normal, >2.5 for anomaly
            normalized = mse / 0.02
            return float(min(normalized, 5.0))  # Cap at 5.0 to avoid extreme values
        except Exception as e:
            print(f"Autoencoder scoring error: {e}")
            return 0.0

    def detect_from_scan(self,
                         scan_result: Dict,
                         sensor_data: Optional[np.ndarray] = None,
                         fusion_threshold: float = 1.0) -> Dict:
        """
        Main detection function with graceful degradation
        """

        is_logbert_active = self.logbert is not None and self.logbert.model is not None
        is_ae_active = self.autoencoder is not None and sensor_data is not None

        # Generate logs and get LogBERT score
        if is_logbert_active:
            try:
                logs = self.convert_scan_to_log(scan_result)
                logbert_scores = [self.get_logbert_score(log) for log in logs]
                max_logbert = max(logbert_scores) if logbert_scores else 0.0
            except Exception as e:
                print(f"LogBERT detection error: {e}")
                logs = [f"LogBERT error: {e}"]
                max_logbert = 0.0
        else:
            logs = self.convert_scan_to_log(scan_result)  # Still generate logs for context
            if not self.logbert:
                logs.append("LogBERT model not loaded")
            max_logbert = 0.0
            max_logbert = 0.0

        if is_ae_active:
            ae_score = self.get_ae_score(sensor_data)
        else:
            ae_score = 0.0

        # Fusion Logic
        if is_logbert_active and is_ae_active:
            fused_score = (self.logbert_weight * max_logbert + self.ae_weight * ae_score)
        elif is_logbert_active:
            fused_score = max_logbert
        elif is_ae_active:
            fused_score = ae_score
        else:
            fused_score = 0.0

        # Anomaly Determination
        is_anomaly = fused_score >= fusion_threshold

        if fused_score >= 1.5:
            severity = "critical"
        elif fused_score >= 1.2:
            severity = "high"
        elif fused_score >= 1.0:
            severity = "medium"
        else:
            severity = "low"

        log_is_anom = max_logbert >= 1.0
        ae_is_anom = ae_score >= 1.0

        if log_is_anom and ae_is_anom:
            source = "both"
        elif log_is_anom:
            source = "log_based"
        elif ae_is_anom:
            source = "sensor_based"
        else:
            source = "none"

        return {
            'device_ip': scan_result.get('ip', 'unknown'),
            'device_hostname': scan_result.get('vendor', 'Unknown_Device'),
            'device_type': scan_result.get('type', 'OT/IT'),
            'is_anomaly': is_anomaly,
            'severity': severity,
            'fused_score': round(fused_score, 4),
            'logbert_score': round(max_logbert, 4),
            'ae_score': round(ae_score, 4),
            'threshold': fusion_threshold,
            'confidence': round(abs(fused_score - fusion_threshold) / (fusion_threshold + 1e-6), 2),
            'detection_source': source,
            'has_sensor_data': sensor_data is not None,
            'generated_logs': logs,
            'timestamp': datetime.now().isoformat()
        }

    def detect_from_logs(self, log_messages: List[str], sensor_data: Optional[np.ndarray] = None, fusion_threshold: float = 1.0) -> Dict:
        """
        Analyze actual log messages (not synthetic ones).
        This is the method that should be used for real-time analysis.
        """
        is_logbert_active = self.logbert is not None and self.logbert.model is not None
        is_ae_active = self.autoencoder is not None and sensor_data is not None

        # Analyze actual logs with LogBERT
        if is_logbert_active and log_messages:
            try:
                logbert_scores = []
                for log in log_messages:
                    score = self.get_logbert_score(log)
                    logbert_scores.append(score)
                max_logbert = max(logbert_scores) if logbert_scores else 0.0
                avg_logbert = sum(logbert_scores) / len(logbert_scores) if logbert_scores else 0.0
            except Exception as e:
                print(f"LogBERT detection error: {e}")
                max_logbert = 0.0
                avg_logbert = 0.0
        else:
            max_logbert = 0.0
            avg_logbert = 0.0

        # Autoencoder score
        if is_ae_active:
            ae_score = self.get_ae_score(sensor_data)
        else:
            ae_score = 0.0

        # Fusion Logic - use average for real logs to be more accurate
        if is_logbert_active and is_ae_active:
            fused_score = (self.logbert_weight * avg_logbert + self.ae_weight * ae_score)
        elif is_logbert_active:
            fused_score = avg_logbert
        elif is_ae_active:
            fused_score = ae_score
        else:
            fused_score = 0.0

        # Anomaly Determination - use higher threshold for real logs since they may differ from training
        # Scores below 1.3 for LogBERT are considered normal for OT logs
        is_anomaly = fused_score >= fusion_threshold

        # Severity based on score - calibrated for real OT logs
        # Scores 1.0-1.3: Normal operation (OT logs differ from training data)
        # Scores 1.3-1.5: Low severity, slight deviation
        # Scores 1.5-2.0: Medium severity, notable deviation  
        # Scores 2.0-2.5: High severity, significant anomaly
        # Scores 2.5+: Critical, likely attack or major malfunction
        if fused_score >= 2.5:
            severity = "critical"
        elif fused_score >= 2.0:
            severity = "high"
        elif fused_score >= 1.5:
            severity = "medium"
        elif fused_score >= 1.3:
            severity = "low"
        else:
            severity = "normal"

        log_is_anom = avg_logbert >= 1.3  # Adjusted threshold for OT logs
        ae_is_anom = ae_score >= 1.5  # AE threshold for sensor data

        if log_is_anom and ae_is_anom:
            source = "both"
        elif log_is_anom:
            source = "log_based"
        elif ae_is_anom:
            source = "sensor_based"
        else:
            source = "none"

        return {
            'is_anomaly': is_anomaly,
            'severity': severity,
            'fused_score': round(fused_score, 4),
            'logbert_score': round(avg_logbert, 4),
            'logbert_max_score': round(max_logbert, 4),
            'ae_score': round(ae_score, 4),
            'threshold': fusion_threshold,
            'confidence': round(abs(fused_score - fusion_threshold) / (fusion_threshold + 1e-6), 2),
            'detection_source': source,
            'logs_analyzed': len(log_messages),
            'has_sensor_data': sensor_data is not None,
            'timestamp': datetime.now().isoformat()
        }
