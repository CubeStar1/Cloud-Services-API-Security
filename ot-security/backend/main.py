# main.py (Updated with Fusion Layer and Alerts Endpoint)
print("--- SCRIPT START ---")
import os
import json
import time
import uuid
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.console import Console
from netaddr import IPNetwork
import paho.mqtt.client as mqtt

import firebase_admin
from firebase_admin import credentials, firestore

from scan import AdvancedITOTScanner
# --- NEW: Import the Fusion Detector ---
from fusion_layer import ProductionFusionDetector

# --- Real-time Log Storage (circular buffer for last 100 logs per device) ---
DEVICE_LOGS = {
    'plc1': deque(maxlen=100),
    'plc2': deque(maxlen=100),
    'plc3': deque(maxlen=100),
    'hmi': deque(maxlen=100),
    'mqtt': deque(maxlen=100),
}
DEVICE_SENSOR_DATA = {
    'plc1': None,
    'plc2': None,
    'plc3': None,
}
MQTT_CONNECTED = False

# --- Configuration & In-Memory Storage ---
SCAN_TASKS = {}
API_CONSOLE = Console(color_system="truecolor")

# --- Model Paths (Corrected to root backend directory) ---
LOGBERT_MODEL_PATH = "logbert_ics_model"  # Will load logbert_ics_model.pt and logbert_ics_model_tokenizer.pkl
AE_MODEL_PATH = "AE-BATADAL-l5-cf2.5.h5"  # BATADAL autoencoder model

# --- NEW: Initialize Fusion Detector ---
try:
    fusion_detector = ProductionFusionDetector(
        logbert_model_path=LOGBERT_MODEL_PATH,
        ae_model_path=AE_MODEL_PATH,
        fusion_weights=(0.5, 0.5) # Equal weights for scanner fusion
    )
    API_CONSOLE.print("[bold green]✅ Fusion detector for scanner initialized successfully.[/bold green]")
except Exception as e:
    fusion_detector = None
    API_CONSOLE.print(f"[bold red]❌ Failed to initialize fusion detector: {e}[/bold red]")

# --- Initialize Firebase Admin SDK ---
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    API_CONSOLE.print("[bold green]✅ Firestore connection successful.[/bold green]")
except Exception as e:
    db = None
    API_CONSOLE.print(f"[bold red]❌ Firestore connection failed: {e}[/bold red]")
    API_CONSOLE.print("[yellow]Scan results will not be saved to the database.[/yellow]")

# --- MQTT Subscriber for Real-time OT Logs ---
def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    global MQTT_CONNECTED
    if rc == 0:
        MQTT_CONNECTED = True
        client.subscribe("ot/logs")
        API_CONSOLE.print("[bold green]✅ MQTT connected and subscribed to ot/logs[/bold green]")
    else:
        API_CONSOLE.print(f"[bold red]❌ MQTT connection failed with code {rc}[/bold red]")

def on_mqtt_message(client, userdata, msg):
    """Handle incoming log messages from OT devices."""
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        device = payload.get('device', 'unknown')
        log_message = payload.get('log_message', '')
        sensor_data = payload.get('sensor_data', None)
        
        # Store log message
        if device in DEVICE_LOGS:
            DEVICE_LOGS[device].append({
                'message': log_message,
                'timestamp': payload.get('timestamp', time.time()),
                'is_anomalous': payload.get('is_anomalous', False)
            })
        
        # Store sensor data for autoencoder (only plc1 has BATADAL data)
        if sensor_data and device in DEVICE_SENSOR_DATA:
            DEVICE_SENSOR_DATA[device] = sensor_data
            
    except Exception as e:
        pass  # Silently ignore parse errors

def start_mqtt_subscriber():
    """Start MQTT subscriber in background thread."""
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fastapi-backend")
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        mqtt_client.connect("mqtt", 1883, 60)
        mqtt_client.loop_forever()
    except Exception as e:
        API_CONSOLE.print(f"[yellow]⚠️ MQTT subscriber failed to start: {e}[/yellow]")

# Start MQTT subscriber in background thread
mqtt_thread = threading.Thread(target=start_mqtt_subscriber, daemon=True)
mqtt_thread.start()

# --- Initialize FastAPI ---
app = FastAPI(title="IT/OT Security Scan API")

# --- CORS Configuration (Keep as is) ---
# ... (your existing CORS middleware code)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
# ... (ScanRequest, ScanResponse, ScanResultItem, TaskStatusResponse are fine)
# --- NEW: Add a model for the fusion detection result ---
class FusionDetectionResult(BaseModel):
    device_ip: str
    is_anomaly: bool
    severity: str
    fused_score: float
    logbert_score: float
    ae_score: float
    detection_source: str

class ScanResultItem(BaseModel):
    ip: str
    mac: str
    vendor: str
    open_ports: list[int]
    ot_services: list[tuple[int, str]]
    it_services: list[tuple[int, str]]
    risk: str
    port_count: int
    # --- NEW: Add fusion result to the scan item ---
    fusion_detection: FusionDetectionResult | None = None

class ScanRequest(BaseModel):
    subnet: str
    scan_type: str = "2"


class ScanResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    timestamp: float
    results: list[ScanResultItem] | None = None
    duration_seconds: float | None = None

# --- Firestore Upload Function (Keep as is) ---
def upload_results_to_firestore(task_id: str, results: list[ScanResultItem]):
    # ... (your existing upload function)
    """Uploads each processed result item to a Firestore collection."""
    if not db:
        API_CONSOLE.print("[bold red]Cannot upload to Firestore: Database client is not available.[/bold red]")
        return

    API_CONSOLE.print(f"[bold blue]Task {task_id}: Uploading {len(results)} devices to Firestore...[/bold blue]")
    try:
        batch = db.batch()
        scans_collection = db.collection('scans')

        for item in results:
            # Pydantic's .model_dump() is perfect for converting the nested model to a dict
            data_to_upload = item.model_dump()
            data_to_upload['scan_task_id'] = task_id
            data_to_upload['timestamp'] = firestore.SERVER_TIMESTAMP

            doc_ref = scans_collection.document()
            batch.set(doc_ref, data_to_upload)

        batch.commit()
        API_CONSOLE.print(f"[bold green]Task {task_id}: Successfully uploaded results to Firestore.[/bold green]")

    except Exception as e:
        API_CONSOLE.print(f"[bold red]Task {task_id}: Failed to upload to Firestore. Error: {e}[/bold red]")


# --- Background Task Function (MODIFIED) ---
def run_scan_in_background(task_id: str, request_data: ScanRequest):
    """Executes the scan, runs results through the fusion detector, and saves."""
    API_CONSOLE.print(f"[bold yellow]Task {task_id}: Starting scan for {request_data.subnet}[/bold yellow]")
    start_time = time.time()

    # ... (subnet validation code is fine)
    try:
        IPNetwork(request_data.subnet)
    except:
        SCAN_TASKS[task_id]['status'] = 'failed'
        SCAN_TASKS[task_id]['error'] = "Invalid subnet format."
        return

    try:
        scanner = AdvancedITOTScanner()
        if request_data.scan_type == "1":
            scanner.ALL_PORTS = [21, 22, 23, 80, 443, 502, 102, 44818, 47808, 4840, 1883, 3389, 445]

        raw_results = scanner.run_comprehensive_scan(request_data.subnet)
        end_time = time.time()
        duration = end_time - start_time

        processed_results = []
        for r in raw_results:
            # ... (your existing risk calculation logic is fine)
            risk = "Low"
            ot_count = len(r['ot_services'])
            if ot_count > 3:
                risk = "Critical"
            elif ot_count > 1:
                risk = "High"
            elif ot_count > 0:
                risk = "Medium"

            # --- NEW: Run each scan result through the fusion detector ---
            fusion_output = None
            if fusion_detector:
                # The fusion layer expects a dict, so we convert `r`
                scan_result_dict = {
                    'ip': r['ip'],
                    'vendor': r['vendor'],
                    'all_services': {p: [s] for p, s in r['it_services'] + r['ot_services']},
                    'ports': r['ports']
                }
                # NOTE: We pass sensor_data=None here. In a real-world scenario, you would
                # query the device's Modbus registers here to get live sensor data.
                detection_result = fusion_detector.detect_from_scan(
                    scan_result=scan_result_dict,
                    sensor_data=None
                )
                fusion_output = FusionDetectionResult(**detection_result)

            processed_results.append(ScanResultItem(
                ip=r['ip'], mac=r['mac'], vendor=r['vendor'],
                open_ports=r['ports'],
                ot_services=r['ot_services'],
                it_services=r['it_services'],
                risk=risk,
                port_count=len(r['ports']),
                fusion_detection=fusion_output # Add the new data here
            ))

        if processed_results:
            upload_results_to_firestore(task_id, processed_results)

        SCAN_TASKS[task_id].update({
            'status': 'completed',
            'results': processed_results,
            'duration_seconds': duration
        })
        API_CONSOLE.print(f"[bold green]Task {task_id}: Scan completed in {duration:.2f} seconds[/bold green]")

    except Exception as e:
        API_CONSOLE.print(f"[bold red]Task {task_id}: Scan failed with error: {e}[/bold red]")
        SCAN_TASKS[task_id]['status'] = 'failed'
        SCAN_TASKS[task_id]['error'] = str(e)


# --- API Endpoints ---
@app.post("/api/scan/start", response_model=ScanResponse)
async def start_scan(request_data: ScanRequest, background_tasks: BackgroundTasks):
    # ... (this endpoint is fine as is)
    task_id = str(uuid.uuid4())
    SCAN_TASKS[task_id] = {
        'status': 'running',
        'timestamp': time.time(),
        'results': None,
        'duration_seconds': None
    }
    background_tasks.add_task(run_scan_in_background, task_id, request_data)
    return ScanResponse(
        task_id=task_id,
        status="running",
        message="Scan started successfully in the background. Use the status endpoint to check progress."
    )


@app.get("/api/scan/status/{task_id}", response_model=TaskStatusResponse)
async def get_scan_status(task_id: str):
    # ... (this endpoint is fine as is)
    task_info = SCAN_TASKS.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    response_data = {
        'task_id': task_id,
        'status': task_info['status'],
        'timestamp': task_info['timestamp'],
        'duration_seconds': task_info.get('duration_seconds')
    }
    if task_info['status'] == 'completed':
        response_data['results'] = task_info['results']
    elif task_info['status'] == 'failed':
        raise HTTPException(status_code=500, detail=f"Scan failed: {task_info.get('error', 'Unknown error')}")
    return response_data


# --- NEW: Endpoint to get real-time alerts from Firestore ---
@app.get("/api/alerts")
async def get_alerts():
    """
    Fetches the latest anomaly alerts from the 'alerts' collection in Firestore.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database service is not available.")
    try:
        alerts_ref = db.collection('alerts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50)
        docs = alerts_ref.stream()
        alerts = [doc.to_dict() for doc in docs]
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch alerts from database: {e}")


# --- NEW: Dedicated Anomaly Detection Endpoint ---
class AnomalyDetectionRequest(BaseModel):
    devices: list[dict] | None = None  # Optional: analyze specific devices

class AnomalyResult(BaseModel):
    device_id: str
    device_name: str
    device_ip: str
    status: str  # "normal" or "anomaly"
    confidence: float
    anomaly_type: str | None = None
    severity: str
    fused_score: float
    logbert_score: float
    ae_score: float
    detection_source: str
    logs: list[str]

@app.post("/api/anomaly/analyze")
async def analyze_anomalies(request: AnomalyDetectionRequest | None = None):
    """
    Run ML-based anomaly detection on OT devices using REAL-TIME logs from MQTT.
    Uses actual log messages collected from the simulator via MQTT subscriber.
    """
    # Default OT devices from Docker network
    default_devices = [
        {'id': 'plc1', 'name': 'PLC-01', 'ip': '172.20.0.21', 'vendor': 'Siemens S7-1200'},
        {'id': 'plc2', 'name': 'PLC-02', 'ip': '172.20.0.22', 'vendor': 'Allen-Bradley'},
        {'id': 'plc3', 'name': 'PLC-03', 'ip': '172.20.0.23', 'vendor': 'Schneider Modicon'},
        {'id': 'hmi', 'name': 'HMI Panel', 'ip': '172.20.0.24', 'vendor': 'Wonderware HMI'},
        {'id': 'mqtt', 'name': 'MQTT Broker', 'ip': '172.20.0.25', 'vendor': 'Eclipse Mosquitto'},
    ]
    
    devices_to_analyze = request.devices if request and request.devices else default_devices
    results = []
    
    for device in devices_to_analyze:
        device_id = device.get('id', device.get('ip', 'unknown'))
        device_name = device.get('name', device.get('vendor', 'Unknown Device'))
        device_ip = device.get('ip', '0.0.0.0')
        
        if fusion_detector:
            try:
                # Get REAL logs from MQTT buffer for this device
                device_logs_buffer = DEVICE_LOGS.get(device_id, deque())
                real_logs = [log['message'] for log in device_logs_buffer if log.get('message')]
                
                # Get sensor data if available (only plc1 has BATADAL data)
                sensor_data = DEVICE_SENSOR_DATA.get(device_id, None)
                sensor_array = np.array([sensor_data]) if sensor_data else None
                
                if real_logs:
                    # Use REAL logs from MQTT - use higher threshold for OT logs
                    detection = fusion_detector.detect_from_logs(
                        log_messages=real_logs[-20:],  # Use last 20 logs
                        sensor_data=sensor_array,
                        fusion_threshold=1.3  # Calibrated for OT log patterns
                    )
                    log_source = "mqtt_realtime"
                    logs_used = real_logs[-5:]  # Show last 5 logs in response
                else:
                    # No real logs - mark as "unknown" status for devices that don't send logs
                    # HMI and MQTT broker don't send logs to ot/logs topic
                    results.append(AnomalyResult(
                        device_id=device_id,
                        device_name=device_name,
                        device_ip=device_ip,
                        status="normal",  # Assume normal when no logs available
                        confidence=0.0,
                        anomaly_type=None,
                        severity="normal",
                        fused_score=0.0,
                        logbert_score=0.0,
                        ae_score=0.0,
                        detection_source="no_logs",
                        logs=[f"No real-time logs available for {device_id}. Device may not send logs to MQTT."]
                    ))
                    continue
                
                results.append(AnomalyResult(
                    device_id=device_id,
                    device_name=device_name,
                    device_ip=device_ip,
                    status="anomaly" if detection['is_anomaly'] else "normal",
                    confidence=detection['confidence'] * 100,
                    anomaly_type=detection['detection_source'] if detection['is_anomaly'] else None,
                    severity=detection['severity'],
                    fused_score=detection['fused_score'],
                    logbert_score=detection['logbert_score'],
                    ae_score=detection['ae_score'],
                    detection_source=f"{detection['detection_source']} ({log_source})",
                    logs=logs_used
                ))
            except Exception as e:
                API_CONSOLE.print(f"[bold red]Error analyzing {device_id}: {e}[/bold red]")
                results.append(AnomalyResult(
                    device_id=device_id,
                    device_name=device_name,
                    device_ip=device_ip,
                    status="normal",
                    confidence=0.0,
                    anomaly_type=None,
                    severity="low",
                    fused_score=0.0,
                    logbert_score=0.0,
                    ae_score=0.0,
                    detection_source="error",
                    logs=[f"Error during analysis: {str(e)}"]
                ))
        else:
            # Fusion detector not available - return mock/default response
            results.append(AnomalyResult(
                device_id=device_id,
                device_name=device_name,
                device_ip=device_ip,
                status="normal",
                confidence=95.0,
                anomaly_type=None,
                severity="low",
                fused_score=0.0,
                logbert_score=0.0,
                ae_score=0.0,
                detection_source="none",
                logs=[f"Fusion detector not initialized - using default response for {device_ip}"]
            ))
    
    return {"results": results, "total_devices": len(results), "anomalies_detected": sum(1 for r in results if r.status == "anomaly")}


@app.get("/api/anomaly/status")
async def get_anomaly_status():
    """Check if the fusion detector is available and ready."""
    if fusion_detector:
        status = fusion_detector.get_status()
        return {
            "fusion_detector_available": True,
            "logbert_available": status.get("logbert_loaded", False),
            "autoencoder_available": status.get("autoencoder_loaded", False),
            "logbert_weight": status.get("logbert_weight", 0),
            "ae_weight": status.get("ae_weight", 0),
        }
    return {
        "fusion_detector_available": False,
        "logbert_available": False,
        "autoencoder_available": False,
    }


@app.get("/")
def read_root():
    return {"message": "IT/OT Scan API is running. Access /docs for documentation."}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "IT/OT Scan API"}


if __name__ == "__main__":
    import uvicorn
    API_CONSOLE.print("[bold green]Starting FastAPI server on http://127.0.0.1:8000[/bold green]")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)