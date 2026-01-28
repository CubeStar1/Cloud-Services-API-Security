# detector.py
import os
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from rich.console import Console

# --- Firebase Admin SDK for Firestore ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- Import your Fusion Detector ---
from fusion_layer import ProductionFusionDetector

# --- Configuration ---
CONSOLE = Console(color_system="truecolor")

# --- Model Paths (Corrected to root backend directory) ---
# Files: logbert_ics_model.pt, logbert_ics_model_tokenizer.pkl, AE-BATADAL-l5-cf2.5.h5
LOGBERT_MODEL_PATH = "logbert_ics_model"  # Base path without .pt extension
AE_MODEL_PATH = "AE-BATADAL-l5-cf2.5.h5"  # BATADAL autoencoder model

# --- MQTT Configuration ---
MQTT_BROKER = "mqtt" # Use the service name from docker-compose
MQTT_PORT = 1883
MQTT_LOG_TOPIC = "ot/logs"

# --- Firestore Configuration ---
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    ALERTS_COLLECTION = db.collection('alerts')
    CONSOLE.print("[bold green]✅ Firestore connection successful.[/bold green]")
except Exception as e:
    db = None
    ALERTS_COLLECTION = None
    CONSOLE.print(f"[bold red]❌ Firestore connection failed: {e}[/bold red]")
    CONSOLE.print("[yellow]Anomaly alerts will not be saved.[/yellow]")


# --- Load Models using the Fusion Layer ---
CONSOLE.print("[bold yellow]Loading ML models via Fusion Layer...[/bold yellow]")
try:
    fusion_detector = ProductionFusionDetector(
        logbert_model_path=LOGBERT_MODEL_PATH,
        ae_model_path=AE_MODEL_PATH,
        fusion_weights=(0.7, 0.3) # Prioritize log analysis for this service
    )
    CONSOLE.print("[bold green]✅ Fusion detector initialized successfully.[/bold green]")
except Exception as e:
    fusion_detector = None
    CONSOLE.print(f"[bold red]❌ Failed to initialize fusion detector: {e}[/bold red]")
    exit() # Exit if models can't be loaded


# --- MQTT Callback Functions ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        CONSOLE.print(f"[bold green]Connected to MQTT Broker. Subscribing to topic: '{MQTT_LOG_TOPIC}'[/bold green]")
        client.subscribe(MQTT_LOG_TOPIC)
    else:
        CONSOLE.print(f"[bold red]Failed to connect to MQTT, return code {rc}[/bold red]")

def on_message(client, userdata, msg):
    """This function is called every time a log is received from the simulator."""
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        log_message = payload.get("log_message", "")
        device = payload.get("device", "unknown_device")
        
        # Extract sensor data if available (from PLC1 BATADAL stream)
        sensor_data_list = payload.get("sensor_data", None)
        sensor_data = np.array(sensor_data_list, dtype=np.float32) if sensor_data_list else None

        # The fusion layer expects a "scan_result" dict. We create a simplified
        # version of it from the log data.
        fake_scan_result = {
            'ip': device,
            'vendor': f"{device.upper()}_Device",
            'type': 'OT',
            'all_services': {},  # Empty as we are processing logs, not scans
            'ports': [502]  # Modbus port
        }

        # Pass sensor data to fusion detector for Autoencoder scoring
        detection_result = fusion_detector.detect_from_scan(
            scan_result=fake_scan_result,
            sensor_data=sensor_data,  # Pass BATADAL sensor data for AE
            fusion_threshold=1.0  # Standard threshold
        )

        # Print the result to the console
        score = detection_result['fused_score']
        is_anomaly = detection_result['is_anomaly']
        color = "red" if is_anomaly else "green"
        CONSOLE.print(f"[{color}]Log from [bold]{device}[/bold] | Fused Score: {score:.4f} | Anomaly: {is_anomaly}[/{color}]")

        # If it's an anomaly, save it to Firestore
        if is_anomaly and ALERTS_COLLECTION:
            try:
                # Add a server-side timestamp for ordering
                detection_result['timestamp'] = firestore.SERVER_TIMESTAMP
                ALERTS_COLLECTION.add(detection_result)
                CONSOLE.print(f"[bold blue]   -> Anomaly detected. Saved to Firestore alerts collection.[/bold blue]")
            except Exception as e:
                CONSOLE.print(f"[bold red]   -> Error saving alert to Firestore: {e}[/bold red]")


    except json.JSONDecodeError:
        CONSOLE.print("[bold red]Received non-JSON MQTT message.[/bold red]")
    except Exception as e:
        CONSOLE.print(f"[bold red]An error occurred in on_message: {e}[/bold red]")


# --- Main Execution ---
if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="detector-service")
    client.on_connect = on_connect
    client.on_message = on_message

    CONSOLE.print("[bold yellow]Attempting to connect to MQTT broker...[/bold yellow]")
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            CONSOLE.print(f"Waiting for MQTT broker at {MQTT_BROKER}... ({e})")
            time.sleep(5)

    # Loop forever to process messages
    client.loop_forever()