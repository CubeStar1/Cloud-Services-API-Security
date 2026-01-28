#!/usr/bin/env python3
"""
A high-accuracy protocol detection module for specific OT services.
Detects Modbus, MQTT, and HTTP services using native client libraries.
"""
import random
import time
import requests
from pymodbus.client.sync import ModbusTcpClient
import paho.mqtt.client as mqtt_client


# --- DETECTION FUNCTIONS ---

def check_modbus(host: str, port: int) -> bool:
    """Detects Modbus/TCP by connecting and attempting a read."""
    client = ModbusTcpClient(host, port, timeout=2)
    try:
        if not client.connect():
            return False
        # A successful read attempt confirms the protocol.
        client.read_holding_registers(address=0, count=1, unit=1)
        return True
    except Exception:
        return False
    finally:
        if client.is_socket_open():
            client.close()


# --- MQTT Detection Helper ---
mqtt_connection_result = {"status": None}


def on_connect_v2(client, userdata, flags, reason_code, properties):
    """Callback to confirm successful MQTT connection."""
    mqtt_connection_result["status"] = True


def check_mqtt(host: str, port: int) -> bool:
    """Detects an MQTT broker by attempting a client connection."""
    mqtt_connection_result["status"] = None
    client_id = f'detector-{random.randint(0, 1000)}'
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.on_connect = on_connect_v2
    try:
        client.connect(host, port, 60)
        client.loop_start()
        time.sleep(1)  # Allow time for connection
        client.loop_stop()
        client.disconnect()
        return mqtt_connection_result["status"] or False
    except Exception:
        return False


def check_http(host: str, port: int) -> bool:
    """Detects an HTTP service by sending a GET request."""
    url = f"http://{host}:{port}"
    try:
        response = requests.get(url, timeout=2, headers={'User-Agent': 'OT-Scanner'})
        return response.ok
    except requests.exceptions.RequestException:
        return False


# --- MAIN EXECUTION (for standalone testing) ---
if __name__ == "__main__":
    print("--- Running Standalone Module Test ---")

    # Test targets (replace with actual IPs if needed)
    MODBUS_HOST = 'localhost'
    MQTT_HOST = 'localhost'
    HMI_HOST = 'localhost'

    detections = {
        'Modbus (port 502)': check_modbus(MODBUS_HOST, 502),
        'MQTT (port 1883)': check_mqtt(MQTT_HOST, 1883),
        'HMI (port 8085)': check_http(HMI_HOST, 8085),
    }

    print("\n--- Detection Summary ---")
    for device, status in detections.items():
        result = "✅ Detected" if status else "❌ Not Found"
        print(f"{device:<20} | {result}")
    print("\n--- Test Complete ---")