#!/usr/bin/env python3
import threading
import time
import random
import json
from pymodbus.client import ModbusTcpClient
from paho.mqtt.client import Client, CallbackAPIVersion

# --- Configuration ---
# NOTE: All PLCs must use the corrected internal port 5020.
PLC_CONFIG = {
    "plc1": {"host": "plc1", "port": 5020},
    "plc2": {"host": "plc2", "port": 5020},
    "plc3": {"host": "plc3", "port": 5020},
}

MQTT_BROKER = "mqtt"
MQTT_PORT = 1883
# CRITICAL: This topic must match the one used by detector.py
MQTT_LOG_TOPIC = "ot/logs"
# FIX: Adjusted scale factor from 1000 to 100. Max value (450) * 100 = 45000,
# which fits in a 16-bit register (max 65535).
SCALE_FACTOR = 100

# --- BATADAL Feature Set (43 Features) ---
# This feature set is used ONLY by plc1, the attacked system.
BATADAL_COLUMNS = [
    'L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
    'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4',
    'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8',
    'F_PU9', 'S_PU9', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11',
    'F_V2', 'S_V2',
    'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302',
    'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422'
]
NUM_BATADAL_COLUMNS = len(BATADAL_COLUMNS)

# --- BATADAL Normalization Parameters (from training data) ---
# These are approximate min/max values for normalizing sensor data to 0-1 range
# so the autoencoder can properly evaluate it
BATADAL_NORMS = {
    'L_T': (0.0, 7.0),      # Tank levels
    'F_PU': (0.0, 12.0),    # Pump flows
    'S_PU': (0.0, 1.0),     # Pump status (binary)
    'F_V': (0.0, 6.0),      # Valve flow
    'S_V': (0.0, 1.0),      # Valve status (binary)
    'P_J': (100.0, 500.0),  # Junction pressures
}

def normalize_batadal_data(values_float):
    """Normalize BATADAL values to 0-1 range for autoencoder."""
    normalized = []
    idx = 0
    
    # Tanks (L_T1 to L_T7): 7 values
    for i in range(7):
        min_v, max_v = BATADAL_NORMS['L_T']
        normalized.append((values_float[idx] - min_v) / (max_v - min_v))
        idx += 1
    
    # Pumps (F_PU* and S_PU*): 11 pairs = 22 values
    for i in range(11):
        min_v, max_v = BATADAL_NORMS['F_PU']
        normalized.append((values_float[idx] - min_v) / (max_v - min_v))
        idx += 1
        min_v, max_v = BATADAL_NORMS['S_PU']
        normalized.append((values_float[idx] - min_v) / (max_v - min_v))
        idx += 1
    
    # Valve (F_V2 and S_V2): 2 values
    min_v, max_v = BATADAL_NORMS['F_V']
    normalized.append((values_float[idx] - min_v) / (max_v - min_v))
    idx += 1
    min_v, max_v = BATADAL_NORMS['S_V']
    normalized.append((values_float[idx] - min_v) / (max_v - min_v))
    idx += 1
    
    # Junction Pressures (P_J*): 12 values
    for i in range(12):
        min_v, max_v = BATADAL_NORMS['P_J']
        normalized.append((values_float[idx] - min_v) / (max_v - min_v))
        idx += 1
    
    # Clip to 0-1 range
    return [max(0.0, min(1.0, v)) for v in normalized]


# ----------------------------------------------------
# ⚠️ ATTACKED SYSTEM (PLC1) - TRANSMITS BATADAL DATA
# ----------------------------------------------------
def simulate_anomalous_stream_plc1(host, port):
    """
    Simulates the attacked PLC (plc1) data stream: generates 43 scaled values,
    writes them to Modbus, and publishes a corresponding log message.
    
    Attack Pattern: Random attack cycles (30-90 sec normal, then 20-60 sec attack)
    """
    plc_name = "plc1"
    print(f"✅ [{plc_name}] BATADAL Streamer (Attack Simulation) started for {host}:{port}.")

    # --- Dynamic Attack Simulation ---
    # Attack cycles: Normal for 30-90 sec, then Attack for 20-60 sec, repeat
    is_anomalous = False
    cycle_end_time = time.time() + random.uniform(30, 90)  # Start with normal period
    attack_types = ["pressure_drop", "flow_spike", "multi_sensor"]
    current_attack = None

    while True:
        try:
            client = ModbusTcpClient(host, port=port, timeout=2)
            if not client.connect():
                print(f"⏳ [{plc_name}] Waiting for Modbus server {host}...")
                time.sleep(3)
                continue

            # Check if we should switch attack state
            if time.time() > cycle_end_time:
                is_anomalous = not is_anomalous
                if is_anomalous:
                    # Start attack period (20-60 seconds)
                    cycle_end_time = time.time() + random.uniform(20, 60)
                    current_attack = random.choice(attack_types)
                    print(f"🚨 [{plc_name}] ATTACK STARTED: {current_attack} (Duration: {int(cycle_end_time - time.time())}s)")
                else:
                    # Start normal period (30-90 seconds)
                    cycle_end_time = time.time() + random.uniform(30, 90)
                    current_attack = None
                    print(f"✅ [{plc_name}] Attack ended. Returning to NORMAL operation.")

            # 1. GENERATE BATADAL DATA (43 features)
            values_float = []

            # Tanks (L_T1 to L_T7): 0 to 5.0
            for i in range(7):
                if is_anomalous and current_attack == "multi_sensor" and i < 3:
                    # Attack: Tanks draining unexpectedly
                    values_float.append(random.uniform(0.0, 0.5))
                else:
                    values_float.append(random.uniform(2.0, 4.5))

            # Pumps (F_PU* and S_PU*): Flow (0 to 10.0), Status (0 or 1)
            for i in range(11):
                if is_anomalous and current_attack == "flow_spike" and i < 4:
                    # Attack: Abnormal flow rates
                    values_float.append(random.uniform(15.0, 25.0))  # Way above normal
                else:
                    values_float.append(random.uniform(3.0, 7.0))
                values_float.append(random.choice([0.0, 1.0]))

            # Valve (F_V2 and S_V2): Flow (0 to 5.0), Status (0 or 1)
            values_float.append(random.uniform(1.0, 4.0))
            values_float.append(random.choice([0.0, 1.0]))

            # Junction Pressures (P_J*): Normal 200-450, Attack varies
            for i in range(12):
                if is_anomalous and current_attack == "pressure_drop":
                    # Attack: Pressure drop (could indicate leak or valve issue)
                    values_float.append(random.uniform(50.0, 120.0))
                elif is_anomalous and current_attack == "multi_sensor" and i < 5:
                    # Attack: Some pressures abnormal
                    values_float.append(random.uniform(80.0, 150.0))
                else:
                    values_float.append(random.uniform(250.0, 400.0))

            # 2. SCALE AND CONVERT TO INT
            values_int = [int(v * SCALE_FACTOR) for v in values_float]

            # 3. WRITE TO MODBUS
            client.write_registers(0, values_int)

            # 4. PUBLISH LOG MESSAGE WITH SENSOR DATA
            if is_anomalous:
                log_type = f"ATTACK:{current_attack.upper()}"
            else:
                log_type = "NORMAL"

            log_message = f"[{log_type}][BATADAL] Write registers to {plc_name}. Count: {NUM_BATADAL_COLUMNS}. P_J300={values_float[34]:.2f}"

            # Normalize sensor data for autoencoder (0-1 range)
            normalized_data = normalize_batadal_data(values_float)
            
            log_payload = {
                "log_message": log_message,
                "device": plc_name,
                "timestamp": time.time(),
                "sensor_data": normalized_data,  # NORMALIZED data for Autoencoder
                "raw_sensor_data": values_float,  # Raw values for reference
                "is_anomalous": is_anomalous,
                "attack_type": current_attack,
            }

            mqtt_client.publish(MQTT_LOG_TOPIC, json.dumps(log_payload))

            status_icon = "🚨" if is_anomalous else "✅"
            print(f"{status_icon} [{plc_name}][DATA] {log_type}. P_J300: {values_float[34]:.2f}")
            client.close()

        except Exception as e:
            print(f"  [MODBUS] Exception in {plc_name}: {e}")

        time.sleep(random.uniform(1.0, 2.0))


# ----------------------------------------------------
# ✅ NORMAL SYSTEMS (PLC2 & PLC3) - MOSTLY NORMAL, RARE ISSUES
# ----------------------------------------------------
def simulate_normal_log_plc(plc_name, host, port):
    """
    Simulates a normal PLC (plc2/plc3) that mostly sends normal logs,
    but occasionally has minor issues (realistic behavior).
    """
    print(f"✅ [{plc_name}] Normal Log Streamer started for {host}:{port}.")

    # Normal operation logs (95% of the time)
    normal_logs = [
        "System check complete. Status: OK.",
        "Modbus client read successful.",
        "Heartbeat signal transmitted.",
        "Configuration file loaded.",
        "Sensor initialization successful.",
        "Routine maintenance check passed.",
        "Communication link verified.",
        "Register values within normal range.",
    ]
    
    # Occasional warning/issue logs (5% of the time) - not attacks, just normal issues
    warning_logs = [
        "WARNING: Sensor calibration due in 24 hours.",
        "NOTICE: Memory usage at 75%.",
        "WARNING: Communication latency elevated.",
        "NOTICE: Backup power test scheduled.",
    ]

    while True:
        try:
            client = ModbusTcpClient(host, port=port, timeout=2)
            if not client.connect():
                time.sleep(3)
                continue

            # 1. WRITE DUMMY NORMAL DATA (3 registers: Temp, Pressure, Vibe)
            temp = round(random.uniform(20.0, 80.0), 2)
            pressure = round(random.uniform(1.0, 10.0), 2)
            vibration = round(random.uniform(0.1, 5.0), 2)

            data_int = [
                int(temp * SCALE_FACTOR),
                int(pressure * SCALE_FACTOR),
                int(vibration * SCALE_FACTOR)
            ]
            client.write_registers(0, data_int)

            # 2. PUBLISH LOG MESSAGE (95% normal, 5% warning)
            if random.random() < 0.05:
                log_message = f"[{plc_name.upper()}][WARNING] {random.choice(warning_logs)}"
                log_status = "⚠️"
            else:
                log_message = f"[{plc_name.upper()}][NORMAL] {random.choice(normal_logs)}"
                log_status = "✅"

            log_payload = {
                "log_message": log_message,
                "device": plc_name,
                "timestamp": time.time(),
            }

            mqtt_client.publish(MQTT_LOG_TOPIC, json.dumps(log_payload))

            print(f"{log_status} [{plc_name}][LOG] {log_message}")
            client.close()

        except Exception as e:
            print(f"  [MODBUS] Exception in {plc_name}: {e}")

        time.sleep(random.uniform(2.5, 4.0))  # Longer interval for normal PLCs


# --- MQTT Orchestrator (Same as before) ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ [Orchestrator] Connected to MQTT Broker.")
    else:
        print(f"  [Orchestrator] Failed to connect to MQTT, return code {rc}")


def on_message(client, userdata, msg):
    pass  # Not expecting incoming messages


def on_connect_fail(client, userdata, rc):
    print("  [Orchestrator] MQTT connection failed.")


# --- Main Execution ---
if __name__ == "__main__":
    import sys
    
    # Check for training mode flag
    TRAINING_MODE = "--training" in sys.argv or "-t" in sys.argv
    
    if TRAINING_MODE:
        print("=" * 60)
        print("  🎓 TRAINING MODE - Only generating NORMAL data")
        print("  Run train_autoencoder.py to collect and train model")
        print("=" * 60)
    else:
        print("Starting OT Network Simulator...")

    # Start MQTT Client
    mqtt_client = Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="simulation-orchestrator"
    )

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.on_connect_fail = on_connect_fail

    # Connect to MQTT Broker
    while True:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
            break
        except Exception as e:
            print(f"⏳ [MQTT] Waiting for MQTT broker at {MQTT_BROKER}... ({e})")
            time.sleep(3)

    mqtt_client.loop_start()
    time.sleep(3)

    if TRAINING_MODE:
        # Training mode: Only run normal PLC1 (no attacks)
        from functools import partial
        
        def simulate_normal_plc1(host, port):
            """Normal-only PLC1 for training data collection."""
            plc_name = "plc1"
            print(f"✅ [{plc_name}] TRAINING MODE - Normal data only")
            
            while True:
                try:
                    client = ModbusTcpClient(host, port=port, timeout=2)
                    if not client.connect():
                        time.sleep(3)
                        continue
                    
                    # Generate NORMAL data only
                    values_float = []
                    
                    # Tanks: stable normal range
                    for i in range(7):
                        values_float.append(random.uniform(2.5, 4.0))
                    
                    # Pumps: normal operation
                    for i in range(11):
                        values_float.append(random.uniform(4.0, 6.0))
                        values_float.append(random.choice([0.0, 1.0]))
                    
                    # Valve
                    values_float.append(random.uniform(2.0, 3.5))
                    values_float.append(random.choice([0.0, 1.0]))
                    
                    # Pressures: normal range
                    for i in range(12):
                        values_float.append(random.uniform(280.0, 380.0))
                    
                    values_int = [int(v * SCALE_FACTOR) for v in values_float]
                    client.write_registers(0, values_int)
                    
                    normalized_data = normalize_batadal_data(values_float)
                    
                    log_payload = {
                        "log_message": f"[NORMAL][BATADAL] Training data. P_J300={values_float[34]:.2f}",
                        "device": plc_name,
                        "timestamp": time.time(),
                        "sensor_data": normalized_data,
                        "is_anomalous": False,
                    }
                    mqtt_client.publish(MQTT_LOG_TOPIC, json.dumps(log_payload))
                    print(f"✅ [{plc_name}] Training sample sent")
                    client.close()
                    
                except Exception as e:
                    print(f"Error: {e}")
                
                time.sleep(1.0)
        
        plc1_config = PLC_CONFIG["plc1"]
        threading.Thread(
            target=simulate_normal_plc1,
            args=(plc1_config["host"], plc1_config["port"]),
            daemon=True
        ).start()
        
        print("\n--- TRAINING MODE: Sending normal data only. Run train_autoencoder.py ---")
        
    else:
        # Normal mode with attacks
        # 1. START ATTACKED SYSTEM (PLC1) - Sends BATADAL data and logs
        plc1_config = PLC_CONFIG["plc1"]
        threading.Thread(
            target=simulate_anomalous_stream_plc1,
            args=(plc1_config["host"], plc1_config["port"]),
            daemon=True
        ).start()

        # 2. START NORMAL SYSTEM (PLC2) - Sends simple normal logs
        plc2_config = PLC_CONFIG["plc2"]
        threading.Thread(
            target=simulate_normal_log_plc,
            args=("plc2", plc2_config["host"], plc2_config["port"]),
            daemon=True
        ).start()

        # 3. START NORMAL SYSTEM (PLC3) - Sends simple normal logs
        plc3_config = PLC_CONFIG["plc3"]
        threading.Thread(
            target=simulate_normal_log_plc,
            args=("plc3", plc3_config["host"], plc3_config["port"]),
            daemon=True
        ).start()

        print("\n--- Simulation running: PLC1 has ATTACKS, PLC2/3 are NORMAL. Press Ctrl+C to stop. ---")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Stopping simulator ---")
        mqtt_client.loop_stop()
        print("Simulation stopped.")
