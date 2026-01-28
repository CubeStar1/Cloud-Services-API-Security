#!/usr/bin/env python3
"""
Autoencoder Training Script for OT Anomaly Detection

This script trains an autoencoder on NORMAL simulated sensor data,
so it can detect anomalies when sensor values deviate from normal patterns.

Usage:
    1. Run simulator in NORMAL mode to collect training data
    2. Run this script to train the model
    3. The new model will be saved as 'ae_model_trained.h5'
"""

import numpy as np
import json
import time
import os
from datetime import datetime

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# MQTT for collecting training data
import paho.mqtt.client as mqtt

# --- Configuration ---
MQTT_BROKER = "localhost"  # Use 'mqtt' if running inside Docker
MQTT_PORT = 1883
MQTT_TOPIC = "ot/logs"
NUM_FEATURES = 43  # BATADAL features
TRAINING_SAMPLES = 500  # Number of normal samples to collect
MODEL_SAVE_PATH = "ae_model_trained.h5"

# --- Data Collection ---
training_data = []
collection_complete = False

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ Connection failed with code {rc}")

def on_message(client, userdata, msg):
    global training_data, collection_complete
    
    if collection_complete:
        return
        
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        
        # Only collect from PLC1 (has sensor data)
        if payload.get('device') != 'plc1':
            return
            
        sensor_data = payload.get('sensor_data')
        is_anomalous = payload.get('is_anomalous', False)
        
        # Only collect NORMAL data for training
        if sensor_data and not is_anomalous and len(sensor_data) == NUM_FEATURES:
            training_data.append(sensor_data)
            print(f"\r📊 Collected {len(training_data)}/{TRAINING_SAMPLES} normal samples", end="")
            
            if len(training_data) >= TRAINING_SAMPLES:
                collection_complete = True
                print("\n✅ Data collection complete!")
                
    except Exception as e:
        pass

def collect_training_data():
    """Collect normal sensor data from MQTT."""
    global training_data, collection_complete
    
    print(f"📡 Connecting to MQTT at {MQTT_BROKER}:{MQTT_PORT}")
    print(f"⏳ Waiting for {TRAINING_SAMPLES} normal samples from plc1...")
    print("⚠️  Make sure simulator is running in NORMAL mode (no attacks)")
    print()
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ae-trainer")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        # Wait for data collection
        timeout = 300  # 5 minutes max
        start_time = time.time()
        
        while not collection_complete:
            if time.time() - start_time > timeout:
                print("\n⚠️ Timeout! Using collected samples.")
                break
            time.sleep(0.5)
            
        client.loop_stop()
        client.disconnect()
        
    except Exception as e:
        print(f"❌ MQTT Error: {e}")
        print("Make sure the MQTT broker is accessible.")
        return None
    
    if len(training_data) < 50:
        print(f"❌ Not enough data collected ({len(training_data)} samples)")
        return None
        
    return np.array(training_data)


def build_autoencoder(input_dim, encoding_dim=16):
    """
    Build an autoencoder for anomaly detection.
    
    Architecture:
    - Encoder: 43 -> 32 -> 16 (compressed representation)
    - Decoder: 16 -> 32 -> 43 (reconstruction)
    
    Normal data will have low reconstruction error.
    Anomalous data will have high reconstruction error.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(x)
    
    # Decoder
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)  # sigmoid for 0-1 normalized data
    
    # Full autoencoder
    autoencoder = Model(inputs, decoded, name='sensor_autoencoder')
    
    # Compile with MSE loss (measures reconstruction error)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder


def train_autoencoder(data, epochs=100, batch_size=32, validation_split=0.2):
    """Train the autoencoder on normal data."""
    
    print(f"\n🏗️  Building autoencoder model...")
    print(f"   Input features: {data.shape[1]}")
    print(f"   Training samples: {data.shape[0]}")
    
    model = build_autoencoder(input_dim=data.shape[1])
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]
    
    print(f"\n🚀 Training for up to {epochs} epochs...")
    
    history = model.fit(
        data, data,  # Autoencoder: input = target
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, data):
    """Evaluate the trained model and determine anomaly threshold."""
    
    print("\n📊 Evaluating model on training data...")
    
    # Get reconstruction error for normal data
    predictions = model.predict(data, verbose=0)
    mse_per_sample = np.mean((data - predictions) ** 2, axis=1)
    
    # Statistics
    mean_mse = np.mean(mse_per_sample)
    std_mse = np.std(mse_per_sample)
    max_mse = np.max(mse_per_sample)
    
    # Threshold: mean + 2*std (catches ~95% of normal as normal)
    threshold = mean_mse + 2 * std_mse
    
    print(f"   Mean MSE (normal): {mean_mse:.6f}")
    print(f"   Std MSE: {std_mse:.6f}")
    print(f"   Max MSE: {max_mse:.6f}")
    print(f"   Recommended threshold: {threshold:.6f}")
    
    return threshold


def main():
    print("=" * 60)
    print("  OT Autoencoder Training Script")
    print("  Trains anomaly detection model on normal sensor data")
    print("=" * 60)
    print()
    
    # Step 1: Collect training data
    print("📥 STEP 1: Collecting training data from MQTT...")
    data = collect_training_data()
    
    if data is None:
        print("\n❌ Failed to collect training data. Exiting.")
        return
    
    print(f"\n✅ Collected {len(data)} training samples")
    
    # Step 2: Train model
    print("\n🎯 STEP 2: Training autoencoder...")
    model, history = train_autoencoder(data, epochs=100)
    
    # Step 3: Evaluate
    print("\n📈 STEP 3: Evaluating model...")
    threshold = evaluate_model(model, data)
    
    # Step 4: Save model
    print(f"\n💾 STEP 4: Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    # Save threshold to file
    with open("ae_threshold.txt", "w") as f:
        f.write(f"{threshold}")
    
    print(f"\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print(f"  Model saved: {MODEL_SAVE_PATH}")
    print(f"  Threshold: {threshold:.6f}")
    print("=" * 60)
    print()
    print("📋 Next steps:")
    print("   1. Copy ae_model_trained.h5 to replace AE-BATADAL-l5-cf2.5.h5")
    print("   2. Update fusion_layer.py to use the new threshold")
    print("   3. Restart the backend container")
    print()


if __name__ == "__main__":
    main()
