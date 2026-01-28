# OT Security Platform - Juniper Networks

A comprehensive **Operational Technology (OT) Security** platform that combines advanced network scanning, real-time anomaly detection using ML/AI models, and a modern React dashboard for monitoring industrial control systems.



##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                  │
│    Network Topology │ Device Discovery │ Anomaly Dashboard      │
└────────────────────────────────┬────────────────────────────────┘
                                 │ REST API
┌────────────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scanner   │  │   Fusion    │  │      Firebase/          │  │
│  │  (IT/OT)    │  │   Layer     │  │      Firestore          │  │
│  └─────────────┘  └──────┬──────┘  └─────────────────────────┘  │
│                          │                                      │
│           ┌──────────────┼──────────────┐                       │
│           ▼              ▼              ▼                       │
│    ┌───────────┐  ┌───────────┐  ┌───────────┐                  │
│    │  LogBERT  │  │Autoencoder│  │   MQTT    │                  │
│    │  (Logs)   │  │ (Sensors) │  │ Subscriber│                  │
│    └───────────┘  └───────────┘  └───────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                       ICS Flat Network                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  OT Devices                                                 ││
│  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌────────┐  ┌──────────┐  ││
│  │  │ PLC1  │  │ PLC2  │  │ PLC3  │  │  MQTT  │  │   HMI    │  ││
│  │  │Modbus │  │Modbus │  │Modbus │  │ Broker │  │ (nginx)  │  ││
│  │  └───────┘  └───────┘  └───────┘  └────────┘  └──────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  IT Devices                                                 ││
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────┐           ││
│  │  │ Webserver  │  │  Database  │  │  Simulator   │           ││
│  │  │  (nginx)   │  │  (MySQL)   │  │  (BATADAL)   │           ││
│  │  └────────────┘  └────────────┘  └──────────────┘           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Features

###  Network Scanning
- **IT/OT Protocol Detection**: Modbus, S7Comm, BACnet, EtherNet/IP, OPC UA, PROFINET, MQTT, DNP3
- **Deep Packet Inspection**: Protocol signature analysis and banner grabbing
- **Device Fingerprinting**: MAC vendor lookup and service identification
- **Risk Assessment**: Automated risk scoring based on exposed services

###  ML-Powered Anomaly Detection
- **LogBERT**: BERT-based log anomaly detection trained on ICS/OT logs
- **Autoencoder**: BATADAL-trained sensor anomaly detection
- **Fusion Layer**: Combines multiple ML models for enhanced accuracy
- **Real-time Inference**: Processes logs via MQTT for instant threat detection

###  Dashboard
- **Device Discovery Table**: Interactive list of discovered OT devices
- **Network Topology**: Visual graph representation of network layout
- **Anomaly Detection Panel**: Real-time alerts and threat visualization
- **Summary Statistics**: Protocol distribution and risk overview

##  Quick Start

### Prerequisites
- **Docker** & **Docker Compose** (v2.0+)
- **Node.js** (v18+) or **Bun** for frontend
- **macOS/Linux** (Windows via WSL2)

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/Operational-Technology-Security-JuniperNetworks.git
cd Operational-Technology-Security-JuniperNetworks
```

### 2. Start the Backend (Docker Compose)
```bash
cd backend
docker-compose up --build
```

This starts:
| Service | Description | Port |
|---------|-------------|------|
| `fastapi-backend` | Main API server | 8000 |
| `plc1`, `plc2`, `plc3` | Modbus PLC simulators | 5021-5023 |
| `mqtt` | Mosquitto MQTT broker | 1883 |
| `hmi` | HMI dashboard (nginx) | 8085 |
| `webserver` | IT web server | 8081 |
| `database` | MySQL database | 3306 |
| `simulator` | OT data simulator | - |

### 3. Start the Frontend
```bash
cd frontend

# Using npm
npm install
npm run dev

# Or using Bun (faster)
bun install
bun dev
```

Frontend will be available at: **http://localhost:5173**

### 4. Access the Application
| Component | URL |
|-----------|-----|
| **Dashboard** | http://localhost:5173 |
| **API Docs** | http://localhost:8000/docs |
| **HMI** | http://localhost:8085 |

##  Project Structure

```
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── scan.py                    # Advanced IT/OT network scanner
│   ├── fusion_layer.py            # ML model fusion (LogBERT + AE)
│   ├── logbert_ics_ot.py          # LogBERT anomaly detector
│   ├── detector.py                # Real-time MQTT log processor
│   ├── simulator.py               # OT device data simulator
│   ├── docker-compose.yml         # Multi-service orchestration
│   ├── Dockerfile                 # FastAPI container
│   ├── Dockerfile.simulator       # Simulator container
│   ├── requirements.txt           # Python dependencies
│   ├── logbert_ics_model.pt       # Trained LogBERT weights
│   ├── AE-BATADAL-l5-cf2.5.h5     # Trained Autoencoder model
│   └── serviceAccountKey.json     # Firebase credentials
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main app component
│   │   ├── pages/
│   │   │   └── Index.tsx          # Dashboard page
│   │   └── components/
│   │       ├── ScannerInput.tsx   # Network scan input
│   │       ├── DeviceTable.tsx    # Device list table
│   │       ├── NetworkTopology.tsx # Topology visualization
│   │       ├── AnomalyDetection.tsx # Anomaly alerts
│   │       └── Summary.tsx        # Statistics summary
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

##  Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Firebase Configuration
GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json

# MQTT Configuration
MQTT_BROKER=mqtt
MQTT_PORT=1883

# Model Paths
LOGBERT_MODEL_PATH=logbert_ics_model
AE_MODEL_PATH=AE-BATADAL-l5-cf2.5.h5
```

### Firebase Setup
1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Generate a service account key
3. Save it as `backend/serviceAccountKey.json`

## API Endpoints

### Scanning
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/scan` | Start network scan |
| `GET` | `/scan/{task_id}` | Get scan status |
| `GET` | `/scan/{task_id}/results` | Get scan results |

### Devices
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/devices` | List discovered devices |
| `GET` | `/devices/{ip}/logs` | Get device logs |

### Anomaly Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/alerts` | Get anomaly alerts |
| `POST` | `/analyze` | Analyze log entry |

## Supported Protocols

### OT/ICS Protocols
- **Modbus TCP** (Ports 502, 802)
- **Siemens S7Comm** (Port 102)
- **BACnet/IP** (Port 47808)
- **EtherNet/IP** (Ports 44818, 2222)
- **OPC UA** (Port 4840)
- **PROFINET** (Ports 34962-34964)
- **MQTT** (Ports 1883, 8883)
- **CODESYS** (Ports 1200, 2455)
- **Niagara Fox** (Port 1911)

### IT Protocols
- **HTTP** (Ports 80, 8080, 8000, 8888)
- **HTTPS** (Ports 443, 8443)
- **SSH** (Port 22)
- **Telnet** (Port 23)
- **FTP** (Port 21)
- **SMTP** (Ports 25, 587)
- **RDP** (Port 3389)
- **SMB** (Ports 445, 139)
- **MySQL** (Port 3306)
- **PostgreSQL** (Port 5432)
- **VNC** (Ports 5900, 5901)

##  Machine Learning Models

### LogBERT
- **Architecture**: BERT-based transformer for log sequence analysis
- **Training Data**: ICS/OT logs (Modbus, DNP3, BACnet)
- **Detection**: Command injection, protocol violations, unauthorized access

### Autoencoder
- **Architecture**: Deep autoencoder for sensor data reconstruction
- **Training Data**: BATADAL water distribution dataset
- **Detection**: Sensor manipulation, process anomalies, data injection attacks

### Fusion Layer
- **Method**: Weighted ensemble of LogBERT + Autoencoder scores
- **Weights**: Configurable (default: 0.6 LogBERT, 0.4 Autoencoder)

##  Docker Commands

```bash
# Start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f fastapi-backend

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild specific service
docker-compose up --build fastapi-backend
```

##  Development

### Running Tests
```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm run test
```

### Local Development (without Docker)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm run dev
```

##  BATADAL Dataset

The autoencoder is trained on the **BATADAL** (BATtle of the Attack Detection ALgorithms) dataset, which simulates a water distribution network with:
- 43 sensor features (tank levels, pump flows, pressures)
- Normal operation and attack scenarios
- Realistic SCADA data patterns



##  Acknowledgments

- [BATADAL Dataset](https://www.batadal.net/) - Water distribution attack dataset
- [Hugging Face Transformers](https://huggingface.co/transformers/) - BERT implementation
- [Scapy](https://scapy.net/) - Network packet manipulation
- [shadcn/ui](https://ui.shadcn.com/) - React UI components

---


