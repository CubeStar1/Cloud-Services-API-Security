# Cloud Services API Security Analysis

## Project Structure
```
Cloud-Services-API-Security/
├── data-collection/           # Traffic capture components
│   ├── agent/                # Automated data collection
│   └── manual/              # Manual traffic capture
├── data/                    # Dataset storage
├── labelling/              # Initial labeling using GPT-4/Gemini
│   ├── labelling.py       # Main labeling script
├── zsl/                    # Zero-shot learning models
│   ├── codebert/          # CodeBERT-based classifier
│   │   ├── train.py       # Training pipeline
│   │   └── inference.py   # Inference with ZSL
│   └── deberta/           # DeBERTa-based classifier
│       └── inference.py   # Multilingual ZSL inference
└── rfc/                    # Random Forest training
```

## Project Overview

1. **Data Collection**: 
   - Automated agent for data collection
   - Manual proxy-based traffic capture
2. **Initial Labeling**: 
3. **Zero-Shot Learning**
4. **Training**: 
   - Random Forest Classifier on labeled data

## Components

### 1. Data Collection (`/data-collection`)
Two approaches for gathering cloud service traffic:

#### a) Automated Agent
```bash
cd data-collection/agent
npm install
cp .env.example .env
npm run build && npm start
```

#### b) Manual Capture
```bash
cd data-collection/manual
anyproxy --port 8001 --rule general-json-key.js
```

### 2. Initial Labeling (`/labelling`)

#### Usage
```bash
cd labelling

# Set up environment
cp .env.example .env
# Add your API keys to .env:
# OPENAI_API_KEY=your_key
# GOOGLE_API_KEY=your_key

# Install dependencies
pip install -r requirements.txt

# Run labeling
python labelling.py
```

### 3. Zero-Shot Learning (`/zsl`)

#### CodeBERT Implementation
Advanced technical text classification using CodeBERT:

```bash
cd zsl/codebert

# Training
python train.py 

# Inference
python inference.py 
```


#### DeBERTa Implementation

```bash
cd zsl/deberta
python inference.py 
```
### 4. Random Forest Training (`/rfc`)
#### Usage
```bash
cd rfc

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py 
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CubeStar1/Cloud-Services-API-Security.git
cd Cloud-Services-API-Security
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Node.js components:
```bash
cd data-collection/agent
npm install
```

4. Install AnyProxy:
```bash
npm install -g anyproxy
```

## Configuration and Workflow

### 1. Data Collection
- Configure services in `data-collection/agent/services.config.ts`
  - Define cloud services to monitor
  - Set up authentication credentials
- Set up proxy rules in `data-collection/manual/general-json-key.js`
  - Define traffic capture patterns

Purpose: Gather raw HTTP traffic data from various cloud services through automated and manual methods.

### 2. Initial Labeling (Training Data Generation)
- Configure API keys in `labelling/.env`:
  ```
  OPENAI_API_KEY=your_key
  GOOGLE_API_KEY=your_key
  ```
- Adjust settings in `labelling/labelling.py`:
  ```python
  CONFIG = {
      'batch_size': 10,
      'use_openai': True  # Toggle between OpenAI/Gemini
  }
  ```

Purpose: Generate initial labeled dataset using GPT-4/Gemini to train the CodeBERT model.

### 3. Zero-Shot Learning
Run both models on unlabeled traffic data:

#### a) CodeBERT
```bash
cd zsl/codebert
python inference.py 
```

#### b) DeBERTa
```bash
cd zsl/deberta
python inference.py 
```

Purpose: Generate high-confidence predictions for both known and unknown patterns in the traffic data.

### 4. Random Forest Training
Configure in `rfc/train.py`:
```python
# Data processing
features = [
    'headers_Host',
    'url',
    'method',
    'requestHeaders_Content_Type',
    # ... other features
]

```

Purpose: Train the final classifier using the combined predictions from CodeBERT and DeBERTa for service and activity classification.


