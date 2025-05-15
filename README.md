# Cloud Services API Security Analysis

![Frontend Dashboard](./frontend/public/cas-dashboard.png)
![RFC Generation](./frontend/public/cas-rfc.png)

## Project Structure
```
Cloud-Services-API-Security/
├── data-collection/           # Traffic capture components
│   ├── agent/                # Automated data collection
│   └── manual/              # Manual traffic capture
├── data/                    # Dataset storage
├── frontend/                # Next.js web application
│   ├── app/                 # Next.js App Router structure
│   ├── components/          # Reusable UI components
│   ├── public/              # Static assets
│   └── lib/                 # Utility functions and hooks
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
5. **Frontend Application**:
   - Interactive dashboard
   - Data visualization
   - Model interaction

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

### 5. Frontend Application (`/frontend`)
Modern Next.js application with interactive UI for the entire pipeline.

#### Features
- Interactive dashboard with model visualizations
- File browser and viewer (excluding CSV files)
- Workflow animations and pipeline visualization
- Dark/light theme support

#### Installation
```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the application.

#### Key Pages
- **Dashboard**: Main metrics and visualization
- **Data Collection**: Interface for AnyProxy
- **DeBERTa/CodeBERT**: Model visualization interfaces
- **RFC Generation**: Random Forest results
- **Files**: File browser and management

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

cd ../../frontend
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

### 5. Frontend Application
Start the interactive frontend application:

```bash
cd frontend
npm run dev
```

Purpose: Provide a centralized interface for monitoring the entire pipeline, visualizing results, and managing files.

## Complete Workflow

1. Collect API traffic data using AnyProxy
2. Process and label initial data with GPT-4/Gemini
3. Apply zero-shot learning with DeBERTa and CodeBERT
4. Train Random Forest classifier on labeled data
5. View results and manage files through the frontend application

## Technologies Used

- **Backend**: Python, Node.js, AnyProxy
- **Models**: DeBERTa, CodeBERT, Random Forest
- **Frontend**: Next.js, React, Tailwind CSS, Framer Motion
- **APIs**: OpenAI GPT-4, Google Gemini

## C Code Generation System Explained

The Random Forest Classifier is converted into optimized C code for runtime efficiency. Here's how the system works, explained in a simple way:

### Hash Table System (Like a Library)

Imagine our system as a library with thousands of words (features). We need to find these words quickly when processing API requests. Here's how it works:

#### 1. The Library Structure
```c
typedef struct {
    int indices[10];  // Like 10 spots on each shelf
    int count;        // How many words are on this shelf
} HashBucket;

typedef struct {
    const char* term;      // The actual word
    int feature_index;     // Word's special number
} FeatureEntry;
```

#### 2. Organization System
- We have 8,192 shelves (HASH_TABLE_SIZE)
- Each shelf can hold up to 10 words
- Words are placed on shelves based on their "hash" (like a shelf number)

#### 3. Real Example
Let's say we have these API words:
```
Words to store:
- "GET"
- "POST"
- "api"
- "users"
```

The system organizes them like this:
```
Shelf 1234: ["GET", "api"]        // Two words on this shelf
Shelf 5678: ["POST"]              // One word on this shelf
... other shelves ...
```

#### 4. Finding Words
When processing an API request like "GET /api/users":

1. Split into words: ["GET", "api", "users"]
2. For each word:
   - Calculate its shelf number (hash)
   - Go directly to that shelf
   - Look through at most 10 words
   - If found, mark it in our features


