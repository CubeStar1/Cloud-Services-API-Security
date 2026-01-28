# Data Collection Components

This directory contains two approaches for collecting cloud service API traffic:

## Directory Structure
```
data-collection/
├── agent/                 # Automated data collection agent
│   ├── logs/             # Traffic logs from automated collection
│   ├── services.config.ts # Service configuration for automation
│   ├── main.ts           # Main automation script
│   ├── utils.ts          # Utility functions
│   └── README.md         # Agent-specific documentation
└── manual/               # Manual traffic capture tools
    ├── logs/            # Traffic logs from manual capture
    ├── processed_files/ # Processed CSV files
    ├── general-json-key.js # AnyProxy rules for traffic capture
    └── README.md        # Manual capture documentation
```

## 1. Automated Agent (`/agent`)

The automated agent provides programmatic data collection from cloud services.

### Features
- Automated login and interaction with cloud services
- Configurable service endpoints and actions
- TypeScript-based implementation
- Environment-based configuration

### Setup
```bash
cd agent
npm install
cp .env.example .env  # Configure your environment variables
```

### Usage
```bash
npm run build
npm start
```

## 2. Manual Capture (`/manual`)

Tools for manually capturing traffic through a proxy setup.

### Features
- AnyProxy-based traffic capture
- JSON to CSV conversion
- Flexible traffic rule configuration

### Setup
```bash
cd manual
npm install -g anyproxy
```

### Usage
1. Start AnyProxy:
```bash
anyproxy --port 8001 --rule general-json-key.js
```

2. Convert captured logs to CSV:
```bash
python csv-creation-without-tagging.py
```

## Configuration

### Service Configuration
Edit `agent/services.config.ts` to add or modify cloud services:
```typescript
export const services = {
  dropbox: {
    endpoints: [...],
    actions: [...]
  },
  // Add more services
};
```

### Proxy Rules
Edit `manual/general-json-key.js` to modify traffic capture rules:
```javascript
module.exports = {
  *beforeSendRequest(requestDetail) {
    // Custom request handling
  },
  *beforeSendResponse(requestDetail, responseDetail) {
    // Custom response handling
  }
};
``` 