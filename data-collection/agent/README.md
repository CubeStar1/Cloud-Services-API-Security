# SaaS Network Traffic Collector

A tool that automates the collection of network traffic data from various SaaS services using Stagehand and AnyProxy. It supports Google Sign-in automation for multiple services and captures all network requests and responses.

NOTE: Work in progress, actions for each services need to be properly configured in `services.config.ts`. Some services may not work as expected.

## Prerequisites

- Node.js (v14 or higher)
- npm
- AnyProxy (globally installed)
- Google Chrome/Chromium

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CubeStar1/saas-network-collector.git
cd saas-network-collector
```

2. Install dependencies:
```bash
npm install
```

3. Install AnyProxy globally:
```bash
npm install -g anyproxy
```


4. Add the `general-json-key.cjs` file to the project's root directory:

Run the following command to capture the network traffic:
```bash
anyproxy --port 8001  --rule .\general-json-key.js --intercept
```

(For the AI agent do the next steps)
5. Copy the example environment file and add your credentials:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env

# OpenAI
OPENAI_API_KEY=your_openai_key

# Google
GOOGLE_EMAIL=your_google_email
GOOGLE_PASSWORD=your_google_password

```

6. To generate the csv file run the following command:
```bash
python csv_creation.py
```

## Usage

###  Command Line

Run both the proxy and automation:
```bash
npm run start:all
```

This will:
- Start AnyProxy on port 8001
- Start the Stagehand automation
- Create log files in the `logs` directory


## Log Files

Network traffic logs are stored in the `logs` directory, organized by service:
```
logs/
  slack/
    traffic-logs.json
  github/
    traffic-logs.json
  trello/
    traffic-logs.json
  ...
```

Each log file contains request and response data in JSON format.

## Configuration

### Services

Services are configured in `services.config.ts`. Each service has:
- name: Service identifier
- url: Starting URL
- actions: Array of automation steps

Example:
```typescript
{
  name: 'slack',
  url: 'https://slack.com/get-started',
  actions: [
    "Click 'Sign in with Google'",
    ...googleSignInActions,
    "Click 'Create a new workspace'"
  ]
}
```

### Credentials

Credentials are managed in `credentials.config.ts` and `.env`. Currently supports:
- Google Sign-in credentials

## Note

- Store sensitive credentials securely in environment variables
- Use test accounts for automation
- Don't commit `.env` file to version control
