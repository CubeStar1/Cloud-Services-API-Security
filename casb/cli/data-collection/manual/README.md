# SaaS Network Traffic Collector

A tool that collects network traffic data from various SaaS services using AnyProxy. It captures all network requests and responses.

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


4. Add the `general-json-key.js` file to the project's root directory:

Run the following command to capture the network traffic:
```bash
anyproxy --port 8001  --rule .\general-json-key.js --intercept
```

5. To generate the csv file run the following command:
```bash
python csv_creation.py
```


