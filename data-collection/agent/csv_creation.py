import json
import csv
import urllib.parse  #Parsing URL-encoded data
import os

def read_logs(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = f.readlines()
    return [json.loads(log.strip(',\n')) for log in logs if log.strip(',\n')]

def process_logs_with_keys(logs):
    processed_logs = []
    for log in logs:
        processed_log = {
            'headers_Host': log.get('headers_Host', ''),
            'url': log.get('url', ''),
            'method': log.get('method', 'UNKNOWN'),
            'requestHeaders_Origin': log.get('requestHeaders_Origin', ''),
            'requestHeaders_Content_Type': log.get('requestHeaders_Content_Type', ''),
            'responseHeaders_Content_Type': log.get('responseHeaders_Content_Type', ''),
            'requestHeaders_Referer': log.get('requestHeaders_Referer', ''),
            'requestHeaders_Accept': log.get('requestHeaders_Accept', ''),
            'request_keys': extract_keys(log.get('body', ''), log.get('type', 'request') == 'request'),
            'response_keys': extract_keys(log.get('body', ''), log.get('type', 'response') == 'response'),
        }
        processed_logs.append(processed_log)
    return processed_logs

def extract_keys(body_data, is_request):
    if not body_data:
        return "none"
    try:
        if "=" in body_data and "&" in body_data:
            # URL-encoded body
            parsed_data = urllib.parse.parse_qs(body_data)
            return "#".join(parsed_data.keys())
        else:
            # JSON-structured body
            body_json = json.loads(body_data)
            if isinstance(body_json, dict):
                return "#".join(body_json.keys())
            elif isinstance(body_json, list):
                return "#".join([str(i) for i in range(len(body_json))])
            else:
                return "unknown_structure"
    except:
        return "unknown_format"

def write_to_csv(processed_logs, output_file):
    headers = [
        'headers_Host', 'url', 'method', 'requestHeaders_Origin',
        'requestHeaders_Content_Type', 'responseHeaders_Content_Type',
        'requestHeaders_Referer', 'requestHeaders_Accept',
        'request_keys', 'response_keys'
    ]

    # Remove existing file if it exists to avoid appending to the old data
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for log in processed_logs:
            row = {key: log.get(key, '') for key in headers}
            writer.writerow(row)

# Paths to input and output files
log_file = "logs/traffic-logs.json"
output_csv = "web_traffic_processed.csv"

# Processing steps
logs = read_logs(log_file)
processed_logs = process_logs_with_keys(logs)
write_to_csv(processed_logs, output_csv)

print(f"Processed dataset created: {output_csv}")
