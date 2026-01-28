import json
import csv
import urllib.parse  #Parsing URL-encoded data
import os
import glob  # Importing glob to find all JSON files

# Base path configuration - point to frontend directory
BASE_PATH = os.path.dirname(os.path.dirname(__file__))  # Get frontend directory

# File paths and directories configuration
PATHS = {
    'data_folder': os.path.join(BASE_PATH, "data"),
    'raw_json_folder': os.path.join(BASE_PATH, "data", "logs", "raw-json"),
    'csv_folder': os.path.join(BASE_PATH, "data", "logs", "csv"),
}

def read_logs(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = f.readlines()
    return [json.loads(log.strip(',\n')) for log in logs if log.strip(',\n')]

def process_logs_with_keys(logs):
    processed_logs = []
    for log in logs:
        processed_log = {
            'headers_Host': log.get('headers_Host') if log.get('headers_Host') not in [None, '', 'null'] else 'none',
            'url': log.get('url') if log.get('url') not in [None, '', 'null'] else 'none',
            'method': log.get('method') if log.get('method') not in [None, '', 'null'] else 'UNKNOWN',
            'requestHeaders_Origin': log.get('requestHeaders_Origin') if log.get('requestHeaders_Origin') not in [None, '', 'null'] else 'none',
            'requestHeaders_Content_Type': log.get('requestHeaders_Content_Type') if log.get('requestHeaders_Content_Type') not in [None, '', 'null'] else 'none',
            'responseHeaders_Content_Type': log.get('responseHeaders_Content_Type') if log.get('responseHeaders_Content_Type') not in [None, '', 'null'] else 'none',
            'requestHeaders_Referer': log.get('requestHeaders_Referer') if log.get('requestHeaders_Referer') not in [None, '', 'null'] else 'none',
            'requestHeaders_Accept': log.get('requestHeaders_Accept') if log.get('requestHeaders_Accept') not in [None, '', 'null'] else 'none',
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

def main():
    print(f"Starting CSV conversion...")
    print(f"Base path: {BASE_PATH}")
    
    # Create necessary directories
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
        print(f"Ensuring directory exists: {path}")

    # Get all JSON files in the raw-json folder
    json_files = glob.glob(os.path.join(PATHS['raw_json_folder'], "*.json"))

    if not json_files:
        print(f"No JSON files found in {PATHS['raw_json_folder']}")
        return

    # Processing steps for each log file
    for json_file in json_files:
        try:
            # Extract the base name without extension for output file naming
            base_name = os.path.basename(json_file).replace('.json', '')
            output_csv = os.path.join(PATHS['csv_folder'], f"{base_name}.csv")

            print(f"Processing {base_name}.json...")
            logs = read_logs(json_file)
            processed_logs = process_logs_with_keys(logs)
            write_to_csv(processed_logs, output_csv)

            print(f"Successfully converted {base_name}.json -> {base_name}.csv")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print("CSV conversion completed!")

if __name__ == "__main__":
    main()
