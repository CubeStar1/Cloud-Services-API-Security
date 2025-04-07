import pandas as pd
import torch
from transformers import pipeline
import os
from urllib.parse import unquote
import glob
import sys

def print_status(message: str):
    """Print status message and flush immediately for real-time monitoring."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'ignore').decode('ascii'), flush=True)

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Paths configuration
PATHS = {
    'data_folder': os.path.join(BASE_PATH, "data"),
    'input_data': os.path.join(BASE_PATH, "data", "logs", "csv"),  # Input CSV files
    'predictions_folder': os.path.join(BASE_PATH, "data", "output", "deberta", "predictions"),  # Output predictions
    'metrics_file': os.path.join(BASE_PATH, "data", "output", "deberta", "accuracy_scores.csv")  # Metrics output
}

# Create necessary directories
os.makedirs(PATHS['predictions_folder'], exist_ok=True)

# CUDA configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

def clean_url(url):
    """Extract the base URL without query parameters."""
    return url.split('?')[0]

def extract_domain(host):
    """Extract the primary domain from the host."""
    if not host:
        return ""
    parts = host.split('.')
    return ".".join(parts[-2:]) if len(parts) > 1 else host

def prepare_service_text(row):
    """Prepare text features for service prediction."""
    try:
        base_url = unquote(row.get('url', '').split('?')[0])
        host = row.get('headers_Host', '').lower()
        return f"{host} {base_url}".strip()
    except Exception as e:
        print_status(f"[!] Error in prepare_service_text: {e}")
        return ""

def prepare_activity_text(row):
    """Prepare text features for activity prediction."""
    try:
        decoded_url = unquote(row.get('url', '').lower())
        return " ".join([
            decoded_url,
            row.get('method', '').upper(),
            row.get('requestHeaders_Content_Type', '').lower(),
            row.get('responseHeaders_Content_Type', '').lower(),
            row.get('requestHeaders_Referer', '').lower()
        ]).strip()
    except Exception as e:
        print_status(f"[!] Error in prepare_activity_text: {e}")
        return ""

def clean_dataset(df):
    """Clean and preprocess the dataset."""
    df = df.dropna(subset=['headers_Host', 'url', 'method'])
    optional_features = ['requestHeaders_Content_Type', 'responseHeaders_Content_Type', 'requestHeaders_Referer']
    df[optional_features] = df[optional_features].fillna('')
    return df

def perform_zero_shot_classification(text, candidate_labels, model_name):
    """Perform zero-shot classification with error handling."""
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            tokenizer_kwargs={"clean_up_tokenization_spaces": True, "max_length": 512}
        )
        
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return classifier(text, candidate_labels)
        return classifier(text, candidate_labels)
    except Exception as e:
        print_status(f"[!] Classification error: {e}")
        return None

def process_service_file(file_path, service_model_name, activity_model_name, sase_services, activity_types):
    """Process a single service file and return predictions and metrics."""
    print_status(f"[*] Processing file: {os.path.basename(file_path)}")
    
    # Read and clean the data
    df_train = pd.read_csv(file_path)
    df_train = clean_dataset(df_train)
    
    print_status(f"[+] Loaded {len(df_train)} records")
    
    # Prepare text features
    print_status("[*] Preparing text features...")
    df_train['service_text'] = df_train.apply(prepare_service_text, axis=1)
    df_train['activity_text'] = df_train.apply(prepare_activity_text, axis=1)
    
    predictions = []
    total_rows = len(df_train)
    
    print_status("[*] Starting classification...")
    for idx, row in df_train.iterrows():
        progress = (idx + 1) / total_rows * 100
        if idx % 10 == 0:  # Update progress every 10 rows
            print_status(f"[*] Progress: {progress:.1f}% - Row {idx + 1}/{total_rows}")
            
        service_result = perform_zero_shot_classification(
            row['service_text'], sase_services, service_model_name
        )
        
        activity_result = perform_zero_shot_classification(
            row['activity_text'], activity_types, activity_model_name
        )
        
        if service_result and activity_result:
            predictions.append({
                'predicted_service': service_result['labels'][0],
                'predicted_service_confidence': service_result['scores'][0],
                'predicted_activity': activity_result['labels'][0],
                'predicted_activity_confidence': activity_result['scores'][0]
            })
        else:
            predictions.append({
                'predicted_service': 'Unknown',
                'predicted_service_confidence': 0,
                'predicted_activity': 'Unknown',
                'predicted_activity_confidence': 0
            })
    
    predictions_df = pd.DataFrame(predictions)
    results = pd.concat([df_train, predictions_df], axis=1)
    
    # Calculate metrics
    metrics = {
        'service_confidence': predictions_df['predicted_service_confidence'].mean(),
        'activity_confidence': predictions_df['predicted_activity_confidence'].mean(),
        'overall_confidence': ((predictions_df['predicted_service_confidence'] +
                              predictions_df['predicted_activity_confidence']) / 2).mean(),
        'processed_records': len(predictions_df)
    }
    
    return results, metrics

def main():
    try:
        print_status("[*] Starting DeBERTa inference process...")
        
        # Print CUDA information
        print_status(f"[*] Using device: {device}")
        if torch.cuda.is_available():
            print_status(f"[+] GPU detected: {torch.cuda.get_device_name(0)}")
            print_status(f"[+] CUDA Version: {torch.version.cuda}")

        # SAAS Services and Activities
        sass_services = [
            "Asana", "Google Cloud", "Google Drive", "Github", "Vercel", "Netlify", "Slack",
            "Microsoft 365", "Dropbox", "Jira", "Salesforce", "AWS", "Azure", "Zoom",
            "Trello", "Notion", "Figma", "Airtable", "Zendesk", "Unknown"
        ]

        activity_types = [
            "Login", "Upload", "Download", "Access", "Editing", "Deleting",
            "Sharing", "Creating", "Updating", "Syncing", "Navigation",
            "Authentication", "Attempt", "Request", "Timeout", "Export", "Import",
            "Comment", "Review", "Approve", "Reject", "Query", "Visualization",
            "Configuration", "Integration", "Deployment", "Rollback", "Scan", "Audit",
            "Permission Change", "Password Reset", "Account Creation", "API Call",
            "Unknown"
        ]

        # Model configuration
        service_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        activity_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

        # Process each file in the data directory
        all_files = glob.glob(os.path.join(PATHS['input_data'], "*.csv"))
        if not all_files:
            print_status("[!] No CSV files found")
            return

        print_status(f"[+] Found {len(all_files)} files to process")
        print_status(f"[*] Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

        for file_path in all_files:
            try:
                service_name = os.path.splitext(os.path.basename(file_path))[0]
                print_status(f"\n[*] Processing service: {service_name}")
                
                results, metrics = process_service_file(
                    file_path, 
                    service_model_name, 
                    activity_model_name,
                    sass_services,
                    activity_types
                )
                
                output_path = os.path.join(PATHS['predictions_folder'], f"{service_name}_predictions.csv")
                results.to_csv(output_path, index=False)
                print_status(f"[+] Predictions saved to: {os.path.basename(output_path)}")
                
                save_accuracy_scores(service_name, metrics)
                
                print_status(f"""
[*] Metrics for {service_name}:
    - Service Confidence: {metrics['service_confidence']:.4f}
    - Activity Confidence: {metrics['activity_confidence']:.4f}
    - Overall Confidence: {metrics['overall_confidence']:.4f}
    - Processed Records: {metrics['processed_records']}
                """)
                
            except Exception as e:
                print_status(f"[!] Error processing {file_path}: {str(e)}")
                continue

        print_status("[+] DeBERTa inference completed successfully!")
        
    except Exception as e:
        print_status(f"[!] Error during inference process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()