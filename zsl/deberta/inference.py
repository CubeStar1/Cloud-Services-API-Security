import pandas as pd
import torch
from transformers import pipeline
import os
from urllib.parse import unquote  # For URL decoding
import glob

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Paths configuration
PATHS = {
    'data_folder': os.path.join(BASE_PATH, "data"),
    'input_data': os.path.join(BASE_PATH, "data", "logs", "csv"),  # Input CSV files
    'predictions_folder': os.path.join(BASE_PATH, "data", "output", "deberta", "predictions"),  # Output predictions
    'metrics_file': os.path.join(BASE_PATH, "data", "output", "deberta", "accuracy_scores.csv")  # Metrics output
}

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

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
        base_url = unquote(row.get('url', '').split('?')[0])  # Decode URL
        host = row.get('headers_Host', '').lower()
        return f"{host} {base_url}".strip()
    except Exception as e:
        print(f"Error in prepare_service_text: {e}")
        return ""

def prepare_activity_text(row):
    """Prepare text features for activity prediction."""
    try:
        decoded_url = unquote(row.get('url', '').lower())  # Decode URL
        return " ".join([
            decoded_url,
            row.get('method', '').upper(),
            row.get('requestHeaders_Content_Type', '').lower(),
            row.get('responseHeaders_Content_Type', '').lower(),
            row.get('requestHeaders_Referer', '').lower()
        ]).strip()
    except Exception as e:
        print(f"Error in prepare_activity_text: {e}")
        return ""

def clean_dataset(df):
    """Clean and preprocess the dataset."""
    # Drop rows with critical missing values
    df = df.dropna(subset=['headers_Host', 'url', 'method'])
    # Fill missing optional features with empty strings
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
            tokenizer_kwargs={"clean_up_tokenization_spaces": True, "max_length": 512}  # Fix warnings
        )
        return classifier(text, candidate_labels)
    except Exception as e:
        print(f"Classification error: {e}")
        return None

def combine_csv_files(data_dir):
    """Combine all CSV files in the data directory into a single DataFrame."""
    print(f"Combining CSV files from {data_dir}")
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        raise Exception(f"No CSV files found in {data_dir}")
    
    combined_df = pd.concat(
        [pd.read_csv(f) for f in all_files], 
        ignore_index=True
    )
    print(f"Combined {len(all_files)} files with total {len(combined_df)} rows")
    return combined_df

def save_accuracy_scores(service_name, metrics, output_file=PATHS['metrics_file']):
    """Save accuracy metrics to a CSV file."""
    metrics_df = pd.DataFrame([{
        'service': service_name,
        'timestamp': pd.Timestamp.now(),
        'avg_service_confidence': metrics['service_confidence'],
        'avg_activity_confidence': metrics['activity_confidence'],
        'avg_overall_confidence': metrics['overall_confidence'],
        'processed_records': metrics['processed_records']
    }])
    
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(output_file, index=False)

def process_service_file(file_path, service_model_name, activity_model_name, sase_services, activity_types):
    """Process a single service file and return predictions and metrics."""
    print(f"\nProcessing file: {file_path}")
    
    # Read and clean the data
    df_train = pd.read_csv(file_path)
    df_train = clean_dataset(df_train)
    
    # Prepare text features
    df_train['service_text'] = df_train.apply(prepare_service_text, axis=1)
    df_train['activity_text'] = df_train.apply(prepare_activity_text, axis=1)
    
    predictions = []
    for idx, row in df_train.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df_train)}")
            
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
    # SAAS Services and Activities
    sase_services = [
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

    # Create predictions directory if it doesn't exist
    os.makedirs(PATHS['predictions_folder'], exist_ok=True)
    
    # Process each file in the data directory
    all_files = glob.glob(os.path.join(PATHS['input_data'], "*.csv"))
    if not all_files:
        print(f"No CSV files found in {PATHS['input_data']}")
        return

    print(f"Found {len(all_files)} files to process")
    print(f"Running on {'GPU' if device == 0 else 'CPU'}")

    for file_path in all_files:
        try:
            # Get service name from filename
            service_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Process the file
            results, metrics = process_service_file(
                file_path, 
                service_model_name, 
                activity_model_name,
                sase_services,
                activity_types
            )
            
            # Save predictions
            output_path = os.path.join(PATHS['predictions_folder'], f"{service_name}_predictions.csv")
            results.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
            
            # Save accuracy scores
            save_accuracy_scores(service_name, metrics)
            
            # Print metrics
            print(f"\nMetrics for {service_name}:")
            print(f"Average Service Confidence Score: {metrics['service_confidence']:.4f}")
            print(f"Average Activity Confidence Score: {metrics['activity_confidence']:.4f}")
            print(f"Average Overall Confidence Score: {metrics['overall_confidence']:.4f}")
            print(f"Processed Records: {metrics['processed_records']}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

if __name__ == "__main__":
    main()