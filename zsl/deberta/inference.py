import pandas as pd
import torch
from transformers import pipeline
import os
from urllib.parse import unquote  # For URL decoding

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

def main():
    # SAAS Services and Activities
    sase_servicesg = [
        "LastPass", "Mediafire", "OneDrive", "Box", "4shared", 
        "Mega", "pCloud", "ZippyShare", "SharePoint", "Salesforce",
        "Sync", "Koofr", "Jumpshare" , "dropbox"
    ]
    activity_types = [
        "Login", "Upload", "Download", "Access", "Editing", "Deleting",
        "Sharing", "Creating", "Updating", "Syncing", "Navigation",
        "Authentication", "Attempt", "Request", "Timeout"
    ]

    # Model configuration
    service_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"  # For service prediction
    activity_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # For activity prediction

    # Dataset paths
    train_path = '/kaggle/input/network-dataset/shuffled_train.csv'  # Train dataset
    test_paths = [
        '/kaggle/input/network-dataset/koofr_all_traffic_dataset.csv',
        '/kaggle/input/network-dataset/sync_all_traffic_dataset.csv',
        '/kaggle/input/lastpass/lastpass_traffic_dataset.csv'
    ]  # List of test datasets

    df_train = clean_dataset(pd.read_csv(train_path))

    # Prepare text features for service and activity
    df_train['service_text'] = df_train.apply(prepare_service_text, axis=1)
    df_train['activity_text'] = df_train.apply(prepare_activity_text, axis=1)

    # Iterate over each test dataset
    for test_path in test_paths:
        dataset_name = os.path.basename(test_path).split('_')[0]  # Extract dataset name for identification
        print(f"\nProcessing dataset: {dataset_name}")

        df_test = clean_dataset(pd.read_csv(test_path))
        df_test['service_text'] = df_test.apply(prepare_service_text, axis=1)
        df_test['activity_text'] = df_test.apply(prepare_activity_text, axis=1)

        # Limit test dataset for demonstration
        df_test = df_test.head(10)

        # Perform zero-shot classification
        predictions = []
        for _, row in df_test.iterrows():
            # Service prediction
            service_result = perform_zero_shot_classification(
                row['service_text'], sase_services, service_model_name
            )

            # Activity prediction
            activity_result = perform_zero_shot_classification(
                row['activity_text'], activity_types, activity_model_name
            )

            # Append results
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

        # Results DataFrame
        predictions_df = pd.DataFrame(predictions)
        results = pd.concat([df_test, predictions_df], axis=1)

        # Confidence metrics
        if not predictions_df.empty:
            avg_service_confidence = predictions_df['predicted_service_confidence'].mean()
            avg_activity_confidence = predictions_df['predicted_activity_confidence'].mean()
            avg_overall_confidence = ((predictions_df['predicted_service_confidence'] + 
                                       predictions_df['predicted_activity_confidence']) / 2).mean()

            print(f"\nAverage Service Confidence Score for {dataset_name}: {avg_service_confidence:.4f}")
            print(f"Average Activity Confidence Score for {dataset_name}: {avg_activity_confidence:.4f}")
            print(f"Average Overall Confidence Score for {dataset_name}: {avg_overall_confidence:.4f}")
            print(f"Running on {'GPU' if device == 0 else 'CPU'}")

            # Save predictions
            output_path = f"{dataset_name}_predictions.csv"
            results.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        else:
            print(f"No predictions made for {dataset_name}.")

if __name__ == "__main__":
    main()