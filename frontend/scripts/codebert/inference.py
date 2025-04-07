import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
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
    'train_data': os.path.join(BASE_PATH, "data", "labelled", "train_set.xlsx"),
    'test_data': os.path.join(BASE_PATH, "data", "logs", "csv"),  # Input CSV files
    'predictions_folder': os.path.join(BASE_PATH, "data", "output", "codebert", "predictions"),  # Output predictions
    'model_file': os.path.join(BASE_PATH, "data", "output", "codebert", "models", "best_codebert_model.pth"),  # Model path
}

ACTIVITY_LABELS = [
    "Login", "Upload", "Download", "Access", "Editing", "Deleting",
    "Sharing", "Creating", "Updating", "Syncing", "Navigation",
    "Authentication", "Attempt", "Request", "Timeout", "Export",
    "Import", "Comment", "Review", "Approve", "Reject", "Query",
    "Visualization", "Configuration", "Integration", "Deployment",
    "Rollback", "Scan", "Audit", "Permission Change", "Password Reset",
    "Account Creation", "API Call",
    "Logout",
    "Build",
    "Email Sending",
    "Email Receiving",
    "Attachment Upload",
    "Attachment Download",
    "Message",
    "Call",
    "Meeting",
    "Guide Viewing",
    "Guide Completion",
    "Data Sync",
    "Configuration Update",
    "Health Check",
    "Unknown Activity"
]

# Create necessary directories
os.makedirs(os.path.join(BASE_PATH, "data", "output", "codebert", "models"), exist_ok=True)
os.makedirs(PATHS['predictions_folder'], exist_ok=True)

class ZeroShotActivityPredictor:
    def __init__(self, activity_labels, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.activity_labels = activity_labels
        self.activity_embeddings = self._get_activity_embeddings()

    def _get_activity_embeddings(self):
        activity_embeddings = {}
        for activity in self.activity_labels:
            inputs = self.tokenizer(activity, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
            activity_embeddings[activity] = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return activity_embeddings

    def predict(self, activity_text):
        inputs = self.tokenizer(activity_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        activity_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        similarities = {}
        for activity, embedding in self.activity_embeddings.items():
            sim = cosine_similarity(activity_embedding, embedding)[0, 0]  # Extract scalar value
            similarities[activity] = sim

        best_match = max(similarities, key=similarities.get)
        return best_match, similarities[best_match]

class CodeBertTransformer(nn.Module):
    def __init__(self, service_num_labels, activity_num_labels, model_name="microsoft/codebert-base"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.service_classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, service_num_labels)
        )
        self.activity_classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, activity_num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        service_pred = self.service_classifier(pooled_output)
        activity_pred = self.activity_classifier(pooled_output)
        return service_pred, activity_pred, pooled_output

class CodeBertPredictor:
    def __init__(self, model_path, training_data_path, model_name="microsoft/codebert-base"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.predefined_activities = ACTIVITY_LABELS

        self.training_df = pd.read_excel(training_data_path, engine='openpyxl')
        self._prepare_data(model_name)
        self._load_model(model_path)

    def _prepare_data(self, model_name):
        self.training_df['service'] = self.training_df['service'].fillna('Unknown')
        self.training_df['activityType'] = self.training_df['activityType'].fillna('Unknown')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.service_encoder = LabelEncoder()
        self.activity_encoder = LabelEncoder()
        self.service_encoder.fit(self.training_df['service'])
        self.activity_encoder.fit(self.training_df['activityType'])

    def _load_model(self, model_path):
        service_num_labels = len(self.service_encoder.classes_)
        activity_num_labels = len(self.activity_encoder.classes_)
        self.model = CodeBertTransformer(service_num_labels, activity_num_labels).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _prepare_text_features(self, row):
        technical_features = [
            str(row.get('url', '')),
            str(row.get('method', '')),
            str(row.get('headers_Host', '')),
            str(row.get('requestHeaders_Content_Type', '')),
            str(row.get('responseHeaders_Content_Type', ''))
        ]
        return " ".join(technical_features)[:512]

    def predict(self, test_df):
        test_texts = test_df.apply(self._prepare_text_features, axis=1)
        print_status(f"[*] Processing {len(test_texts)} records")
        
        encodings = self.tokenizer(
            test_texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        print_status("[+] Text encoding completed")

        service_confidences = defaultdict(list)
        activity_confidences = defaultdict(list)
        service_activity_confidences = defaultdict(list)
        all_predictions = {
            'predicted_service': [],
            'service_confidence': [],
            'predicted_activity': [],
            'activity_confidence': []
        }

        print_status("[*] Starting predictions...")
        total_records = len(test_texts)
        
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Process in batches of 32
            batch_size = 32
            for i in range(0, len(input_ids), batch_size):
                if i % (batch_size * 10) == 0:  # Print progress every 10 batches
                    progress = (i / len(input_ids)) * 100
                    print_status(f"[*] Progress: {progress:.1f}% - Record {i}/{len(input_ids)}")
                    
                batch_input_ids = input_ids[i:i + batch_size]
                batch_attention_mask = attention_mask[i:i + batch_size]
                
                service_pred, activity_pred, embeddings = self.model(
                    batch_input_ids, batch_attention_mask
                )

                service_probs = F.softmax(service_pred, dim=1)
                activity_probs = F.softmax(activity_pred, dim=1)

                service_max_probs, service_preds = torch.max(service_probs, dim=1)
                activity_max_probs, activity_preds = torch.max(activity_probs, dim=1)

                # Process batch predictions
                for j in range(len(service_preds)):
                    idx = i + j
                    if idx >= total_records:
                        break
                        
                    service = self.service_encoder.inverse_transform([service_preds[j].item()])[0]
                    service_conf = float(service_max_probs[j].item())
                    service_confidences[service].append(service_conf)
                    all_predictions['predicted_service'].append(service)
                    all_predictions['service_confidence'].append(service_conf)

                    activity = self.activity_encoder.inverse_transform([activity_preds[j].item()])[0]
                    if activity.lower() == 'unknown':
                        mapped_activity, confidence = self.zsl_model.predict(test_texts[idx])
                        activity_conf = float(confidence)
                    else:
                        mapped_activity = activity
                        activity_conf = float(activity_max_probs[j].item())

                    activity_confidences[mapped_activity].append(activity_conf)
                    all_predictions['predicted_activity'].append(mapped_activity)
                    all_predictions['activity_confidence'].append(activity_conf)
                    service_activity_confidences[service].append((mapped_activity, activity_conf))

        # Print results
        print_status("\n[+] === Overall Confidence Scores ===")
        overall_service_confidence = np.mean([conf for conf_list in service_confidences.values() for conf in conf_list])
        overall_activity_confidence = np.mean([conf for conf_list in activity_confidences.values() for conf in conf_list])
        print_status(f"[*] Service Confidence: {float(overall_service_confidence):.4f}")
        print_status(f"[*] Activity Confidence: {float(overall_activity_confidence):.4f}")

        print_status("\n[+] === Service Confidence Scores with Activity Confidences ===")
        for service, confidences in sorted(service_confidences.items()):
            service_mean_conf = np.mean(confidences)
            service_count = len(confidences)
            activity_confs = [act_conf for _, act_conf in service_activity_confidences[service]]
            activity_mean_conf = np.mean(activity_confs) if activity_confs else 0.0
            print_status(f"[*] {service:30} Service Conf: {float(service_mean_conf):.4f} | Activity Conf: {float(activity_mean_conf):.4f} (Count: {service_count})")

        print_status("\n[+] === Activity Confidence Scores ===")
        for activity, confidences in sorted(activity_confidences.items()):
            mean_conf = np.mean(confidences)
            count = len(confidences)
            print_status(f"[*] {activity:30} Confidence: {float(mean_conf):.4f} (Count: {count})")

        # Add predictions to DataFrame
        for key, values in all_predictions.items():
            test_df[key] = values

        return test_df

class CodeBertPredictorWithZSL(CodeBertPredictor):
    def __init__(self, model_path, training_data_path, zsl_model, model_name="microsoft/codebert-base"):
        super().__init__(model_path, training_data_path, model_name)
        self.zsl_model = zsl_model

def main():
    try:
        print_status("[*] Starting CodeBERT inference process...")
        
        # Print CUDA information
        print_status(f"[*] Using device: {device}")
        if torch.cuda.is_available():
            print_status(f"[+] GPU detected: {torch.cuda.get_device_name(0)}")
            print_status(f"[+] CUDA Version: {torch.version.cuda}")
        
        # Process each file in the data directory
        all_files = glob.glob(os.path.join(PATHS['test_data'], "*.csv"))
        if not all_files:
            print_status("[!] No CSV files found")
            return

        print_status(f"[+] Found {len(all_files)} files to process")
        
        for file_path in all_files:
            try:
                service_name = os.path.splitext(os.path.basename(file_path))[0]
                print_status(f"\n[*] Processing service: {service_name}")
                
                # Read and process the file
                test_df = pd.read_csv(file_path)
                print_status(f"[+] Loaded {len(test_df)} records from {service_name}")
                
                # Initialize models
                zsl_model = ZeroShotActivityPredictor(ACTIVITY_LABELS)
                predictor = CodeBertPredictorWithZSL(
                    model_path=PATHS['model_file'],
                    training_data_path=PATHS['train_data'],
                    zsl_model=zsl_model
                )
                
                # Make predictions
                predictions_df = predictor.predict(test_df)
                
                # Save predictions
                output_path = os.path.join(PATHS['predictions_folder'], f"{service_name}_predictions.csv")
                predictions_df.to_csv(output_path, index=False)
                print_status(f"[+] Predictions saved to: {os.path.basename(output_path)}")
                
            except Exception as e:
                print_status(f"[!] Error processing {file_path}: {str(e)}")
                continue

        print_status("[+] CodeBERT inference completed successfully!")
        
    except Exception as e:
        print_status(f"[!] Error during inference process: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()