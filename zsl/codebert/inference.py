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

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Paths configuration
PATHS = {
    'train_data': os.path.join(BASE_PATH, "data", "labelled", "train_set.xlsx"),
    'test_data': os.path.join(BASE_PATH, "data", "labelled", "test_set.xlsx"),
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
        encodings = self.tokenizer(
            test_texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )

        service_confidences = defaultdict(list)
        activity_confidences = defaultdict(list)
        # Create a new dictionary to store service-activity confidence pairs
        service_activity_confidences = defaultdict(list)
        all_predictions = {
            'predicted_service': [],
            'service_confidence': [],
            'predicted_activity': [],
            'activity_confidence': []
        }

        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            service_pred, activity_pred, embeddings = self.model(input_ids, attention_mask)

            service_probs = F.softmax(service_pred, dim=1)
            activity_probs = F.softmax(activity_pred, dim=1)

            service_max_probs, service_preds = torch.max(service_probs, dim=1)
            activity_max_probs, activity_preds = torch.max(activity_probs, dim=1)

            service_max_probs = service_max_probs.cpu().numpy()
            service_preds = service_preds.cpu().numpy()
            activity_max_probs = activity_max_probs.cpu().numpy()
            activity_preds = activity_preds.cpu().numpy()

            for i in range(len(service_preds)):
                # Handle service predictions
                service = self.service_encoder.inverse_transform([service_preds[i]])[0]
                service_conf = float(service_max_probs[i])
                service_confidences[service].append(service_conf)
                all_predictions['predicted_service'].append(service)
                all_predictions['service_confidence'].append(service_conf)

                # Handle activity predictions
                activity = self.activity_encoder.inverse_transform([activity_preds[i]])[0]
                if activity.lower() == 'unknown':
                    mapped_activity, confidence = self.zsl_model.predict(test_texts[i])
                    activity_conf = float(confidence.item())  # Extract scalar value from numpy array
                else:
                    mapped_activity = activity
                    activity_conf = float(activity_max_probs[i])

                activity_confidences[mapped_activity].append(activity_conf)
                all_predictions['predicted_activity'].append(mapped_activity)
                all_predictions['activity_confidence'].append(activity_conf)

                # Store the service-activity pair with both confidences
                service_activity_confidences[service].append((mapped_activity, activity_conf))

            # Calculate overall scores
            overall_service_confidence = np.mean(service_max_probs)
            overall_activity_confidence = np.mean([conf for conf_list in activity_confidences.values() for conf in conf_list])

            # Print results
            print("\n=== Overall Confidence Scores ===")
            print(f"Service Confidence: {float(overall_service_confidence):.4f}")
            print(f"Activity Confidence: {float(overall_activity_confidence):.4f}")

            print("\n=== Service Confidence Scores with Activity Confidences ===")
            for service, confidences in sorted(service_confidences.items()):
                service_mean_conf = np.mean(confidences)
                service_count = len(confidences)

                # Calculate average activity confidence for this service
                activity_confs = [act_conf for _, act_conf in service_activity_confidences[service]]
                activity_mean_conf = np.mean(activity_confs) if activity_confs else 0.0

                # Create a formatted string with both service and activity confidences
                print(f"{service:30} Service Conf: {float(service_mean_conf):.4f} | Activity Conf: {float(activity_mean_conf):.4f} (Count: {service_count})")

                # Optionally print detailed activity breakdown for each service
                activities_for_service = defaultdict(list)
                for activity, conf in service_activity_confidences[service]:
                    activities_for_service[activity].append(conf)

                # Print top activities for this service
                for activity, act_confs in sorted(activities_for_service.items(),
                                                key=lambda x: np.mean(x[1]),
                                                reverse=True)[:3]:  # Show top 3 activities
                    act_mean = np.mean(act_confs)
                    act_count = len(act_confs)
                    print(f"  - {activity:25} Activity Conf: {float(act_mean):.4f} (Count: {act_count})")

            print("\n=== Activity Confidence Scores ===")
            for activity, confidences in sorted(activity_confidences.items()):
                mean_conf = np.mean(confidences)
                count = len(confidences)
                print(f"{activity:30} Confidence: {float(mean_conf):.4f} (Count: {count})")

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
        predefined_activities = ACTIVITY_LABELS
        zsl_model = ZeroShotActivityPredictor(predefined_activities)
        test_df = pd.read_excel(PATHS['test_data'], engine='openpyxl')
        predictor = CodeBertPredictorWithZSL(
            model_path=PATHS['model_file'],
            training_data_path=PATHS['train_data'],
            zsl_model=zsl_model
        )
        predictions_df = predictor.predict(test_df)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()