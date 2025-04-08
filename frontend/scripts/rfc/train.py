import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
   accuracy_score, classification_report, confusion_matrix,
   precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path configuration
PATHS = {
    'input_data': os.path.join(BASE_PATH, 'data', 'output', 'codebert'),
    'output_data': os.path.join(BASE_PATH, 'data', 'output', 'rfc'),
    'models': os.path.join(BASE_PATH, 'data', 'models', 'rfc'),
    'test_data': os.path.join(BASE_PATH, 'data', 'output', 'rfc', 'test')
}

# Create necessary directories
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

def print_status(message: str):
    """Print status message with immediate flush"""
    print(message, flush=True)

def train_model(input_file: str):
    print_status(f"Loading data from {input_file}")
    
    # Get current timestamp for model naming
    timestamp = datetime.now().strftime("%Y%m%d")
    model_prefix = f"rfc_model_{timestamp}"
    
    # Load and preprocess training dataset
    df = pd.read_csv(os.path.join(PATHS['input_data'], input_file))

    # Get all unique labels before splitting
    all_services = df['service'].unique()
    all_activities = df['activityType'].unique()

    print_status(f"Found {len(all_services)} unique services and {len(all_activities)} unique activities")

    # Create train and test splits
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save test set
    test_data_path = os.path.join(PATHS['test_data'], f'test_set_{timestamp}.csv')
    test_df.to_csv(test_data_path, index=False)
    print_status(f"Test set saved to {test_data_path}")

    # Process training data
    train_df['combined_headers'] = (
       train_df['headers_Host'].fillna('') + ' ' +
       train_df['url'].fillna('') + ' ' +
       train_df['method'].fillna('') + ' ' +
       train_df['requestHeaders_Origin'].fillna('') + ' ' +
       train_df['requestHeaders_Content_Type'].fillna('') + ' ' +
       train_df['responseHeaders_Content_Type'].fillna('') + ' ' +
       train_df['requestHeaders_Referer'].fillna('') + ' ' +
       train_df['requestHeaders_Accept'].fillna('') + ' ' 
    )

    # Clean and encode labels
    train_df['service'] = train_df['service'].astype(str)
    train_df['activityType'] = train_df['activityType'].astype(str)

    # Encode labels using all possible classes
    le_service = LabelEncoder()
    le_service.fit(np.concatenate([all_services, train_df['service']]))
    train_df['service_encoded'] = le_service.transform(train_df['service'])

    le_activity = LabelEncoder()
    le_activity.fit(np.concatenate([all_activities, train_df['activityType']]))
    train_df['activityType_encoded'] = le_activity.transform(train_df['activityType'])

    # Split training data for validation
    X_train, X_val, y_train_service, y_val_service = train_test_split(
       train_df['combined_headers'], train_df['service_encoded'], test_size=0.2, random_state=42
    )
    _, _, y_train_activity, y_val_activity = train_test_split(
       train_df['combined_headers'], train_df['activityType_encoded'], test_size=0.2, random_state=42
    )

    print_status("Training service classification model...")
    # Train Service classification model
    pipeline_service = make_pipeline(
       TfidfVectorizer(max_features=5000),
       RandomForestClassifier(n_estimators=100, random_state=42)
    )
    pipeline_service.fit(X_train, y_train_service)
    y_pred_service = pipeline_service.predict(X_val)

    print_status("Training activity classification model...")
    # Train Activity Type classification model
    pipeline_activity = make_pipeline(
       TfidfVectorizer(max_features=5000),
       RandomForestClassifier(n_estimators=100, random_state=42)
    )
    pipeline_activity.fit(X_train, y_train_activity)
    y_pred_activity = pipeline_activity.predict(X_val)

    # Get validation metrics
    service_accuracy = accuracy_score(y_val_service, y_pred_service)
    activity_accuracy = accuracy_score(y_val_activity, y_pred_activity)

    print_status(f"Service Classification Accuracy: {service_accuracy:.4f}")
    print_status(f"Activity Classification Accuracy: {activity_accuracy:.4f}")

    # Save models and label encoders
    import joblib
    
    models_path = PATHS['models']
    os.makedirs(models_path, exist_ok=True)
    
    joblib.dump(pipeline_service, os.path.join(models_path, 'service_classifier.joblib'))
    joblib.dump(pipeline_activity, os.path.join(models_path, 'activity_classifier.joblib'))
    joblib.dump(le_service, os.path.join(models_path, 'service_encoder.joblib'))
    joblib.dump(le_activity, os.path.join(models_path, 'activity_encoder.joblib'))
    
    print_status("Models and encoders saved successfully")

    return {
        'service_accuracy': service_accuracy,
        'activity_accuracy': activity_accuracy,
        'unique_services': len(all_services),
        'unique_activities': len(all_activities)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        train_model(input_file)
    else:
        print("Please provide an input file name")