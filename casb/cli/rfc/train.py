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
import joblib # Added for saving models/encoders

# Ensure output directory exists
output_dir = 'rfc_results'
os.makedirs(output_dir, exist_ok=True)

# --- Helper Functions ---

def plot_confusion_matrix(y_true, y_pred, labels, class_names, title, filename):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # Close the plot to prevent displaying inline if not desired
    print(f"Saved confusion matrix to {filename}")

def plot_feature_importance(pipeline, feature_names, top_n, title, filename):
    """Generates and saves a feature importance plot."""
    model = pipeline.named_steps['randomforestclassifier']
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, top_n / 2))
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved feature importance plot to {filename}")

def save_classification_report(y_true, y_pred, labels, class_names, title, filename):
    """Generates, prints, and saves a classification report as CSV."""
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    print(f"{title}:")
    print(report_df)
    report_df.to_csv(filename)
    print(f"Saved classification report to {filename}")

# --- Main Script ---

# Load and preprocess training dataset
data_path = 'test_data_with_predictions_code_bert_1.csv'
df = pd.read_csv(data_path)

# Get all unique labels before splitting to ensure we have all possible classes
# Ensure labels are strings and handle potential NaN values
all_services = df['service'].dropna().astype(str).unique()
all_activities = df['activityType'].dropna().astype(str).unique()

# Create train and test splits of the original data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # Removed stratify due to single-member classes

# Save test set to a new file
test_data_path = os.path.join(output_dir, 'test_set.csv')
test_df.to_csv(test_data_path, index=False)
print(f"Test set saved to {test_data_path}")

# Print unique values in service and activityType columns after split
print("Unique services in Training Set:", train_df['service'].dropna().astype(str).unique())
print("Unique activities in Training Set:", train_df['activityType'].dropna().astype(str).unique())
print("Unique services in Test Set:", test_df['service'].dropna().astype(str).unique())
print("Unique activities in Test Set:", test_df['activityType'].dropna().astype(str).unique())

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
).astype(str) # Ensure combined headers are strings

# Clean and convert labels to strings for training data
train_df['service'] = train_df['service'].fillna('Unknown').astype(str)
train_df['activityType'] = train_df['activityType'].fillna('Unknown').astype(str)

# Encode labels using all possible classes discovered initially, plus 'Unknown'
le_service = LabelEncoder()
le_service.fit(np.append(all_services, 'Unknown'))
train_df['service_encoded'] = le_service.transform(train_df['service'])
joblib.dump(le_service, os.path.join(output_dir, 'label_encoder_service.joblib')) # Save encoder

le_activity = LabelEncoder()
le_activity.fit(np.append(all_activities, 'Unknown'))
train_df['activityType_encoded'] = le_activity.transform(train_df['activityType'])
joblib.dump(le_activity, os.path.join(output_dir, 'label_encoder_activity.joblib')) # Save encoder


# Split training data for model validation (using already processed train_df)
X_train, X_val, y_train_service, y_val_service = train_test_split(
   train_df['combined_headers'], train_df['service_encoded'], 
   test_size=0.25, random_state=42, # 0.25 of 0.8 = 0.2 original data for validation
   stratify=train_df['service_encoded']
)
_, _, y_train_activity, y_val_activity = train_test_split(
   train_df['combined_headers'], train_df['activityType_encoded'], 
   test_size=0.25, random_state=42, # Keep splits consistent
   stratify=train_df['activityType_encoded']
)

# --- Service Model Training and Validation ---
print("--- Training Service Classifier ---")
pipeline_service = make_pipeline(
   TfidfVectorizer(max_features=5000, stop_words='english'),
   RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1) # Increased estimators, balanced weights
)
pipeline_service.fit(X_train, y_train_service)
joblib.dump(pipeline_service, os.path.join(output_dir, 'pipeline_service.joblib')) # Save model
y_pred_service_val = pipeline_service.predict(X_val)

# Get TF-IDF feature names for plotting
tfidf_service_features = pipeline_service.named_steps['tfidfvectorizer'].get_feature_names_out()

# Validation Evaluation - Service
val_service_labels = np.unique(np.concatenate((y_val_service, y_pred_service_val)))
val_service_names = le_service.inverse_transform(val_service_labels)

save_classification_report(y_val_service, y_pred_service_val, 
                           labels=val_service_labels, 
                           class_names=val_service_names,
                           title="Service Classification Report (Validation Set)",
                           filename=os.path.join(output_dir, 'validation_service_report.csv'))

plot_confusion_matrix(y_val_service, y_pred_service_val,
                      labels=val_service_labels,
                      class_names=val_service_names,
                      title="Service Confusion Matrix (Validation Set)",
                      filename=os.path.join(output_dir, 'validation_service_cm.png'))

plot_feature_importance(pipeline_service, tfidf_service_features, 
                        top_n=20, 
                        title="Top 20 Features for Service Classification",
                        filename=os.path.join(output_dir, 'feature_importance_service.png'))


# --- Activity Type Model Training and Validation ---
print("--- Training Activity Type Classifier ---")
pipeline_activity = make_pipeline(
   TfidfVectorizer(max_features=5000, stop_words='english'),
   RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1) # Increased estimators, balanced weights
)
pipeline_activity.fit(X_train, y_train_activity)
joblib.dump(pipeline_activity, os.path.join(output_dir, 'pipeline_activity.joblib')) # Save model
y_pred_activity_val = pipeline_activity.predict(X_val)

# Get TF-IDF feature names for plotting
tfidf_activity_features = pipeline_activity.named_steps['tfidfvectorizer'].get_feature_names_out()

# Validation Evaluation - Activity Type
val_activity_labels = np.unique(np.concatenate((y_val_activity, y_pred_activity_val)))
val_activity_names = le_activity.inverse_transform(val_activity_labels)

save_classification_report(y_val_activity, y_pred_activity_val, 
                           labels=val_activity_labels,
                           class_names=val_activity_names,
                           title="Activity Type Classification Report (Validation Set)",
                           filename=os.path.join(output_dir, 'validation_activity_report.csv'))

plot_confusion_matrix(y_val_activity, y_pred_activity_val,
                      labels=val_activity_labels,
                      class_names=val_activity_names,
                      title="Activity Type Confusion Matrix (Validation Set)",
                      filename=os.path.join(output_dir, 'validation_activity_cm.png'))

plot_feature_importance(pipeline_activity, tfidf_activity_features, 
                        top_n=20, 
                        title="Top 20 Features for Activity Type Classification",
                        filename=os.path.join(output_dir, 'feature_importance_activity.png'))

# --- Test Set Evaluation ---
print("--- Evaluating on Test Set ---")

# Preprocess test data (handle NaNs and ensure string type)
test_df['service'] = test_df['service'].fillna('Unknown').astype(str)
test_df['activityType'] = test_df['activityType'].fillna('Unknown').astype(str)
test_df['combined_headers'] = (
    test_df['headers_Host'].fillna('') + ' ' +
    test_df['url'].fillna('') + ' ' +
    test_df['method'].fillna('') + ' ' +
    test_df['requestHeaders_Origin'].fillna('') + ' ' +
    test_df['requestHeaders_Content_Type'].fillna('') + ' ' +
    test_df['responseHeaders_Content_Type'].fillna('') + ' ' +
    test_df['requestHeaders_Referer'].fillna('') + ' ' +
    test_df['requestHeaders_Accept'].fillna('') + ' '
).astype(str)

# Predict on test set
test_pred_service_encoded = pipeline_service.predict(test_df['combined_headers'])
test_pred_activity_encoded = pipeline_activity.predict(test_df['combined_headers'])

# Add predictions to test DataFrame
test_df['predicted_service'] = le_service.inverse_transform(test_pred_service_encoded)
test_df['predicted_activityType'] = le_activity.inverse_transform(test_pred_activity_encoded)

# Save test predictions
output_path = os.path.join(output_dir, 'test_predictions.csv')
test_df.to_csv(output_path, index=False)
print(f"Test predictions saved to '{output_path}'")

# Prepare true labels for test evaluation (handle unseen labels)
test_df['service_encoded'] = test_df['service'].apply(lambda x: le_service.transform([x])[0] if x in le_service.classes_ else -1) # -1 for unseen
test_df['activityType_encoded'] = test_df['activityType'].apply(lambda x: le_activity.transform([x])[0] if x in le_activity.classes_ else -1) # -1 for unseen

# Filter out rows with unseen true labels for standard metric calculation
test_service_mask = test_df['service_encoded'] != -1
test_activity_mask = test_df['activityType_encoded'] != -1

y_true_service_test = test_df.loc[test_service_mask, 'service_encoded']
y_pred_service_test = test_pred_service_encoded[test_service_mask]

y_true_activity_test = test_df.loc[test_activity_mask, 'activityType_encoded']
y_pred_activity_test = test_pred_activity_encoded[test_activity_mask]

# Test Evaluation - Service
if len(y_true_service_test) > 0:
    test_service_labels = np.unique(np.concatenate((y_true_service_test, y_pred_service_test)))
    test_service_names = le_service.inverse_transform(test_service_labels)
    
    save_classification_report(y_true_service_test, y_pred_service_test,
                               labels=test_service_labels,
                               class_names=test_service_names,
                               title="Service Classification Report (Test Set - Known Labels)",
                               filename=os.path.join(output_dir, 'test_service_report.csv'))

    plot_confusion_matrix(y_true_service_test, y_pred_service_test,
                          labels=test_service_labels,
                          class_names=test_service_names,
                          title="Service Confusion Matrix (Test Set - Known Labels)",
                          filename=os.path.join(output_dir, 'test_service_cm.png'))
    
    test_service_accuracy = accuracy_score(y_true_service_test, y_pred_service_test)
    print(f"Test Set Overall Accuracy (Service - Known Labels): {test_service_accuracy:.4f}")
else:
    print("No known service labels in the test set to evaluate.")


# Test Evaluation - Activity Type
if len(y_true_activity_test) > 0:
    test_activity_labels = np.unique(np.concatenate((y_true_activity_test, y_pred_activity_test)))
    test_activity_names = le_activity.inverse_transform(test_activity_labels)

    save_classification_report(y_true_activity_test, y_pred_activity_test,
                               labels=test_activity_labels,
                               class_names=test_activity_names,
                               title="Activity Type Classification Report (Test Set - Known Labels)",
                               filename=os.path.join(output_dir, 'test_activity_report.csv'))

    plot_confusion_matrix(y_true_activity_test, y_pred_activity_test,
                          labels=test_activity_labels,
                          class_names=test_activity_names,
                          title="Activity Type Confusion Matrix (Test Set - Known Labels)",
                          filename=os.path.join(output_dir, 'test_activity_cm.png'))

    test_activity_accuracy = accuracy_score(y_true_activity_test, y_pred_activity_test)
    print(f"Test Set Overall Accuracy (Activity - Known Labels): {test_activity_accuracy:.4f}")
else:
    print("No known activity labels in the test set to evaluate.")


# Report on unseen labels encountered during testing
unseen_services = set(test_df[test_df['service_encoded'] == -1]['service'].unique())
unseen_activities = set(test_df[test_df['activityType_encoded'] == -1]['activityType'].unique())

if unseen_services:
    print("Unseen services encountered in test set:", unseen_services)
if unseen_activities:
    print("Unseen activities encountered in test set:", unseen_activities)

print(f"All results saved in '{output_dir}' directory.")


# --- Removed old plotting/printing sections ---
# (The old plt.show() and print(classification_report(...)) calls are replaced by the helper functions)