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

# Load and preprocess training dataset
data_path = 'test_data_with_predictions_code_bert_1.csv'
df = pd.read_csv(data_path)

# Get all unique labels before splitting to ensure we have all possible classes
all_services = df['service'].unique()
all_activities = df['activityType'].unique()

# Create train and test splits of the original data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save test set to a new file
test_data_path = 'test_set.csv'
test_df.to_csv(test_data_path, index=False)
print(f"Test set saved to {test_data_path}")

# Print unique values in service and activityType columns
print("\nUnique services:", train_df['service'].unique())
print("Unique activities:", train_df['activityType'].unique())

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

# Clean and convert labels to strings
train_df['service'] = train_df['service'].astype(str)
train_df['activityType'] = train_df['activityType'].astype(str)

# Encode labels using all possible classes
le_service = LabelEncoder()
le_service.fit(np.concatenate([all_services, train_df['service']]))
train_df['service_encoded'] = le_service.transform(train_df['service'])

le_activity = LabelEncoder()
le_activity.fit(np.concatenate([all_activities, train_df['activityType']]))
train_df['activityType_encoded'] = le_activity.transform(train_df['activityType'])

# Split training data for model validation
X_train, X_val, y_train_service, y_val_service = train_test_split(
   train_df['combined_headers'], train_df['service_encoded'], test_size=0.2, random_state=42
)
_, _, y_train_activity, y_val_activity = train_test_split(
   train_df['combined_headers'], train_df['activityType_encoded'], test_size=0.2, random_state=42
)

# Build and train Service classification model
pipeline_service = make_pipeline(
   TfidfVectorizer(max_features=5000),
   RandomForestClassifier(n_estimators=100, random_state=42)
)
pipeline_service.fit(X_train, y_train_service)
y_pred_service = pipeline_service.predict(X_val)

# Build and train Activity Type classification model
pipeline_activity = make_pipeline(
   TfidfVectorizer(max_features=5000),
   RandomForestClassifier(n_estimators=100, random_state=42)
)
pipeline_activity.fit(X_train, y_train_activity)
y_pred_activity = pipeline_activity.predict(X_val)

# Get unique classes present in validation set
unique_services_val = np.unique(y_val_service)
unique_activities_val = np.unique(y_val_activity)

# Get class names for only the classes present in validation set and ensure they're strings
service_names = [str(name) for name in le_service.classes_[unique_services_val]]
activity_names = [str(name) for name in le_activity.classes_[unique_activities_val]]

# Display metrics
print("\nService Classification Report (Validation Set):")
print(classification_report(y_val_service, y_pred_service, 
                          target_names=service_names,
                          labels=unique_services_val))

print("\nActivity Type Classification Report (Validation Set):")
print(classification_report(y_val_activity, y_pred_activity, 
                          target_names=activity_names,
                          labels=unique_activities_val))

# Confusion Matrices
plt.figure(figsize=(15, 10))
sns.heatmap(confusion_matrix(y_val_service, y_pred_service), 
            annot=True, fmt='d', cmap='Blues')
plt.title("Service Confusion Matrix (Validation Set)")
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(confusion_matrix(y_val_activity, y_pred_activity), 
            annot=True, fmt='d', cmap='Blues')
plt.title("Activity Type Confusion Matrix (Validation Set)")
plt.show()

# Function to predict on test data
def predict_test_data(test_df, pipeline_service, pipeline_activity, le_service, le_activity):
   test_df['combined_headers'] = (
       test_df['headers_Host'].fillna('') + ' ' +
       test_df['url'].fillna('') + ' ' +
       test_df['method'].fillna('') + ' ' +
       test_df['requestHeaders_Origin'].fillna('') + ' ' +
       test_df['requestHeaders_Content_Type'].fillna('') + ' ' +
       test_df['responseHeaders_Content_Type'].fillna('') + ' ' +
       test_df['requestHeaders_Referer'].fillna('') + ' ' +
       test_df['requestHeaders_Accept'].fillna('') + ' ' 
   )

   # Predictions
   service_pred = pipeline_service.predict(test_df['combined_headers'])
   activity_pred = pipeline_activity.predict(test_df['combined_headers'])

   # Decode predictions
   test_df['predicted_service'] = le_service.inverse_transform(service_pred)
   test_df['predicted_activityType'] = le_activity.inverse_transform(activity_pred)

   # Save results
   output_path = 'test_predictions.csv'
   test_df.to_csv(output_path, index=False)
   print(f"\nTest predictions saved to '{output_path}'")

   return test_df

# Process test set
test_df['service'] = test_df['service'].astype(str)
test_df['activityType'] = test_df['activityType'].astype(str)
test_df['combined_headers'] = (
    test_df['headers_Host'].fillna('') + ' ' +
    test_df['url'].fillna('') + ' ' +
    test_df['method'].fillna('') + ' ' +
    test_df['requestHeaders_Origin'].fillna('') + ' ' +
    test_df['requestHeaders_Content_Type'].fillna('') + ' ' +
    test_df['responseHeaders_Content_Type'].fillna('') + ' ' +
    test_df['requestHeaders_Referer'].fillna('') + ' ' +
    test_df['requestHeaders_Accept'].fillna('') + ' ' 
)

# Get predictions on test set
test_service_pred = pipeline_service.predict(test_df['combined_headers'])
test_activity_pred = pipeline_activity.predict(test_df['combined_headers'])

# Calculate overall accuracy for samples where we have labels
test_service_mask = np.isin(test_df['service'], le_service.classes_)
test_activity_mask = np.isin(test_df['activityType'], le_activity.classes_)

test_service_accuracy = accuracy_score(
    le_service.transform(test_df.loc[test_service_mask, 'service']),
    test_service_pred[test_service_mask]
)

test_activity_accuracy = accuracy_score(
    le_activity.transform(test_df.loc[test_activity_mask, 'activityType']),
    test_activity_pred[test_activity_mask]
)

# Print results
print("\nTest Set Results:")
print(f"Number of services in test set: {len(test_df['service'].unique())}")
print(f"Number of activities in test set: {len(test_df['activityType'].unique())}")
print(f"Number of samples with known service labels: {test_service_mask.sum()}/{len(test_df)}")
print(f"Number of samples with known activity labels: {test_activity_mask.sum()}/{len(test_df)}")

print(f"\nTest Set Overall Accuracy (for known labels):")
print(f"Service Classification: {test_service_accuracy:.4f}")
print(f"Activity Classification: {test_activity_accuracy:.4f}")

# Print unseen labels if any exist
unseen_services = set(test_df['service'].unique()) - set(le_service.classes_)
unseen_activities = set(test_df['activityType'].unique()) - set(le_activity.classes_)

if unseen_services:
    print("\nUnseen services in test set:", unseen_services)
if unseen_activities:
    print("Unseen activities in test set:", unseen_activities)