import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import emlearn
import joblib
import re
import c_code_templates

MAX_FEATURES_CONFIG = 127 
EMLEARN_MODEL_METHOD = 'inline'
NUM_C_EXAMPLE_SAMPLES = 1000
N_C_TEST_SAMPLES = 1000    
C_HASH_TABLE_SIZE = 8192
C_MAX_BUCKET_SIZE = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))) 
INPUT_DATA_BASE_DIR = os.path.join(PROJECT_ROOT_PATH, "data", "output", "codebert", "predictions")

EMLEARN_OUTPUT_DIR = os.path.join(PROJECT_ROOT_PATH, "data", "output", "rfc", "em-codegen")

EMLEARN_INCLUDE_DIR = os.path.join(EMLEARN_OUTPUT_DIR, "include")
C_INFERENCE_FILE_PATH = os.path.join(EMLEARN_OUTPUT_DIR, "rfc_em_inference.c")
MODELS_SAVE_DIR = os.path.join(EMLEARN_OUTPUT_DIR, "models")
LABEL_MAPPINGS_FILE_PATH = os.path.join(EMLEARN_OUTPUT_DIR, "label_mappings.txt")
FEATURE_ENGINEERING_H_PATH = os.path.join(EMLEARN_INCLUDE_DIR, "feature_engineering.h")

def print_status(message: str):
    print(message, flush=True)

def create_directories():
    os.makedirs(EMLEARN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EMLEARN_INCLUDE_DIR, exist_ok=True)
    os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
    print_status(f"Ensured output directories exist: {EMLEARN_OUTPUT_DIR}, {EMLEARN_INCLUDE_DIR}")

def load_and_preprocess_data():
    print_status(f"Scanning for prediction files in: {INPUT_DATA_BASE_DIR}")
    prediction_files = [f for f in os.listdir(INPUT_DATA_BASE_DIR) if f.endswith('_predictions.csv')]
    
    if not prediction_files:
        print_status(f"ERROR: No '_predictions.csv' files found in {INPUT_DATA_BASE_DIR}")
        sys.exit(1)
    
    print_status(f"Found prediction files: {prediction_files}")
    dfs = []
    for file_name in prediction_files:
        file_path = os.path.join(INPUT_DATA_BASE_DIR, file_name)
        try:
            df_temp = pd.read_csv(file_path)
            dfs.append(df_temp)
            print_status(f"Loaded {len(df_temp)} records from {file_name}.")
        except Exception as e:
            print_status(f"Warning: Could not load or process {file_name}. Error: {e}")
            continue
    
    if not dfs:
        print_status(f"ERROR: No data could be loaded from any of the found CSV files.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print_status(f"Total records loaded from all files: {len(df)}")

    df['combined_headers_for_py'] = (
        df['headers_Host'].fillna('').astype(str) + ' ' +
        df['url'].fillna('').astype(str) + ' ' +
        df['method'].fillna('').astype(str) + ' ' +
        df['requestHeaders_Origin'].fillna('').astype(str) + ' ' +
        df['requestHeaders_Content_Type'].fillna('').astype(str) + ' ' +
        df['responseHeaders_Content_Type'].fillna('').astype(str) + ' ' +
        df['requestHeaders_Referer'].fillna('').astype(str) + ' ' +
        df['requestHeaders_Accept'].fillna('').astype(str)
    ).str.lower()

    df['service'] = df['service'].astype(str)
    df['activityType'] = df['activityType'].astype(str)

    if len(df) < 2:
        print_status("ERROR: Not enough data to perform train/test split.")
        sys.exit(1)
        
    # Perform train/test split
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    print_status(f"Training set size: {len(train_df)}, Evaluation set size: {len(eval_df)}")

    c_example_samples_df = eval_df.head(NUM_C_EXAMPLE_SAMPLES).copy()
    c_test_set_df = eval_df.head(N_C_TEST_SAMPLES).copy() # Data for C accuracy calculation

    return train_df, eval_df, c_example_samples_df, c_test_set_df

def fnv1a_hash_py(string_to_hash):
    hash_val = 2166136261
    for byte_char in string_to_hash.encode('utf-8'):
        hash_val ^= byte_char
        hash_val *= 16777619
        hash_val &= 0xFFFFFFFF
    return hash_val

def generate_feature_engineering_header_content(vectorizer: CountVectorizer):
    vocabulary = vectorizer.vocabulary_
    feature_name_list = [None] * len(vocabulary)
    for term, index in vocabulary.items():
        if index < len(feature_name_list):
            feature_name_list[index] = term
    
    indexed_feature_terms = [(term, idx) for idx, term in enumerate(feature_name_list) if term is not None]

    feature_table_entries_list = []
    for term, c_feature_index in indexed_feature_terms:
        escaped_term = term.replace('\\', '\\\\').replace('"', '\\"')
        feature_table_entries_list.append(f'    {{ "{escaped_term}", {c_feature_index} }}')
    
    feature_table_entries_dynamic = ",\n".join(feature_table_entries_list)

    c_buckets = [[] for _ in range(C_HASH_TABLE_SIZE)]
    for table_idx, (term, _) in enumerate(indexed_feature_terms):
        hash_val = fnv1a_hash_py(term) % C_HASH_TABLE_SIZE
        if len(c_buckets[hash_val]) < C_MAX_BUCKET_SIZE:
            c_buckets[hash_val].append(table_idx)

    hash_bucket_entries_list = []
    for bucket_list in c_buckets:
        indices_str = ", ".join(map(str, bucket_list + [-1] * (C_MAX_BUCKET_SIZE - len(bucket_list))))
        hash_bucket_entries_list.append(f"    {{ {{ {indices_str} }}, {len(bucket_list)} }}")
    hash_bucket_entries_dynamic = ",\n".join(hash_bucket_entries_list)

    return c_code_templates.C_FEATURE_ENGINEERING_H_TEMPLATE.format(
        MAX_C_FEATURES_DEF=MAX_FEATURES_CONFIG,
        C_HASH_TABLE_SIZE_DEF=C_HASH_TABLE_SIZE,
        C_MAX_BUCKET_SIZE_DEF=C_MAX_BUCKET_SIZE,
        feature_table_entries_dynamic=feature_table_entries_dynamic,
        hash_bucket_entries_dynamic=hash_bucket_entries_dynamic
    )

def generate_c_main_with_test_samples(c_test_set_df: pd.DataFrame, service_model_c_name: str, activity_model_c_name: str, le_service: LabelEncoder, le_activity: LabelEncoder):
    c_code_block = "" 

    # --- Initial Single Sample Test ---
    if not c_test_set_df.empty:
        first_sample_row = c_test_set_df.iloc[0]
        host = first_sample_row['headers_Host'].replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row['headers_Host']) else ""
        url = first_sample_row['url'].replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row['url']) else ""
        method = first_sample_row['method'].replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row['method']) else ""
        origin = first_sample_row.get('requestHeaders_Origin', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row.get('requestHeaders_Origin')) else ""
        req_content = first_sample_row.get('requestHeaders_Content_Type', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row.get('requestHeaders_Content_Type')) else ""
        res_content = first_sample_row.get('responseHeaders_Content_Type', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row.get('responseHeaders_Content_Type')) else ""
        referer = first_sample_row.get('requestHeaders_Referer', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row.get('requestHeaders_Referer')) else ""
        accept = first_sample_row.get('requestHeaders_Accept', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(first_sample_row.get('requestHeaders_Accept')) else ""

        c_code_block += "    printf(\"Processing example test sample...\\n\");\n"
        c_code_block += f"    s_host = \"{host}\"; s_url = \"{url}\"; s_method = \"{method}\";\n"
        c_code_block += f"    s_origin = \"{origin}\"; s_req_content_type = \"{req_content}\"; s_res_content_type = \"{res_content}\";\n"
        c_code_block += f"    s_referer = \"{referer}\"; s_accept = \"{accept}\";\n"
        c_code_block += (
            f"    printf(\"  Input Host: %s\\n  Input URL: %s\\n  Input Method: %s\\n" \
            f"  Input Origin: %s\\n  Input Req Content-Type: %s\\n  Input Res Content-Type: %s\\n" \
            f"  Input Referer: %s\\n  Input Accept: %s\\n\\n\", " \
            f"s_host, s_url, s_method, s_origin, s_req_content_type, s_res_content_type, s_referer, s_accept);\n"
        )
        c_code_block += "    extract_features_from_strings(s_host, s_url, s_method, s_origin, s_req_content_type, s_res_content_type, s_referer, s_accept, features);\n"
        c_code_block += f"    service_idx = {service_model_c_name}_predict(features, MAX_C_FEATURES);\n"
        c_code_block += f"    activity_idx = {activity_model_c_name}_predict(features, MAX_C_FEATURES);\n"
        c_code_block += "    printf(\"Predictions for example sample:\\n\");\n"
        c_code_block += "    printf(\"  Service Index: %d\\n\", service_idx);\n"
        c_code_block += "    printf(\"  Activity Index: %d\\n\\n\", activity_idx);\n"
    else:
        c_code_block += "    printf(\"No example sample available to display.\\n\");\n"

    # --- Accuracy Testing Loop ---
    if not c_test_set_df.empty:
        true_service_labels_c = "    const int16_t true_service_labels[] = { " + ", ".join(map(str, c_test_set_df['service_encoded'].tolist())) + " };\n"
        true_activity_labels_c = "    const int16_t true_activity_labels[] = { " + ", ".join(map(str, c_test_set_df['activityType_encoded'].tolist())) + " };\n"
        c_code_block += true_service_labels_c
        c_code_block += true_activity_labels_c
        c_code_block += f"    int num_c_test_samples = {len(c_test_set_df)};\n"
        c_code_block += "    int correct_service = 0;\n    int correct_activity = 0;\n\n"
        c_code_block += "    printf(\"\\n--- Running C-Side Accuracy Tests ---\\n\");\n"
        
        c_sample_idx_counter = 0 
        for _, row in c_test_set_df.iterrows(): 
            host = row['headers_Host'].replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row['headers_Host']) else ""
            url = row['url'].replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row['url']) else ""
            method = row['method'].replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row['method']) else ""
            origin = row.get('requestHeaders_Origin', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row.get('requestHeaders_Origin')) else ""
            req_content = row.get('requestHeaders_Content_Type', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row.get('requestHeaders_Content_Type')) else ""
            res_content = row.get('responseHeaders_Content_Type', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row.get('responseHeaders_Content_Type')) else ""
            referer = row.get('requestHeaders_Referer', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row.get('requestHeaders_Referer')) else ""
            accept = row.get('requestHeaders_Accept', "").replace('\\', '\\\\').replace('"', '\\"') if pd.notna(row.get('requestHeaders_Accept')) else ""

            c_code_block += f"    // Processing C Test Sample {c_sample_idx_counter}\n"
            c_code_block += f"    s_host = \"{host}\"; s_url = \"{url}\"; s_method = \"{method}\";\n"
            c_code_block += f"    s_origin = \"{origin}\"; s_req_content_type = \"{req_content}\"; s_res_content_type = \"{res_content}\";\n"
            c_code_block += f"    s_referer = \"{referer}\"; s_accept = \"{accept}\";\n"
            c_code_block += "    extract_features_from_strings(s_host, s_url, s_method, s_origin, s_req_content_type, s_res_content_type, s_referer, s_accept, features);\n"
            c_code_block += f"    service_idx = {service_model_c_name}_predict(features, MAX_C_FEATURES);\n"
            c_code_block += f"    activity_idx = {activity_model_c_name}_predict(features, MAX_C_FEATURES);\n"
            
            c_code_block += f"    if (service_idx == true_service_labels[{c_sample_idx_counter}]) correct_service++;\n"
            c_code_block += f"    if (activity_idx == true_activity_labels[{c_sample_idx_counter}]) correct_activity++;\n"
            
            c_sample_idx_counter += 1
            
        c_code_block += "\n    printf(\"C-Side Test Results:\\n\");\n" 
        c_code_block += "    printf(\"  Total C Test Samples: %d\\n\", num_c_test_samples);\n"
        c_code_block += "    printf(\"  Correct Service Predictions (C): %d\\n\", correct_service);\n"
        c_code_block += "    printf(\"  Service Accuracy (C): %.4f\\n\", num_c_test_samples > 0 ? (float)correct_service / num_c_test_samples : 0.0);\n"
        c_code_block += "    printf(\"  Correct Activity Predictions (C): %d\\n\", correct_activity);\n"
        c_code_block += "    printf(\"  Activity Accuracy (C): %.4f\\n\", num_c_test_samples > 0 ? (float)correct_activity / num_c_test_samples : 0.0);\n"
    else:
        print_status("Warning: C test set (c_test_set_df) is empty. Skipping C-side accuracy tests.")
        c_code_block += "    printf(\"\\nNo C-side accuracy tests run as test set was empty.\\n\");\n"

    return c_code_templates.C_MAIN_TEMPLATE.format(
        MAX_C_FEATURES=MAX_FEATURES_CONFIG, 
        c_test_cases_dynamic=c_code_block
    )

def generate_full_c_inference_code(c_test_set_df, service_model_c_name, activity_model_c_name, le_service, le_activity):
    c_includes = c_code_templates.C_INCLUDES_TEMPLATE.format(
        service_model_c_name=service_model_c_name,
        activity_model_c_name=activity_model_c_name,
        MAX_C_FEATURES_MAIN=MAX_FEATURES_CONFIG 
    )
    main_function_c = generate_c_main_with_test_samples(c_test_set_df, service_model_c_name, activity_model_c_name, le_service, le_activity)
    full_c_code = c_includes + main_function_c
    return full_c_code

def save_label_mappings(le_service: LabelEncoder, le_activity: LabelEncoder, file_path: str):
    print_status(f"Saving label mappings to: {file_path}")
    with open(file_path, "w", encoding='utf-8') as f:
        f.write("Service Class Mappings (Index: Label):\n")
        for index, label in enumerate(le_service.classes_):
            f.write(f"{index}: {label}\n")
        
        f.write("\nActivityType Class Mappings (Index: Label):\n")
        for index, label in enumerate(le_activity.classes_):
            f.write(f"{index}: {label}\n")
    print_status("Label mappings saved successfully.")

def main():
    create_directories()
    # Get train, evaluation, C example, and C test set dataframes
    train_df, eval_df, c_example_samples_df, c_test_set_df = load_and_preprocess_data()

    if train_df.empty:
        print_status("No training data to process after loading. Exiting.")
        return

    print_status("Performing feature engineering and label encoding...")
    # Fit vectorizer on training data only
    vectorizer = CountVectorizer(max_features=MAX_FEATURES_CONFIG, binary=True, lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    X_train = vectorizer.fit_transform(train_df['combined_headers_for_py'])
    joblib.dump(vectorizer, os.path.join(MODELS_SAVE_DIR, 'count_vectorizer.joblib'))
    print_status(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")

    # Transform evaluation data using the fitted vectorizer
    X_eval = vectorizer.transform(eval_df['combined_headers_for_py'])

    # Fit LabelEncoders on all unique labels in the original dataset to avoid missing labels in splits
    print_status("Fitting LabelEncoders on full original dataset for robustness...")
    all_original_dfs = []
    prediction_files_for_le = [f for f in os.listdir(INPUT_DATA_BASE_DIR) if f.endswith('_predictions.csv')]
    if not prediction_files_for_le:
        print_status(f"ERROR: No '_predictions.csv' files found in {INPUT_DATA_BASE_DIR} for fitting label encoders. Exiting.")
        sys.exit(1) 

    for file_name_le in prediction_files_for_le:
        try:
            df_temp_le = pd.read_csv(os.path.join(INPUT_DATA_BASE_DIR, file_name_le))
            all_original_dfs.append(df_temp_le)
        except Exception as e:
            print_status(f"Warning: Could not load or process {file_name_le} for label encoding. Error: {e}")

            pass 
    
    if not all_original_dfs:
        print_status("ERROR: No data could be loaded from any CSV files for label encoding. Exiting.")
        sys.exit(1)

    full_original_df = pd.concat(all_original_dfs, ignore_index=True)

    le_service = LabelEncoder()
    le_service.fit(full_original_df['service'].astype(str))
    train_df['service_encoded'] = le_service.transform(train_df['service'])
    eval_df['service_encoded'] = le_service.transform(eval_df['service'])
    # Ensure c_test_set_df also gets the transformation if its source (eval_df) hasn't been transformed yet for this column
    # Since c_test_set_df is a copy of eval_df.head(), transform eval_df first, then c_test_set_df will reflect it OR transform it directly.
    # For safety, transform c_test_set_df directly if its labels might not be covered by train_df fitting.
    # However, le_service is fit on full_original_df, so it knows all labels.
    c_test_set_df['service_encoded'] = le_service.transform(c_test_set_df['service'])
    joblib.dump(le_service, os.path.join(MODELS_SAVE_DIR, 'service_label_encoder.joblib'))

    le_activity = LabelEncoder()
    le_activity.fit(full_original_df['activityType'].astype(str))
    train_df['activityType_encoded'] = le_activity.transform(train_df['activityType'])
    eval_df['activityType_encoded'] = le_activity.transform(eval_df['activityType'])
    c_test_set_df['activityType_encoded'] = le_activity.transform(c_test_set_df['activityType'])
    joblib.dump(le_activity, os.path.join(MODELS_SAVE_DIR, 'activity_label_encoder.joblib'))
    print_status("Label encoders created and saved.")

    save_label_mappings(le_service, le_activity, LABEL_MAPPINGS_FILE_PATH)

    feature_eng_h_content = generate_feature_engineering_header_content(vectorizer)
    with open(FEATURE_ENGINEERING_H_PATH, 'w', encoding='utf-8') as f:
        f.write(feature_eng_h_content)
    print_status(f"Feature engineering H file generated: {FEATURE_ENGINEERING_H_PATH}")

    X_train_features_array = X_train.toarray().astype(np.int16)
    X_eval_features_array = X_eval.toarray().astype(np.int16)

    print_status("Training Service RandomForestClassifier...")
    rfc_service = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc_service.fit(X_train_features_array, train_df['service_encoded'])
    service_model_c_name = "service_model"
    cmodel_service = emlearn.convert(rfc_service, method=EMLEARN_MODEL_METHOD)
    service_h_path = os.path.join(EMLEARN_INCLUDE_DIR, f"{service_model_c_name}.h")
    cmodel_service.save(file=service_h_path, name=service_model_c_name)
    print_status(f"Service model C code saved to: {service_h_path}")

    print_status("Training Activity RandomForestClassifier...")
    rfc_activity = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc_activity.fit(X_train_features_array, train_df['activityType_encoded'])
    activity_model_c_name = "activity_model"
    cmodel_activity = emlearn.convert(rfc_activity, method=EMLEARN_MODEL_METHOD)
    activity_h_path = os.path.join(EMLEARN_INCLUDE_DIR, f"{activity_model_c_name}.h")
    cmodel_activity.save(file=activity_h_path, name=activity_model_c_name)
    print_status(f"Activity model C code saved to: {activity_h_path}")

    # Scikit-learn accuracy reporting
    print_status("\n---  Model Evaluation ---")
    service_preds_sklearn = rfc_service.predict(X_eval_features_array)
    activity_preds_sklearn = rfc_activity.predict(X_eval_features_array)
    
    service_accuracy_sklearn = accuracy_score(eval_df['service_encoded'], service_preds_sklearn)
    activity_accuracy_sklearn = accuracy_score(eval_df['activityType_encoded'], activity_preds_sklearn)
    
    print_status(f" Service Model Accuracy: {service_accuracy_sklearn:.4f}")
    print_status(f" Activity Model Accuracy: {activity_accuracy_sklearn:.4f}")

    print_status(f"\nGenerating main C inference file: {C_INFERENCE_FILE_PATH}")
    # Pass c_test_set_df for C testing (it already has encoded labels)
    full_c_inference_content = generate_full_c_inference_code(
        c_test_set_df, 
        service_model_c_name,
        activity_model_c_name,
        le_service, 
        le_activity
    )
    with open(C_INFERENCE_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(full_c_inference_content)
    print_status("Main C inference file generated successfully.")
    
    print_status(f"\nScript finished. emlearn models in: {EMLEARN_INCLUDE_DIR}")
    print_status(f"Feature engineering H file: {FEATURE_ENGINEERING_H_PATH}")
    print_status(f"Main C inference file: {C_INFERENCE_FILE_PATH}")
    print_status(f"Human-readable label mappings: {LABEL_MAPPINGS_FILE_PATH}")
    print_status(f"To compile C code (from {EMLEARN_OUTPUT_DIR}):")
    print_status(f"  gcc rfc_em_inference.c -o rfc_em_inference -Iinclude -I/path/to/emlearn/include -lm")

if __name__ == "__main__":
    main()
