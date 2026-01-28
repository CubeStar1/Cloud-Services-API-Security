import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from datetime import datetime

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path configuration
PATHS = {
    'input_data': os.path.join(BASE_PATH, "data", "output", "codebert", "predictions"),  # Directory with prediction files
    'output_folder': os.path.join(BASE_PATH, "data", "output", "rfc"),  # RFC output directory
    'codegen_folder': os.path.join(BASE_PATH, "data", "output", "rfc", "codegen"),  # Generated C code
    'label_mappings': os.path.join(BASE_PATH, "data", "output", "rfc", "codegen", "label_mappings.txt")  # Label mappings
}

def print_status(message: str):
    """Print status message with immediate flush"""
    print(message, flush=True)

def create_directories():
    try:
        # Only create directories, not files
        directories = {
            'input_data': PATHS['input_data'],
            'output_folder': PATHS['output_folder'],
            'codegen_folder': PATHS['codegen_folder']
        }
        for path in directories.values():
            os.makedirs(path, exist_ok=True)
        print_status("Successfully created directories")
    except Exception as e:
        print_status(f"Error creating directories: {e}")
        raise

create_directories()

def tree_to_c_code(trees, feature_names, label_encoders, vectorizer, test_data, file=None):
    """
    Generates C code for multiple decision trees from scikit-learn RandomForestClassifiers.
    
    Args:
        trees: Dictionary containing service and activity trees
        feature_names: List of feature names from TF-IDF
        label_encoders: Dictionary containing service and activity LabelEncoders
        vectorizer: TfidfVectorizer instance
        test_data: DataFrame containing test data
        file: File object to write the code to
    """
    def write_line(line, indent=0):
        indent_str = "    " * indent
        if file:
            file.write(f"{indent_str}{line}\n")
        else:
            print(f"{indent_str}{line}")

    def generate_tree_function(tree, function_name, label_encoder, depth=1):
        def recurse(node, depth):
            if tree.feature[node] != -2:  # Not a leaf node
                feature_idx = tree.feature[node]
                threshold = tree.threshold[node]
                write_line(f"if (features[{feature_idx}] <= {threshold:.6f}) {{", depth)
                recurse(tree.children_left[node], depth + 1)
                write_line("} else {", depth)
                recurse(tree.children_right[node], depth + 1)
                write_line("}", depth)
            else:  # Leaf node
                class_id = np.argmax(tree.value[node])
                write_line(f"return {class_id};  // {label_encoder.inverse_transform([class_id])[0]}", depth)

        write_line(f"int {function_name}(float features[]) {{")
        recurse(0, 1)
        write_line("}")
        write_line("")

    # Write header
    write_line("#include <stdio.h>")
    write_line("#include <string.h>")
    write_line("#include <ctype.h>")  # Needed for tolower() and isalnum()
    write_line("")

    # Add hash table implementation
    write_line("// Hash table size (power of 2 for efficient modulo)")
    write_line("#define HASH_TABLE_SIZE 8192")
    write_line(f"#define MAX_FEATURES {len(feature_names)}")
    write_line("")

    # Static feature entry structure
    write_line("// Feature entry structure")
    write_line("typedef struct {")
    write_line("    const char* term;", 1)
    write_line("    int feature_index;", 1)
    write_line("} FeatureEntry;")
    write_line("")

    # Create static feature table
    write_line("// Static feature table")
    write_line("static const FeatureEntry FEATURE_TABLE[] = {")
    vocabulary = vectorizer.vocabulary_
    for term, idx in vocabulary.items():
        write_line(f'    {{ "{term}", {idx} }},', 1)
    write_line("};")
    write_line("")

    write_line(f"#define NUM_FEATURES {len(vocabulary)}")
    write_line("")

    # Add optimized hash function
    write_line("// FNV-1a hash function")
    write_line("static inline unsigned int hash_string(const char* str) {")
    write_line("    unsigned int hash = 2166136261u;", 1)
    write_line("    while (*str) {", 1)
    write_line("        hash ^= (unsigned char)*str;", 2)
    write_line("        hash *= 16777619;", 2)
    write_line("        str++;", 2)
    write_line("    }", 1)
    write_line("    return hash % HASH_TABLE_SIZE;", 1)
    write_line("}")
    write_line("")

    # Static hash table
    write_line("// Static hash table buckets")
    write_line("typedef struct {")
    write_line("    int indices[10];  // Allow up to 10 entries per bucket", 1)
    write_line("    int count;", 1)
    write_line("} HashBucket;")
    write_line("")

    write_line("static const HashBucket HASH_BUCKETS[HASH_TABLE_SIZE] = {")
    
    # Pre-compute hash buckets
    buckets = [[] for _ in range(8192)]
    for i, (term, _) in enumerate(vocabulary.items()):
        hash_val = 2166136261
        for c in term.encode():
            hash_val ^= c
            hash_val *= 16777619
            hash_val &= 0xFFFFFFFF
        hash_val %= 8192
        buckets[hash_val].append(i)

    # Write pre-computed buckets
    for bucket in buckets:
        indices = bucket + [-1] * (10 - len(bucket))  # Pad with -1
        write_line(f"    {{ {{ {', '.join(map(str, indices))} }}, {len(bucket)} }},", 1)
    write_line("};")
    write_line("")

    # Add optimized feature lookup
    write_line("// Find feature index")
    write_line("static inline int find_feature(const char* term) {")
    write_line("    unsigned int hash = hash_string(term);", 1)
    write_line("    const HashBucket* bucket = &HASH_BUCKETS[hash];", 1)
    write_line("    for (int i = 0; i < bucket->count; i++) {", 1)
    write_line("        int idx = bucket->indices[i];", 2)
    write_line("        if (strcmp(FEATURE_TABLE[idx].term, term) == 0) {", 2)
    write_line("            return FEATURE_TABLE[idx].feature_index;", 3)
    write_line("        }", 2)
    write_line("    }", 1)
    write_line("    return -1;  // Term not found", 1)
    write_line("}")
    write_line("")

    # Generate feature extraction function
    write_line("void extract_features(const char* headers_host, const char* url, const char* method,")
    write_line("                     const char* headers_origin, const char* content_type,")
    write_line("                     const char* response_content_type, const char* referer,")
    write_line("                     const char* accept, float features[]) {")
    
    # Initialize features to 0
    write_line("    // Initialize all features to 0", 1)
    write_line(f"    for(int i = 0; i < {len(feature_names)}; i++) {{", 1)
    write_line("        features[i] = 0.0f;", 2)
    write_line("    }", 1)
    write_line("", 1)

    # Process input fields
    write_line("    // Process each input field", 1)
    write_line("    const char* inputs[] = {", 1)
    write_line("        headers_host ? headers_host : \"\",", 2)
    write_line("        url ? url : \"\",", 2)
    write_line("        method ? method : \"\",", 2)
    write_line("        headers_origin ? headers_origin : \"\",", 2)
    write_line("        content_type ? content_type : \"\",", 2)
    write_line("        response_content_type ? response_content_type : \"\",", 2)
    write_line("        referer ? referer : \"\",", 2)
    write_line("        accept ? accept : \"\"", 2)
    write_line("    };", 1)
    write_line("", 1)

    write_line("    char token_buffer[1024]; // Buffer for current token", 1)
    write_line("    int token_len;", 1)
    write_line("", 1)
    
    write_line("    // Process each input field, mimicking CountVectorizer tokenization", 1)
    write_line("    for (int i = 0; i < 8; i++) {", 1)
    write_line("        const char* p = inputs[i];", 2)
    write_line("        while (*p) {", 2)
    write_line("            // Skip non-alphanumeric characters", 3)
    write_line("            while (*p && !isalnum((unsigned char)*p)) {", 3)
    write_line("                p++;", 4)
    write_line("            }", 3)
    write_line("            if (!*p) break; // End of string", 3)
    write_line("", 3)
    write_line("            // Mark start of token and find its end", 3)
    write_line("            const char* token_start = p;", 3)
    write_line("            token_len = 0;", 3)
    write_line("            while (*p && isalnum((unsigned char)*p) && token_len < sizeof(token_buffer) - 1) {", 3)
    write_line("                token_buffer[token_len++] = tolower((unsigned char)*p);", 4)
    write_line("                p++;", 4)
    write_line("            }", 3)
    write_line("            token_buffer[token_len] = '\\0'; // Null-terminate the token", 3)
    write_line("", 3)
    write_line("            // Process token if length >= 2 (like \\w\\w+)", 3)
    write_line("            if (token_len >= 2) {", 3)
    write_line("                int feature_idx = find_feature(token_buffer);", 4)
    write_line("                if (feature_idx >= 0) {", 4)
    write_line("                    features[feature_idx] = 1.0f;", 5)
    write_line("                }", 4)
    write_line("            }", 3)
    write_line("        }", 2)
    write_line("    }", 1)
    write_line("}")
    write_line("")

    # Generate tree functions
    for tree_idx, tree in enumerate(trees['service'].estimators_):
        generate_tree_function(tree.tree_, f"service_tree_{tree_idx}", label_encoders['service'])
    
    for tree_idx, tree in enumerate(trees['activity'].estimators_):
        generate_tree_function(tree.tree_, f"activity_tree_{tree_idx}", label_encoders['activity'])

    # Generate prediction functions that combine tree predictions
    write_line("int predict_service(float features[]) {")
    write_line(f"    int votes[{len(label_encoders['service'].classes_)}] = {{0}};  // Array size matches number of service classes", 1)
    for i in range(len(trees['service'].estimators_)):
        write_line(f"    votes[service_tree_{i}(features)]++;", 1)
    write_line("    int max_votes = 0, predicted_class = 0;", 1)
    write_line(f"    for(int i = 0; i < {len(label_encoders['service'].classes_)}; i++) {{", 1)
    write_line("        if(votes[i] > max_votes) {", 2)
    write_line("            max_votes = votes[i];", 3)
    write_line("            predicted_class = i;", 3)
    write_line("        }", 2)
    write_line("    }", 1)
    write_line("    return predicted_class;", 1)
    write_line("}")
    write_line("")

    write_line("int predict_activity(float features[]) {")
    write_line(f"    int votes[{len(label_encoders['activity'].classes_)}] = {{0}};  // Array size matches number of activity classes", 1)
    for i in range(len(trees['activity'].estimators_)):
        write_line(f"    votes[activity_tree_{i}(features)]++;", 1)
    write_line("    int max_votes = 0, predicted_class = 0;", 1)
    write_line(f"    for(int i = 0; i < {len(label_encoders['activity'].classes_)}; i++) {{", 1)
    write_line("        if(votes[i] > max_votes) {", 2)
    write_line("            max_votes = votes[i];", 3)
    write_line("            predicted_class = i;", 3)
    write_line("        }", 2)
    write_line("    }", 1)
    write_line("    return predicted_class;", 1)
    write_line("}")
    write_line("")

    # Function to process a batch of test samples
    write_line("void process_test_samples() {")
    write_line("    float features[5000];  // Feature array for predictions", 1)
    write_line("    int total_samples = 0, correct_service = 0, correct_activity = 0;", 1)
    write_line("    int predicted_service, predicted_activity;  // Declare prediction variables once", 1)
    write_line("", 1)

    # Write test data as C array initializers
    for idx, row in test_data.iterrows():
        write_line(f'    // Test sample {idx + 1}', 1)
        
        # Helper function to sanitize and escape strings
        def sanitize_string(s):
            if pd.isna(s):
                return ""
            # Escape backslashes and quotes first
            s = str(s).replace('\\', '\\\\').replace('"', '\\"')
            # Replace UTF-8 charset declaration with escaped version
            s = s.replace('charset=UTF-8', 'charset=\\\"UTF-8\\\"')
            # Handle other special characters if needed
            return s

        # Sanitize all input strings
        host = sanitize_string(row["headers_Host"])
        url = sanitize_string(row["url"])
        method = sanitize_string(row["method"])
        origin = sanitize_string(row.get("requestHeaders_Origin", ""))
        content_type = sanitize_string(row.get("requestHeaders_Content_Type", ""))
        response_type = sanitize_string(row.get("responseHeaders_Content_Type", ""))
        referer = sanitize_string(row.get("requestHeaders_Referer", ""))
        accept = sanitize_string(row.get("requestHeaders_Accept", ""))

        # Write sanitized strings
        write_line(f'    const char* host_{idx} = "{host}";', 1)
        write_line(f'    const char* url_{idx} = "{url}";', 1)
        write_line(f'    const char* method_{idx} = "{method}";', 1)
        write_line(f'    const char* origin_{idx} = "{origin}";', 1)
        write_line(f'    const char* content_type_{idx} = "{content_type}";', 1)
        write_line(f'    const char* response_type_{idx} = "{response_type}";', 1)
        write_line(f'    const char* referer_{idx} = "{referer}";', 1)
        write_line(f'    const char* accept_{idx} = "{accept}";', 1)
        write_line("", 1)
        
        write_line(f'    printf("Processing sample {idx + 1}...\\n");', 1)
        write_line(f'    extract_features(host_{idx}, url_{idx}, method_{idx}, origin_{idx},', 1)
        write_line(f'                    content_type_{idx}, response_type_{idx},', 1)
        write_line(f'                    referer_{idx}, accept_{idx}, features);', 1)
        write_line("", 1)
        
        write_line("    predicted_service = predict_service(features);", 1)
        write_line("    predicted_activity = predict_activity(features);", 1)
        
        # Add actual labels for comparison
        true_service = row['service_encoded']
        true_activity = row['activityType_encoded']
        
        write_line(f'    printf("  Predicted Service: %d (True: {true_service})\\n", predicted_service);', 1)
        write_line(f'    printf("  Predicted Activity: %d (True: {true_activity})\\n", predicted_activity);', 1)
        write_line("", 1)
        
        write_line(f"    if (predicted_service == {true_service}) correct_service++;", 1)
        write_line(f"    if (predicted_activity == {true_activity}) correct_activity++;", 1)
        write_line("    total_samples++;", 1)
        write_line("", 1)

    # Print final accuracy statistics
    write_line('    printf("\\nFinal Results:\\n");', 1)
    write_line('    printf("Total samples: %d\\n", total_samples);', 1)
    write_line('    printf("Service Accuracy: %.4f\\n", (float)correct_service / total_samples);', 1)
    write_line('    printf("Activity Accuracy: %.4f\\n", (float)correct_activity / total_samples);', 1)
    write_line("}")
    write_line("")

    # Generate a simple main function for testing individual samples
    write_line("int main() {")
    write_line('    printf("API Classifier Demo\\n\\n");', 1)
    write_line("    float features[5000] = {0};  // Feature array for predictions", 1)
    write_line("", 1)
    write_line("    // Example test case (Dropbox Upload Precheck API call)", 1)
    write_line('    const char* test_host = "https://www.dropbox.com/cmd/upload_precheck";', 1)
    write_line('    const char* test_url = "https://www.dropbox.com/cmd/upload_precheck";', 1)
    write_line('    const char* test_method = "POST";', 1)
    write_line('    const char* test_origin = "";', 1)
    write_line('    const char* test_content_type = "";', 1)
    write_line('    const char* test_response_type = "text/plain; charset=utf-8";', 1)
    write_line('    const char* test_referer = "";', 1)
    write_line('    const char* test_accept = "";', 1)
    write_line("", 1)
    write_line('    printf("Processing test sample...\\n");', 1)
    write_line("    extract_features(test_host, test_url, test_method, test_origin,", 1)
    write_line("                    test_content_type, test_response_type,", 1)
    write_line("                    test_referer, test_accept, features);", 1)
    write_line("", 1)
    write_line("    int predicted_service = predict_service(features);", 1)
    write_line("    int predicted_activity = predict_activity(features);", 1)
    write_line("", 1)
    write_line('    printf("Predictions for Dropbox Upload Precheck API call:\\n");', 1)
    write_line('    printf("  Service: %d\\n", predicted_service);', 1)
    write_line('    printf("  Activity: %d\\n", predicted_activity);', 1)
    write_line("", 1)
    write_line('    printf("\\nTo run full test set, uncomment process_test_samples() in main()\\n");', 1)
    write_line("    process_test_samples();  // Uncomment to run full test set", 1)
    write_line("", 1)
    write_line("    return 0;", 1)
    write_line("}")

# Load and preprocess training dataset
def load_and_preprocess_data():
    # Find all prediction files in the input directory
    prediction_files = [f for f in os.listdir(PATHS['input_data']) if f.endswith('_predictions.csv')]
    if not prediction_files:
        raise Exception(f"No prediction files found in {PATHS['input_data']}")
    
    # Combine all prediction files
    dfs = []
    for file in prediction_files:
        df = pd.read_csv(os.path.join(PATHS['input_data'], file))
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Process features using apply function
    def combine_features(row):
        features = [
            row['headers_Host'],
            row['url'],
            row['method'],
            row['requestHeaders_Origin'],
            row['requestHeaders_Content_Type'],
            row['responseHeaders_Content_Type'],
            row['requestHeaders_Referer'],
            row['requestHeaders_Accept']
        ]
        return ' '.join(str(f) if pd.notna(f) else '' for f in features)
    
    df['combined_headers'] = df.apply(combine_features, axis=1)

    # Use predicted labels from CodeBERT
    # df['service'] = df['predicted_service'].astype(str)
    # df['activityType'] = df['predicted_activity'].astype(str)

    return df

def main():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()

        # Split data into training and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Limit test set to 100 samples
        # if len(test_df) > 1000:
        #     test_df = test_df.head(1000)
        
        print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")

        # Create and fit label encoders
        le_service = LabelEncoder()
        le_activity = LabelEncoder()
        
        # Fit encoders on all data to ensure all classes are known
        le_service.fit(df['service'])
        le_activity.fit(df['activityType'])
        
        # Transform labels for both train and test sets
        train_df['service_encoded'] = le_service.transform(train_df['service'])
        train_df['activityType_encoded'] = le_activity.transform(train_df['activityType'])
        test_df['service_encoded'] = le_service.transform(test_df['service'])
        test_df['activityType_encoded'] = le_activity.transform(test_df['activityType'])

        # Create and fit vectorizer using Binary Count Features
        print("Using CountVectorizer (binary=True) to match C code feature extraction...")
        vectorizer = CountVectorizer(max_features=5000, binary=True) # Use binary features
        X_train = vectorizer.fit_transform(train_df['combined_headers'])
        X_test = vectorizer.transform(test_df['combined_headers'])

        # Train Random Forest models
        print("Training Random Forest models...")
        rf_service = RandomForestClassifier(n_estimators=5, random_state=42)
        rf_activity = RandomForestClassifier(n_estimators=5, random_state=42)

        rf_service.fit(X_train, train_df['service_encoded'])
        rf_activity.fit(X_train, train_df['activityType_encoded'])

        # Evaluate on test set
        print("\nEvaluating on test set...")
        service_predictions = rf_service.predict(X_test)
        activity_predictions = rf_activity.predict(X_test)
        
        service_accuracy = np.mean(service_predictions == test_df['service_encoded'])
        activity_accuracy = np.mean(activity_predictions == test_df['activityType_encoded'])
        
        print(f"Service Classification Accuracy: {service_accuracy:.4f}")
        print(f"Activity Classification Accuracy: {activity_accuracy:.4f}")

        # Generate C code and write to file
        output_file = os.path.join(PATHS['codegen_folder'], "api_classifier.c")
        try:
            with open(output_file, "w") as f:
                tree_to_c_code(
                    {'service': rf_service, 'activity': rf_activity},
                    vectorizer.get_feature_names_out(),
                    {'service': le_service, 'activity': le_activity},
                    vectorizer,
                    test_df,  # Pass test data to generate C code with test samples
                    file=f
                )
            print(f"\nC code has been written to {output_file}")
        except Exception as e:
            print(f"Error writing C code: {e}")
            raise

        # Save label mappings for reference
        try:
            with open(PATHS['label_mappings'], "w") as f:
                f.write("Service Class Mappings:\n")
                for i, label in enumerate(le_service.classes_):
                    f.write(f"{i}: {label}\n")
                f.write("\nActivity Class Mappings:\n")
                for i, label in enumerate(le_activity.classes_):
                    f.write(f"{i}: {label}\n")
            print(f"Label mappings saved to {PATHS['label_mappings']}")
        except Exception as e:
            print(f"Error writing label mappings: {e}")
            raise

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 