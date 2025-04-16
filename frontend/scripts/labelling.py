import pandas as pd
import os
from typing import Tuple, Optional, List, Dict
from google import generativeai
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import glob
import json
from datetime import datetime
import sys

def print_status(message: str):
    """Print status message and flush immediately for real-time monitoring."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        # Fallback for encoding issues
        print(message.encode('ascii', 'ignore').decode('ascii'), flush=True)

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(__file__))

# File paths and directories
PATHS = {
    'data_folder': os.path.join(BASE_PATH, "data"),
    'logs_folder': os.path.join(BASE_PATH, "data", "logs", "csv"),  # Source CSV files
    'labelled_folder': os.path.join(BASE_PATH, "data", "labelled"),  # Output directory
    'metadata_file': os.path.join(BASE_PATH, "data", "labelled", "metadata.json"),
    'train_file': "train_set.csv",
    'test_file': "test_set.csv",
    'rows_per_file': 1000
}

# Load environment variables
load_dotenv()

print_status("[*] Initializing labelling process...")

# Configure APIs
client = OpenAI()
groq_client = Groq()
generativeai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration settings
CONFIG = {
    'openai_model': 'gpt-4o-mini',
    'gemini_model': 'gemini-2.0-flash-thinking-exp',
    'groq_model': 'llama3-70b-8192',
    'batch_size': 10,
    'test_rows': 20,
    'use_openai': False,
    'use_groq': True,
    'rows_per_file': 1000,
    'test_split': 0.2,
    'recursive_search': True
}

SERVICES = [
    "Bugzilla", "Unknown Service", "Webcompat"
]

ACTIVITIES = [
    "Login", "Upload", "Download", "Logout", "Unknown Activity",
    "New Bug"
]

def save_metadata(metadata: Dict) -> None:
    with open(PATHS['metadata_file'], 'w') as f:
        json.dump(metadata, f, indent=4)
    print_status("[+] Updated metadata file")

def load_metadata() -> Dict:
    if os.path.exists(PATHS['metadata_file']):
        with open(PATHS['metadata_file'], 'r') as f:
            return json.load(f)
    return {'processed_files': {}}

def find_csv_files(data_folder: str, recursive: bool = True) -> List[str]:
    print_status("[*] Searching for CSV files...")
    files = glob.glob(os.path.join(PATHS['logs_folder'], "*.csv"))
    print_status(f"[+] Found {len(files)} CSV files")
    return files

def create_prompt(row: pd.Series) -> str:
    return f"""Based on the following HTTP request data, classify the service being accessed and the activity being performed.
    
    Host: {row['headers_Host']}
    Method: {row['method']}
    URL: {row['url']}
    Content-Type: {row['requestHeaders_Content_Type']}
    Accept: {row['requestHeaders_Accept']}
    Origin: {row.get('requestHeaders_Origin', 'N/A')}
    Referer: {row.get('requestHeaders_Referer', 'N/A')}
    
    Consider these aspects for classification:
    - Authentication related URLs contain: auth, signin, login, sso, token
    - Upload activities use PUT or POST methods
    - Download activities typically use GET method
    - Message activities contain: message, chat
    - Meeting activities contain: meeting, schedule
    
    Available Services: {', '.join(SERVICES)}
    Available Activities: {', '.join(ACTIVITIES)}
    
    Respond in the following format only:
    Service: <service_name>
    Activity: <activity_name>
    """

def get_openai_classification(prompt: str) -> Tuple[str, str]:
    try:
        completion = client.chat.completions.create(
            model=CONFIG['openai_model'],
            messages=[
                {"role": "system", "content": "You are a classifier that categorizes HTTP requests into services and activities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result = completion.choices[0].message.content
        service = result.split("Service:")[1].split("Activity:")[0].strip()
        activity = result.split("Activity:")[1].strip()
        return service, activity
    except Exception as e:
        print_status(f"[!] OpenAI API error: {str(e)}")
        return "Unknown Service", "Unknown Activity"

def get_gemini_classification(prompt: str) -> Tuple[str, str]:
    try:
        model = generativeai.GenerativeModel(CONFIG['gemini_model'])
        response = model.generate_content(prompt)
        result = response.text
        service = result.split("Service:")[1].split("Activity:")[0].strip()
        activity = result.split("Activity:")[1].strip()
        return service, activity
    except Exception as e:
        print_status(f"[!] Gemini API error: {str(e)}")
        return "Unknown Service", "Unknown Activity"

def get_groq_classification(prompt: str) -> Tuple[str, str]:
    try:
        completion = groq_client.chat.completions.create(
            model=CONFIG['groq_model'],
            messages=[
                {"role": "system", "content": "You are a classifier that categorizes HTTP requests into services and activities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result = completion.choices[0].message.content
        service = result.split("Service:")[1].split("Activity:")[0].strip()
        activity = result.split("Activity:")[1].strip()
        return service, activity
    except Exception as e:
        print_status(f"[!] Groq API error: {str(e)}")
        return "Unknown Service", "Unknown Activity"

def label_dataset(csv_path: str, use_openai: bool = True, use_groq: bool = False) -> pd.DataFrame:
    print_status(f"[*] Processing file: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    
    if 'predicted_service' not in df.columns:
        df['predicted_service'] = None
    if 'predicted_activity' not in df.columns:
        df['predicted_activity'] = None
    
    api_name = "OpenAI" if use_openai else "Groq" if use_groq else "Gemini"
    print_status(f"[*] Using {api_name} API for classification")
    classify_func = get_openai_classification if use_openai else get_groq_classification if use_groq else get_gemini_classification
    
    total_rows = len(df)
    for idx, row in df.iterrows():
        if pd.isna(row['predicted_service']) or pd.isna(row['predicted_activity']):
            prompt = create_prompt(row)
            service, activity = classify_func(prompt)
            
            df.at[idx, 'predicted_service'] = service
            df.at[idx, 'predicted_activity'] = activity
            
            progress = (idx + 1) / total_rows * 100
            print_status(f"[*] Progress: {progress:.1f}% - Row {idx + 1}/{total_rows}")
            
            if idx % CONFIG['batch_size'] == 0:
                df.to_csv(csv_path, index=False)
                print_status(f"[+] Progress saved at row {idx + 1}")
    
    df.to_csv(csv_path, index=False)
    print_status(f"[+] Completed processing {os.path.basename(csv_path)}")
    return df

def combine_datasets(data_folder: str, rows_per_file: int = 300) -> pd.DataFrame:
    print_status("[*] Combining datasets...")
    all_data = []
    metadata = load_metadata()
    current_time = datetime.now().isoformat()
    
    csv_files = find_csv_files(PATHS['logs_folder'])
    
    for file_path in csv_files:
        try:
            print_status(f"[*] Reading file: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            
            if len(df) > rows_per_file:
                print_status(f"[*] Sampling {rows_per_file} rows from {len(df)} total rows")
                df = df.sample(n=rows_per_file, random_state=42)
            
            source_file = os.path.basename(file_path)
            df['source_file'] = source_file
            all_data.append(df)
            
            metadata['processed_files'][source_file] = {
                'last_processed': current_time,
                'rows_sampled': len(df),
                'total_rows': len(pd.read_csv(file_path))
            }
            
        except Exception as e:
            print_status(f"[!] Error processing {file_path}: {str(e)}")
    
    save_metadata(metadata)
    
    if not all_data:
        print_status("[!] No CSV files found!")
        raise ValueError(f"No CSV files found in {PATHS['logs_folder']}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print_status(f"[+] Combined {len(all_data)} files with total {len(combined_df)} rows")
    return combined_df

if __name__ == "__main__":
    try:
        print_status("[*] Starting labelling process...")
        
        # Create necessary directories
        os.makedirs(PATHS['labelled_folder'], exist_ok=True)
        print_status("[+] Created output directories")
        
        # Combine datasets
        print_status("[*] Combining datasets...")
        combined_df = combine_datasets(PATHS['logs_folder'], rows_per_file=CONFIG['rows_per_file'])
        
        # Create train-test split
        print_status("[*] Creating train-test split...")
        train_df, test_df = train_test_split(
            combined_df, 
            test_size=CONFIG['test_split'], 
            random_state=42
        )
        
        # Save datasets
        train_path = os.path.join(PATHS['labelled_folder'], PATHS['train_file'])
        test_path = os.path.join(PATHS['labelled_folder'], PATHS['test_file'])
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print_status(f"""
[*] Dataset split complete:
    - Total samples: {len(combined_df)}
    - Training set: {len(train_df)} samples
    - Test set: {len(test_df)} samples
        """)
        
        # Process with LLM
        api_name = "OpenAI" if CONFIG['use_openai'] else "Groq" if CONFIG['use_groq'] else "Gemini"
        print_status(f"[*] Starting classification using {api_name} API")
        
        # Process training set
        print_status("[*] Processing training set...")
        train_df = label_dataset(train_path, use_openai=CONFIG['use_openai'], use_groq=CONFIG['use_groq'])
        
        # Process test set
        print_status("[*] Processing test set...")
        test_df = label_dataset(test_path, use_openai=CONFIG['use_openai'], use_groq=CONFIG['use_groq'])
        
        # Print results summary
        print_status("""
[*] Results Summary:
        """)
        print_status("[*] Training Set:")
        print_status(f"    Services found: {train_df['predicted_service'].value_counts().to_dict()}")
        print_status(f"    Activities found: {train_df['predicted_activity'].value_counts().to_dict()}")
        
        print_status("[*] Test Set:")
        print_status(f"    Services found: {test_df['predicted_service'].value_counts().to_dict()}")
        print_status(f"    Activities found: {test_df['predicted_activity'].value_counts().to_dict()}")
        
        print_status("[+] Labelling process completed successfully!")
        
    except Exception as e:
        print_status(f"[!] Error during labelling process: {str(e)}")
        sys.exit(1)
