import pandas as pd
import os
from typing import Tuple, Optional, List, Dict
from groq import Groq
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import glob
import json
from datetime import datetime
import sys
import time

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

# Configuration settings
CONFIG = {
    'groq_model': 'llama-3.1-8b-instant',
    'batch_size': 10,
    'test_rows': 20,
    'rows_per_file': 1000,
    'test_split': 0.2,
    'recursive_search': True
}

SERVICES = [
    "FB Marketplace", "Unknown"
]

ACTIVITIES = [
    "Login", "Upload", "Download", "Logout", "Unknown", "Search", "API Call", "Message", "Payment"
]

def save_metadata(metadata: Dict) -> None:
    with open(PATHS['metadata_file'], 'w') as f:
        json.dump(metadata, f, indent=4)
    print_status("[+] Updated metadata file")

def load_metadata() -> Dict:
    if os.path.exists(PATHS['metadata_file']):
        with open(PATHS['metadata_file'], 'r') as f:
            return json.load(f)
    return {
        'processed_files': {},
        'labelling_progress': {
            PATHS['train_file']: {'last_row': 0},
            PATHS['test_file']: {'last_row': 0}
        }
    }

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

def get_groq_client():
    """Get a fresh Groq client with the current API key"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)

def get_groq_classification(prompt: str) -> Tuple[str, str]:
    try:
        groq_client = get_groq_client()
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
        raise Exception(f"Groq API error: {str(e)}")

def update_groq_api_key():
    """Prompt user to update the Groq API key"""
    print_status("\n[!] Hit rate limit or API error with Groq. You need to provide a new API key.")
    print_status("[*] Options:")
    print_status("    1. Enter a new API key")
    print_status("    2. Quit and resume later")
    
    choice = input("Choose an option (1-2): ").strip()
    
    if choice == '1':
        new_api_key = input("Enter new Groq API key: ").strip()
        if new_api_key:
            os.environ["GROQ_API_KEY"] = new_api_key
            print_status("[+] API key updated. Resuming in 5 seconds...")
            time.sleep(5)
            return True
        else:
            print_status("[!] No API key provided. Exiting...")
            return False
    else:
        print_status("[+] Exiting. You can resume later by running the script again.")
        return False

def label_dataset(csv_path: str) -> pd.DataFrame:
    """Label dataset with the ability to resume from the last processed row"""
    file_name = os.path.basename(csv_path)
    print_status(f"[*] Processing file: {file_name}")
    
    # Load metadata to get the last processed row
    metadata = load_metadata()
    
    # Initialize or get labelling progress for this file
    if file_name not in metadata['labelling_progress']:
        metadata['labelling_progress'][file_name] = {'last_row': 0}
    
    last_row = metadata['labelling_progress'][file_name]['last_row']
    print_status(f"[*] Resuming from row {last_row}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize columns if they don't exist
    if 'predicted_service' not in df.columns:
        df['predicted_service'] = None
    if 'predicted_activity' not in df.columns:
        df['predicted_activity'] = None
    
    total_rows = len(df)
    
    try:
        # Process rows starting from the last_row
        for idx in range(last_row, total_rows):
            row = df.iloc[idx]
            
            if pd.isna(row['predicted_service']) or pd.isna(row['predicted_activity']):
                try:
                    prompt = create_prompt(row)
                    service, activity = get_groq_classification(prompt)
                    
                    df.at[idx, 'predicted_service'] = service
                    df.at[idx, 'predicted_activity'] = activity
                    
                    # Update progress in metadata
                    metadata['labelling_progress'][file_name]['last_row'] = idx + 1
                    
                    progress = (idx + 1) / total_rows * 100
                    print_status(f"[*] Progress: {progress:.1f}% - Row {idx + 1}/{total_rows}")
                    
                    # Save progress periodically
                    if (idx + 1) % CONFIG['batch_size'] == 0:
                        df.to_csv(csv_path, index=False)
                        save_metadata(metadata)
                        print_status(f"[+] Progress saved at row {idx + 1}")
                
                except Exception as e:
                    # Save current progress before handling error
                    df.to_csv(csv_path, index=False)
                    save_metadata(metadata)
                    print_status(f"[!] Error at row {idx + 1}: {str(e)}")
                    
                    # Try to update API key and continue
                    if not update_groq_api_key():
                        # User chose to exit
                        return df
                    
    except KeyboardInterrupt:
        print_status("\n[!] Process interrupted by user")
        # Save progress before exiting
        df.to_csv(csv_path, index=False)
        save_metadata(metadata)
        print_status(f"[+] Progress saved at row {metadata['labelling_progress'][file_name]['last_row']}")
    
    # Final save
    df.to_csv(csv_path, index=False)
    print_status(f"[+] Completed processing {file_name}")
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

def should_regenerate_datasets() -> bool:
    """Determine if datasets should be regenerated or use existing ones"""
    train_path = os.path.join(PATHS['labelled_folder'], PATHS['train_file'])
    test_path = os.path.join(PATHS['labelled_folder'], PATHS['test_file'])
    
    # Check if both files exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        print_status("[*] Existing datasets found.")
        response = input("Do you want to use existing datasets? (y/n): ").strip().lower()
        return response != 'y'
    
    return True

if __name__ == "__main__":
    try:
        print_status("[*] Starting labelling process with Groq API...")
        
        # Check if Groq API key is set
        if not os.getenv("GROQ_API_KEY"):
            api_key = input("Enter your Groq API key: ").strip()
            if not api_key:
                print_status("[!] No API key provided. Exiting...")
                sys.exit(1)
            os.environ["GROQ_API_KEY"] = api_key
        
        # Create necessary directories
        os.makedirs(PATHS['labelled_folder'], exist_ok=True)
        print_status("[+] Created output directories")
        
        # Determine if we should regenerate datasets
        regenerate = should_regenerate_datasets()
        
        if regenerate:
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
            
            # Reset labelling progress in metadata
            metadata = load_metadata()
            metadata['labelling_progress'] = {
                PATHS['train_file']: {'last_row': 0},
                PATHS['test_file']: {'last_row': 0}
            }
            save_metadata(metadata)
        else:
            print_status("[*] Using existing datasets")
        
        train_path = os.path.join(PATHS['labelled_folder'], PATHS['train_file'])
        test_path = os.path.join(PATHS['labelled_folder'], PATHS['test_file'])
        
        # Process training set
        print_status("[*] Processing training set...")
        train_df = label_dataset(train_path)
        
        # Process test set
        print_status("[*] Processing test set...")
        test_df = label_dataset(test_path)
        
        # Print results summary
        print_status("""
[*] Results Summary:
        """)
        
        train_df = pd.read_csv(train_path)
        print_status("[*] Training Set:")
        print_status(f"    Services found: {train_df['predicted_service'].value_counts().to_dict()}")
        print_status(f"    Activities found: {train_df['predicted_activity'].value_counts().to_dict()}")
        
        test_df = pd.read_csv(test_path)
        print_status("[*] Test Set:")
        print_status(f"    Services found: {test_df['predicted_service'].value_counts().to_dict()}")
        print_status(f"    Activities found: {test_df['predicted_activity'].value_counts().to_dict()}")
        
        print_status("[+] Labelling process completed successfully!")
        
    except Exception as e:
        print_status(f"[!] Error during labelling process: {str(e)}")
        sys.exit(1)