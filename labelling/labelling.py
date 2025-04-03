import pandas as pd
import os
from typing import Tuple, Optional, List, Dict
from google import generativeai
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import glob
import json
from datetime import datetime

# Base path configuration
BASE_PATH = os.path.dirname(os.path.dirname(__file__))

# File paths and directories
PATHS = {
    'data_folder': os.path.join(BASE_PATH, "data"),
    'logs_folder': os.path.join(BASE_PATH, "data", "logs", "csv"),  # Source CSV files
    'labelled_folder': os.path.join(BASE_PATH, "data", "labelled"),  # Output directory
    'metadata_file': os.path.join(BASE_PATH, "data", "labelled", "metadata.json"),
    'train_file': "train_set.xlsx",  # Changed to xlsx as per screenshot
    'test_file': "test_set.xlsx",    # Changed to xlsx as per screenshot
    'rows_per_file': 300
}

# Load environment variables
load_dotenv()

# Configure APIs
client = OpenAI()
generativeai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration settings
CONFIG = {
    'openai_model': 'gpt-4o-mini',
    'gemini_model': 'gemini-2.0-flash-thinking-exp',
    'batch_size': 10,
    'test_rows': 20,
    'use_openai': True,  # Set to True to use OpenAI instead of Gemini
    'rows_per_file': 300,  # Number of rows to sample from each file
    'test_split': 0.2,    # Fraction of data to use for testing
    'recursive_search': True  # Whether to search subdirectories for data files
}

# Define possible services and activities
SERVICES = [
    "Adobe", "ChatGPT", "Circle", "Canva", "ClickUp", "Firebase",
    "Netlify", "Quip", "UserGuilding", "Dropbox", "Evernote",
    "Google Docs", "GitHub", "GitLab", "Gmail", "Google Sheets",
    "GoTo", "Google Slides", "Heroku", "OneDrive", "Outlook",
    "SheetDB", "Slack", "Microsoft Teams", "Travis", "Vercel",
    "Webex", "Zendesk", "Zoom", "Unknown Service"
]

ACTIVITIES = [
    # Original activities
    "Login", "Upload", "Download", "Access", "Editing", "Deleting",
    "Sharing", "Creating", "Updating", "Syncing", "Navigation",
    "Authentication", "Attempt", "Request", "Timeout", "Export",
    "Import", "Comment", "Review", "Approve", "Reject", "Query",
    "Visualization", "Configuration", "Integration", "Deployment",
    "Rollback", "Scan", "Audit", "Permission Change", "Password Reset",
    "Account Creation", "API Call", "Logout", "Build",
    "Email Sending", "Email Receiving", "Attachment Upload",
    "Attachment Download", "Message", "Call", "Meeting",
    "Guide Viewing", "Guide Completion", "Data Sync",
    "Configuration Update", "Health Check", "Unknown Activity"
]

def save_metadata(metadata: Dict) -> None:
    """Save metadata about the processed files."""
    with open(PATHS['metadata_file'], 'w') as f:
        json.dump(metadata, f, indent=4)

def load_metadata() -> Dict:
    """Load metadata about previously processed files."""
    if os.path.exists(PATHS['metadata_file']):
        with open(PATHS['metadata_file'], 'r') as f:
            return json.load(f)
    return {'processed_files': {}}

def find_csv_files(data_folder: str, recursive: bool = True) -> List[str]:
    """
    Find all CSV files in the logs folder.
    
    Args:
        data_folder: Root folder to search
        recursive: Whether to search subdirectories
    """
    return glob.glob(os.path.join(PATHS['logs_folder'], "*.csv"))

def create_prompt(row: pd.Series) -> str:
    """Create a prompt for the LLM based on the row data."""
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
    """Get classification using OpenAI API."""
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
        
        # Parse the response
        service = result.split("Service:")[1].split("Activity:")[0].strip()
        activity = result.split("Activity:")[1].strip()
        
        return service, activity
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Unknown Service", "Unknown Activity"

def get_gemini_classification(prompt: str) -> Tuple[str, str]:
    """Get classification using Gemini API."""
    try:
        model = generativeai.GenerativeModel(CONFIG['gemini_model'])
        response = model.generate_content(prompt)
        result = response.text
        
        # Parse the response
        service = result.split("Service:")[1].split("Activity:")[0].strip()
        activity = result.split("Activity:")[1].strip()
        
        return service, activity
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Unknown Service", "Unknown Activity"

def label_dataset(csv_path: str, use_openai: bool = True) -> pd.DataFrame:
    """
    Label the dataset using the specified LLM API.
    
    Args:
        csv_path: Path to the input CSV file
        use_openai: If True, use OpenAI API; if False, use Gemini API
    """
    # Read the dataset
    df = pd.read_csv(csv_path)
    
    # Initialize new columns if they don't exist
    if 'predicted_service' not in df.columns:
        df['predicted_service'] = None
    if 'predicted_activity' not in df.columns:
        df['predicted_activity'] = None
    
    # Get classification function based on selected API
    classify_func = get_openai_classification if use_openai else get_gemini_classification
    
    # Process rows that haven't been labeled yet
    for idx, row in df.iterrows():
        if pd.isna(row['predicted_service']) or pd.isna(row['predicted_activity']):
            prompt = create_prompt(row)
            service, activity = classify_func(prompt)
            
            df.at[idx, 'predicted_service'] = service
            df.at[idx, 'predicted_activity'] = activity
            
            print(f"Processed row {idx}: Service={service}, Activity={activity}")
            
            # Save progress after each batch
            if idx % CONFIG['batch_size'] == 0:
                df.to_csv(csv_path, index=False)
                print(f"Progress saved at row {idx}")
    
    # Save final results
    df.to_csv(csv_path, index=False)
    return df

def combine_datasets(data_folder: str, rows_per_file: int = 300) -> pd.DataFrame:
    """
    Combines data from all CSV files in the logs folder.
    
    Args:
        data_folder: Path to folder containing CSV files
        rows_per_file: Number of rows to sample from each file
    """
    all_data = []
    metadata = load_metadata()
    current_time = datetime.now().isoformat()
    
    # Get all CSV files from logs directory
    csv_files = find_csv_files(PATHS['logs_folder'])
    
    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Sample rows if the dataset is larger than rows_per_file
            if len(df) > rows_per_file:
                df = df.sample(n=rows_per_file, random_state=42)
            
            # Add source file information
            source_file = os.path.basename(file_path)
            df['source_file'] = source_file
            
            all_data.append(df)
            
            # Update metadata
            metadata['processed_files'][source_file] = {
                'last_processed': current_time,
                'rows_sampled': len(df),
                'total_rows': len(pd.read_csv(file_path))
            }
            
            print(f"Processed {source_file}: {len(df)} rows sampled from {metadata['processed_files'][source_file]['total_rows']} total rows")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save updated metadata
    save_metadata(metadata)
    
    if not all_data:
        raise ValueError(f"No CSV files found in {PATHS['logs_folder']}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(PATHS['labelled_folder'], exist_ok=True)
    
    # Combine datasets from logs folder
    print(f"Combining datasets from {PATHS['logs_folder']}...")
    combined_df = combine_datasets(PATHS['logs_folder'], rows_per_file=CONFIG['rows_per_file'])
    
    # Create train-test split
    train_df, test_df = train_test_split(
        combined_df, 
        test_size=CONFIG['test_split'], 
        random_state=42
    )
    
    # Save train and test sets as Excel files
    train_path = os.path.join(PATHS['labelled_folder'], PATHS['train_file'])
    test_path = os.path.join(PATHS['labelled_folder'], PATHS['test_file'])
    
    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)
    
    print(f"\nDataset split complete:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Training set: {len(train_df)} samples saved to {train_path}")
    print(f"Test set: {len(test_df)} samples saved to {test_path}")
    
    # Process with LLM
    print(f"\nStarting classification using {'OpenAI' if CONFIG['use_openai'] else 'Gemini'} API...")
    
    # Process training set
    print("\nProcessing training set...")
    train_df = label_dataset(train_path, use_openai=CONFIG['use_openai'])
    
    # Process test set
    print("\nProcessing test set...")
    test_df = label_dataset(test_path, use_openai=CONFIG['use_openai'])
    
    # Print results summary
    print("\nResults Summary:")
    print("\nTraining Set:")
    print("Services found:", train_df['predicted_service'].value_counts())
    print("\nActivities found:", train_df['predicted_activity'].value_counts())
    
    print("\nTest Set:")
    print("Services found:", test_df['predicted_service'].value_counts())
    print("\nActivities found:", test_df['predicted_activity'].value_counts())
