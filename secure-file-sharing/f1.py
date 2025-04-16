import os
import csv
import requests
import json
import datetime
import ssl
import certifi
import hashlib
import re
import numpy as np
import smtplib
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import logging
import logging.handlers
import time
import sys
import slack_notify
import dropbox  # New import for Dropbox

# Configure the logger
logger = logging.getLogger('SaaS_Monitoring')
logger.setLevel(logging.INFO)

# Configure the SysLogHandler
syslog_handler = logging.handlers.SysLogHandler(address=('127.0.0.1', 1514))
formatter = logging.Formatter('%(asctime)s %(name)s: %(message)s', datefmt='%b %d %H:%M:%S')
syslog_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(syslog_handler)

# **🔹 Load Embedding Model**
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# **🔹 SSL Fix**
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Configure proxy if needed - comment out if proxy is not needed
# os.environ['HTTP_PROXY'] = 'http://localhost:8001'  # Update with your actual proxy address
# os.environ['HTTPS_PROXY'] = 'http://localhost:8001'  # Use HTTP protocol for HTTPS requests


# **🔹 Configuration**
# Google Drive Configuration
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "token2.json"  # Ensure this is properly configured
CREDENTIALS_FILE = "credentials2.json"  # Replace with your actual credentials file

# Dropbox Configuration
DROPBOX_ACCESS_TOKEN = "sl.u.AFoGQUyiXqVQNB4PqvhDDKjha83Go1XMZck1P-0v6cY1p9L-02eHjuRZTe5PZdTF-1oo4oTt-bEc6Sk2CFPQ5tukAjdRD9ezPDi0mPd_nhtNBp-buBDwEdL-YtqmMivdRXOtgyTg4NFlbL2xNnYqUD-wXZU50egaemXunxAkMCWOASdKdMdla3KpMODoGj-9WAXzJ7sTYMYczOg0eyuhsHbFlsvJMh3BVXuxi9x23kCgeTCVDabMsnIzjy0SuZsbEd89bngVmuXhJH-uZzdJhzlqCbvX-X_JlgMvNR35SFT_hiIMCrmYv_HxRiyxeqjJ1nxVQ-ujYGNCRa7OEtr9wMLliwmheeDZ3sECCtkVVR9yLqTlaSYM65G-BhL6epKFY3f2uoZNPqGXRJUsp8HRRywvLe7rKaafyak0ZzSOluYfzpForyuS4s3BgX1aqT4QufcWRifZUKR8i3pD4zmhNcPDzgh9UIJ27Lc8wNQWjRoI-4G24HhN-5-mOAhLCJMOBQ9WlYj1snm2BV5IfwoBcgbEuzI-yro63cWyKHfUvGQgCrzvtfNwI2HCXNDA8H4Gn8PFFIi0thLDVEC6q6xvB6dUyaLJ3knz25Gf-8dkSQcm5dnFgzmshz1lRk2vgq2NC3RI9yy0BQ3k5IYxKeE1Fgp8y2n95QoLpDQSjMDj5Ep4KUC1QcZ3OvlZf178UvrlphNJj1-bzieYMpemcdrxuRrprY9OwKftrXQJmkQYCWXRydgsiJOlQBCfigCtxO8S79emcGIppvPsdqTd4MLK8TLTxnDWi-sZWeEjz90s2vbELyWD9mTIIlp4pasYgDxhUPOeYnEVPYZMjdKQQYzAjQhOxDBwBv5f3sgdYErABhtJTBrtxSIWO4LuPoSVhfHPrPY-NVW7YqHKK8pmNwTtm5OpJF9j9YMbiiY7_hMad1Fvyr1ddTRCT7vdmNKb_YHceFRCGMhF8XpCjyPSkywLkrkuvNQcfqSfzS4CXOOJ74giS6t762-kedfiE0rJoBvaNKXi0YCjtjgRAzpGI9-ThND2mOdoUl4EFWrh1rlj_mkYwPhhN4S71p11CCWQYxL938FrspcbYc5xUyTQxvt68se3mZi3GEeevMhwTLPG5tBCUWprkywlvOwOgz0hPO4svTT9sqGY0heaqIdx4B3pZ_TvH5fU6zAkzrz9GpYNn3VOsbCcpCeHOCPkIoZsTncaICR7fYkIMVDGhDFnrLS8ukq_d8NaLuYO-t7egUyVY1PpjyEbq_aX9eUFgTzcY09dDMgLRIuEl7y9onTzd9nJwdh943M_vDOW9ufT71POyE-E7fmUkhs9NnqmUm2SHAPF23eLZi_01hh90g0RBZpEGOf4Em89ueispHSogYdXraOqB7E4A5M_tuXr8q2-7NVSg-qLMtMbNAGhcYsRk4d9U_cG"  # Replace with the new token that has sharing.read scope

# Other Configuration
PINECONE_API_KEY = "pcsk_3xs3j4_RqkwrJt6UHbym2YJM16TvT5yunfKtSbeJt3HnJZrcB1nwhJD9q9Gsv1t2ZoYe8k"  # 🔐 Replace with your actual Pinecone API key
GDRIVE_INDEX_NAME = "drive-metadata-index"
DROPBOX_INDEX_NAME = "cloud-metadata-index3"
LOG_FILE = "syslog_client.log"  # Log file for monitoring
ALERTS_FILE = "alerts_log.json"  # Store previously alerted changes
DROPBOX_ALERTS_FILE = "dropbox_alerts_log.json"  # Store Dropbox alerts

# Slack Config
SLACK_GDRIVE_CHANNEL = "#all-sfs"
SLACK_DROPBOX_CHANNEL = "#all-sfs"
SLACK_TOKEN = "xoxb-8649015195078-8653518511733-s7bOWL8qcH2xy6hDW5Iwe0VR"

# **🔹 Initialize Services**
# Initialize Pinecone for Google Drive
pc = Pinecone(api_key=PINECONE_API_KEY)
if GDRIVE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=GDRIVE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
gdrive_index = pc.Index(GDRIVE_INDEX_NAME)

# Initialize Pinecone for Dropbox
if DROPBOX_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=DROPBOX_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
dropbox_index = pc.Index(DROPBOX_INDEX_NAME)

# Initialize Dropbox client with proper scopes
dbx = dropbox.Dropbox(
    DROPBOX_ACCESS_TOKEN,
    scope=['files.metadata.read', 'sharing.read', 'files.content.read']
)


def log_event(message):
    """Save logs to a file and print them."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - SaaS_Monitor - {message}"
    print(log_message)  # Print to console

    try:
        # Write to syslog via logger
        if isinstance(message, dict):
            # Ensure the message is properly formatted as JSON
            json_str = json.dumps(message)
            logger.info(json_str)
        else:
            # If it's a string, log as is
            logger.info(message)

        # Force logger to flush its handlers
        for handler in logger.handlers:
            handler.flush()

        # Direct write fallback - ensure we get something in the log file
        try:
            if isinstance(message, dict):
                json_str = json.dumps(message)
                with open("syslog.log", "a", encoding="utf-8") as f:
                    f.write(json_str + "\n")
            else:
                with open("syslog.log", "a", encoding="utf-8") as f:
                    f.write(f"{message}\n")
        except Exception as e:
            print(f"Error writing directly to syslog.log: {e}")
    except Exception as e:
        print(f"Error writing to syslog: {e}")


def load_alerts():
    """Load previously sent alerts to avoid duplicate notifications."""
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_alerts(alerts):
    """Save the updated alerts log."""
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=4)


def clean_vector(vector):
    """Ensure the vector is 384-dimensional and contains valid values."""
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)  # Replace invalid values
    if len(vector) != 384:
        print(f"Warning: Adjusting vector size from {len(vector)} to 384.")
        vector = np.pad(vector, (0, max(0, 384 - len(vector))), mode='constant')[:384]  # Trim or pad
    return vector.tolist()


def get_existing_metadata(file_id):
    """Fetch stored metadata & permissions from Pinecone."""
    try:
        fetched_vectors = gdrive_index.fetch(ids=[file_id])
        if file_id in fetched_vectors.vectors:
            metadata = fetched_vectors.vectors[file_id].metadata
            # Convert JSON strings back to objects
            for key, value in metadata.items():
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, (dict, list)):
                            metadata[key] = parsed
                    except:
                        pass  # Keep as string if not valid JSON
            return metadata
    except Exception as e:
        log_event(f"Error fetching metadata from Pinecone: {e}")
    return None


def process_drive_file(file, service):
    """Process a single drive file."""
    try:
        file_id = file.get('id')
        if not file_id:
            return

        metadata = {
            'id': file_id,
            'name': file.get('name', 'Unknown'),
            'mimeType': file.get('mimeType', 'Unknown'),
            'permissions': file.get('permissions', []),
            'trashed': file.get('trashed', False)
        }

        # If file is trashed, we don't need to process it further
        # as it will be handled by the deletion check in main()
        if metadata.get('trashed', False):
            return

        changes = detect_change(file_id, metadata, service)
        if changes:
            log_event({
                "file_id": file_id,
                "file_name": metadata.get("name", "Unknown"),
                "owner": next((p.get("emailAddress") for p in metadata.get("permissions", [])
                               if p.get("role") == "owner"), "Unknown"),
                "details": changes,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    except Exception as e:
        log_event(f"Error processing file {file.get('id', 'Unknown')}: {e}")


def calculate_metadata_hash(metadata):
    """Calculate a hash of the metadata to detect changes."""
    relevant_data = {
        "name": metadata.get("name"),
        "permissions": sorted([
            (p.get("emailAddress", ""), p.get("role", ""))
            for p in metadata.get("permissions", [])
        ])
    }
    return hashlib.md5(json.dumps(relevant_data, sort_keys=True).encode()).hexdigest()


def sanitize_metadata(metadata):
    """Convert complex metadata to simple types for Pinecone."""
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            sanitized[key] = value
        else:
            # Convert complex objects to JSON string
            sanitized[key] = json.dumps(value)
    return sanitized


def detect_change(file_id, new_metadata, service):
    """Detects metadata changes and sends alerts only for new changes."""
    try:
        existing_metadata = get_existing_metadata(file_id)
        if existing_metadata and not isinstance(existing_metadata, dict):
            log_event(f"Invalid metadata format for {file_id}")
            existing_metadata = None

        new_hash = calculate_metadata_hash(new_metadata)

        # Load alerts
        alerts = load_alerts()
        alert_key = f"{file_id}"

        # Check if this exact state was already processed
        if alert_key in alerts and alerts[alert_key] == new_hash:
            return None

        # Get the user who made the change from the Drive API
        try:
            # First try to get the last modifier from revision history
            revision = service.revisions().list(
                fileId=file_id,
                fields="revisions(lastModifyingUser)",
                pageSize=1
            ).execute()

            # Then get the file's change history
            history = service.files().get(
                fileId=file_id,
                fields="lastModifyingUser,sharingUser",
                supportsAllDrives=True
            ).execute()

            # For permission changes, use sharingUser if available
            if history.get('sharingUser'):
                modifier_info = {
                    "name": history['sharingUser'].get('displayName', 'Unknown'),
                    "email": history['sharingUser'].get('emailAddress', 'Unknown')
                }
            # For other changes, use lastModifyingUser
            elif history.get('lastModifyingUser'):
                modifier_info = {
                    "name": history['lastModifyingUser'].get('displayName', 'Unknown'),
                    "email": history['lastModifyingUser'].get('emailAddress', 'Unknown')
                }
            # Fallback to revision history
            elif revision.get('revisions') and revision['revisions'][-1].get('lastModifyingUser'):
                last_modifier = revision['revisions'][-1]['lastModifyingUser']
                modifier_info = {
                    "name": last_modifier.get('displayName', 'Unknown'),
                    "email": last_modifier.get('emailAddress', 'Unknown')
                }
            else:
                modifier_info = {
                    "name": "Unknown",
                    "email": "Unknown"
                }
        except Exception as e:
            log_event(f"Error getting last modifier: {e}")
            modifier_info = {
                "name": "Unknown",
                "email": "Unknown"
            }

        # Prepare change details for other changes
        if not existing_metadata:
            change_details = {
                "type": "new_file",
                "file_name": new_metadata.get("name", "Unknown"),
                "owner": next((p.get("emailAddress") for p in new_metadata.get("permissions", [])
                               if p.get("role") == "owner"), "Unknown"),
                "permissions": new_metadata.get("permissions", []),
                "modified_by": modifier_info,
                "source": "google_drive",  # Mark the source as Google Drive
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            changes = []

            # Check name changes
            if existing_metadata.get("name") != new_metadata.get("name"):
                changes.append({
                    "type": "name_change",
                    "old": existing_metadata.get("name"),
                    "new": new_metadata.get("name"),
                    "modified_by": modifier_info
                })

            # Check permission changes
            old_perms = {p.get("emailAddress"): p for p in existing_metadata.get("permissions", [])}
            new_perms = {p.get("emailAddress"): p for p in new_metadata.get("permissions", [])}

            for email, perm in new_perms.items():
                if email not in old_perms:
                    changes.append({
                        "type": "permission_added",
                        "user": email,
                        "user_name": perm.get("displayName", "Unknown"),
                        "role": perm.get("role"),
                        "modified_by": modifier_info
                    })
                elif old_perms[email].get("role") != perm.get("role"):
                    changes.append({
                        "type": "permission_changed",
                        "user": email,
                        "user_name": perm.get("displayName", "Unknown"),
                        "old_role": old_perms[email].get("role"),
                        "new_role": perm.get("role"),
                        "modified_by": modifier_info
                    })

            for email, perm in old_perms.items():
                if email not in new_perms:
                    changes.append({
                        "type": "permission_removed",
                        "user": email,
                        "user_name": perm.get("displayName", "Unknown"),
                        "role": perm.get("role"),
                        "modified_by": modifier_info
                    })

            if not changes:
                return None

            change_details = {
                "type": "changes",
                "file_name": new_metadata.get("name", "Unknown"),
                "owner": next((p.get("emailAddress") for p in new_metadata.get("permissions", [])
                               if p.get("role") == "owner"), "Unknown"),
                "changes": changes,
                "modified_by": modifier_info,
                "source": "google_drive",  # Mark the source as Google Drive
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # Update alerts with new hash
        alerts[alert_key] = new_hash
        save_alerts(alerts)

        # Store the new metadata in Pinecone
        try:
            # Create a vector of 384 dimensions with small random values
            vector = [float(0.1) for _ in range(384)]

            # Sanitize metadata for Pinecone
            sanitized_metadata = sanitize_metadata(new_metadata)

            gdrive_index.upsert(vectors=[{
                "id": file_id,
                "metadata": sanitized_metadata,
                "values": vector
            }])
        except Exception as e:
            log_event(f"Error updating Pinecone: {e}")

        # Log and send alert
        log_message = {
            "file_id": file_id,
            "file_name": new_metadata.get("name", "Unknown"),
            "owner": next((p.get("emailAddress") for p in new_metadata.get("permissions", [])
                           if p.get("role") == "owner"), "Unknown"),
            "details": change_details,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        log_event(log_message)

        try:
            send_email_alert(file_id, change_details)
        except Exception as e:
            log_event(f"Failed to send email alert: {e}")

        return change_details

    except Exception as e:
        log_event(f"Error in detect_change: {e}")
        return None


def send_email_alert(file_id, change_details):
    """Send email alert for changes."""
    try:
        # For testing, we'll just log the email content
        sender = "your-email@gmail.com"
        receivers = ["admin@example.com"]

        # Get modifier information
        modifier = change_details.get("modified_by", {})
        modifier_text = f"{modifier.get('name', 'Unknown')}"
        if modifier.get('email') and modifier.get('email') != "Unknown":
            modifier_text += f" ({modifier.get('email')})"

        # Identify the source (Google Drive or Dropbox)
        source = change_details.get("source", "google_drive")
        source_display = "Google Drive" if source == "google_drive" else "Dropbox"

        # Create a more detailed message
        if change_details["type"] == "new_file":
            subject = f"New {source_display} file detected: {change_details['file_name']}"
            body = [
                f"New {source_display} file '{change_details['file_name']}' was created by {modifier_text}",
                f"Owner: {change_details['owner']}",
                "\nInitial permissions:"
            ]
            for perm in change_details.get("permissions", []):
                body.append(f"- {perm.get('displayName', 'Unknown')} ({perm.get('emailAddress')}): {perm.get('role')}")
        elif change_details["type"] == "file_deleted":
            subject = f"{source_display} file deleted: {change_details['file_name']}"
            body = [
                f"{source_display} file '{change_details['file_name']}' was deleted",
                f"Previous owner: {change_details['owner']}",
                f"File ID: {file_id}"
            ]
            if modifier_text != "Unknown":
                body.insert(1, f"Deleted by: {modifier_text}")
        else:
            subject = f"Changes detected in {source_display} file: {change_details['file_name']}"
            body = [f"The following changes were detected in {source_display} file '{change_details['file_name']}':"]

            for change in change_details.get("changes", []):
                change_modifier = change.get("modified_by", {})
                change_modifier_text = f"{change_modifier.get('name', 'Unknown')}"
                if change_modifier.get('email') and change_modifier.get('email') != "Unknown":
                    change_modifier_text += f" ({change_modifier.get('email')})"

                if change["type"] == "name_change":
                    body.append(f"- File renamed from '{change['old']}' to '{change['new']}' by {change_modifier_text}")
                elif change["type"] == "permission_added":
                    body.append(
                        f"- {change_modifier_text} added {change['role']} permission for {change['user_name']} ({change['user']})")
                elif change["type"] == "permission_removed":
                    body.append(
                        f"- {change_modifier_text} removed permission for {change['user_name']} ({change['user']})")
                elif change["type"] == "permission_changed":
                    body.append(
                        f"- {change_modifier_text} changed {change['user_name']} ({change['user']})'s role from {change['old_role']} to {change['new_role']}")

        body.append(f"\nTimestamp: {change_details['timestamp']}")
        body.append(f"File ID: {file_id}")

        # For now, just log the email content instead of sending
        email_content = f"Would send email:\nSubject: {subject}\nBody:\n" + "\n".join(body)
        log_event(email_content)

        # Create a more concise Slack message
        slack_message = f"*{subject}*\n"
        if change_details["type"] == "changes":
            slack_message += "Changes made:\n"
            for line in body[1:]:  # Skip the first line as it's redundant with the subject
                if line.startswith("-"):  # Only include the change lines
                    slack_message += f"{line}\n"
        else:
            slack_message += "\n".join(body)

        # Send to Slack - use different channels based on source
        channel = SLACK_GDRIVE_CHANNEL if source == "google_drive" else SLACK_DROPBOX_CHANNEL
        slack_notify.send_slack_message(
            SLACK_TOKEN,
            channel,
            slack_message
        )

    except Exception as e:
        log_event(f"Failed to prepare email alert: {str(e)}")


def test_logging():
    """Test function for logging functionality."""
    log_event("Testing logging functionality")
    log_event({
        "test": "Sample JSON log entry",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    print("Logging test completed")


def main():
    """Main function to monitor both Google Drive and Dropbox."""
    try:
        # First monitor Google Drive
        monitor_google_drive()

        # Then monitor Dropbox
        monitor_dropbox()

    except Exception as e:
        log_event(f"Error in main monitoring function: {e}")
    finally:
        log_event("All monitoring completed.")


def monitor_google_drive():
    """Shows basic usage of the Drive v3 API."""
    try:
        creds = None
        if os.path.exists("token2.json"):
            creds = Credentials.from_authorized_user_file("token2.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials2.json", SCOPES)
                creds = flow.run_local_server(port=52166)
            with open("token2.json", "w") as token:
                token.write(creds.to_json())

        service = build("drive", "v3", credentials=creds)

        log_event("Real-time Google Drive Metadata Monitoring Started...")

        try:
            # Get all files including trashed ones
            results = service.files().list(
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, permissions, trashed)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True
            ).execute()

            log_event("Successfully fetched Drive metadata!")

            # Get current file IDs
            current_files = {file['id']: file for file in results.get('files', [])}

            # Get previously stored file IDs from Pinecone
            try:
                # Query Pinecone to get all stored file IDs
                query_response = gdrive_index.query(
                    vector=[0.1] * 384,  # Dummy vector
                    top_k=1000,
                    include_metadata=True
                )

                # Convert metadata strings back to objects
                stored_files = {}
                for match in query_response.matches:
                    if match.id and match.metadata:
                        metadata = match.metadata
                        # Convert JSON strings back to objects
                        for key, value in metadata.items():
                            if isinstance(value, str):
                                try:
                                    parsed = json.loads(value)
                                    if isinstance(parsed, (dict, list)):
                                        metadata[key] = parsed
                                except:
                                    pass  # Keep as string if not valid JSON
                        stored_files[match.id] = metadata

                # Check for deleted files (files that were in storage but not in current files)
                for file_id, metadata in stored_files.items():
                    if file_id not in current_files:
                        # File was deleted
                        change_details = {
                            "type": "file_deleted",
                            "file_name": metadata.get("name", "Unknown"),
                            "owner": next((p.get("emailAddress") for p in metadata.get("permissions", [])
                                           if p.get("role") == "owner"), "Unknown"),
                            "source": "google_drive",  # Mark the source as Google Drive
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # Log the deletion
                        log_message = {
                            "file_id": file_id,
                            "file_name": change_details["file_name"],
                            "owner": change_details["owner"],
                            "details": change_details,
                            "timestamp": change_details["timestamp"]
                        }
                        log_event(log_message)

                        try:
                            send_email_alert(file_id, change_details)
                        except Exception as e:
                            log_event(f"Failed to send email alert: {e}")

                        # Remove the deleted file from Pinecone
                        try:
                            gdrive_index.delete(ids=[file_id])
                            log_event(f"Removed deleted file {file_id} from Pinecone")
                        except Exception as e:
                            log_event(f"Error removing deleted file from Pinecone: {e}")

            except Exception as e:
                log_event(f"Error checking for deleted files: {e}")

            # Process current files
            for file in current_files.values():
                process_drive_file(file, service)

        except Exception as error:
            log_event(f'An error occurred: {error}')

    except Exception as e:
        log_event(f"Error in Google Drive monitoring: {e}")
    finally:
        log_event("Google Drive monitoring complete.")


# === Dropbox Functions ===

def load_dropbox_alerts():
    """Load previously sent Dropbox alerts to avoid duplicate notifications."""
    if os.path.exists(DROPBOX_ALERTS_FILE):
        with open(DROPBOX_ALERTS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_dropbox_alerts(alerts):
    """Save the updated Dropbox alerts log."""
    with open(DROPBOX_ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=4)


def get_existing_dropbox_metadata(file_id):
    """Fetch stored Dropbox metadata from Pinecone."""
    try:
        result = dropbox_index.fetch(ids=[file_id])
        if file_id in result.vectors:
            return result.vectors[file_id].metadata
        return {}
    except Exception as e:
        log_event(f"Error fetching Dropbox metadata from Pinecone: {e}")
        return {}


def get_dropbox_files():
    """Fetch all files from Dropbox with their metadata."""
    try:
        result = dbx.files_list_folder("", recursive=True)
        files = []

        # Process initial batch
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                files.append({
                    "id": entry.id,
                    "name": entry.name,
                    "path": entry.path_display,
                    "size": str(entry.size),
                    "client_modified": str(entry.client_modified),
                    "server_modified": str(entry.server_modified)
                })

        # Handle pagination
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    files.append({
                        "id": entry.id,
                        "name": entry.name,
                        "path": entry.path_display,
                        "size": str(entry.size),
                        "client_modified": str(entry.client_modified),
                        "server_modified": str(entry.server_modified)
                    })

        log_event(f"Successfully fetched {len(files)} files from Dropbox")
        return files

    except dropbox.exceptions.ApiError as e:
        log_event(f"Dropbox API error: {e}")
        return []
    except Exception as e:
        log_event(f"Error fetching Dropbox files: {e}")
        return []


def get_dropbox_file_permissions(file_path):
    """Get actual sharing permissions for a Dropbox file."""
    try:
        permissions = []
        modifier_info = None

        # First try to get sharing metadata
        try:
            sharing_info = dbx.sharing_get_file_metadata(file_path)

            if sharing_info:
                # Get file members and sharing history
                try:
                    members = dbx.sharing_list_file_members(file_path)

                    # Get sharing history to identify who made the changes
                    try:
                        # Get the file's sharing history
                        history = dbx.sharing_get_file_metadata(file_path)
                        if history and hasattr(history, 'modified_by'):
                            modifier = history.modified_by
                            modifier_info = {
                                'name': getattr(modifier, 'display_name', 'Unknown'),
                                'email': getattr(modifier, 'email', 'Unknown'),
                                'timestamp': str(datetime.datetime.now())
                            }

                        # If no modifier found in metadata, try to get it from the owner
                        if not modifier_info and hasattr(history, 'owner'):
                            owner = history.owner
                            modifier_info = {
                                'name': getattr(owner, 'display_name', 'Unknown'),
                                'email': getattr(owner, 'email', 'Unknown'),
                                'timestamp': str(datetime.datetime.now())
                            }
                    except Exception as e:
                        log_event(f"Error getting sharing history: {str(e)}")

                    # Process active users
                    if members and hasattr(members, 'users'):
                        for user in members.users:
                            # Get user info
                            user_info = user.user
                            access_type = str(user.access_type).lower()

                            permission = {
                                'displayName': getattr(user_info, 'display_name', 'Unknown'),
                                'emailAddress': getattr(user_info, 'email', 'Unknown'),
                                'role': 'reader',  # Default role
                                'modified_by': modifier_info  # Add modifier info
                            }

                            # Map access levels to roles
                            if 'owner' in access_type:
                                permission['role'] = 'owner'
                            elif 'editor' in access_type:
                                permission['role'] = 'writer'
                            elif 'viewer' in access_type:
                                permission['role'] = 'reader'

                            permissions.append(permission)

                    # Process pending invites
                    if members and hasattr(members, 'invitees'):
                        for invitee in members.invitees:
                            access_type = str(invitee.access_type).lower()

                            # For invitees, get the email value
                            if hasattr(invitee, 'invitee'):
                                invitee_info = invitee.invitee
                                # Get the actual email string value
                                if hasattr(invitee_info, '_email_value'):
                                    invitee_email = invitee_info._email_value
                                elif hasattr(invitee_info, '_value'):
                                    invitee_email = invitee_info._value
                                else:
                                    # Try direct access or string representation
                                    try:
                                        invitee_email = str(invitee_info).split("'")[1]
                                    except:
                                        invitee_email = "Unknown"

                                permission = {
                                    'displayName': 'Pending User',
                                    'emailAddress': invitee_email,
                                    'role': 'reader',  # Default role
                                    'status': 'pending',  # Add status to indicate pending invitation
                                    'modified_by': modifier_info  # Add modifier info
                                }

                                # Map access levels to roles
                                if 'owner' in access_type:
                                    permission['role'] = 'owner'
                                elif 'editor' in access_type:
                                    permission['role'] = 'writer'
                                elif 'viewer' in access_type:
                                    permission['role'] = 'reader'

                                permissions.append(permission)

                    # Also check groups if any
                    if members and hasattr(members, 'groups'):
                        for group in members.groups:
                            group_info = group.group
                            access_type = str(group.access_type).lower()

                            permission = {
                                'displayName': f"Group: {getattr(group_info, 'display_name', 'Unknown Group')}",
                                'emailAddress': f"group:{getattr(group_info, 'group_id', 'unknown')}",
                                'role': 'reader',  # Default role
                                'modified_by': modifier_info  # Add modifier info
                            }

                            # Map access levels to roles
                            if 'editor' in access_type:
                                permission['role'] = 'writer'
                            elif 'viewer' in access_type:
                                permission['role'] = 'reader'

                            permissions.append(permission)

                except Exception as e:
                    log_event(f"Error getting file members: {str(e)}")

                # If no permissions found yet, try to get owner from sharing info
                if not permissions and hasattr(sharing_info, 'access_type'):
                    try:
                        # Get current account as it might be the owner
                        current_account = dbx.users_get_current_account()
                        if current_account and 'owner' in str(sharing_info.access_type).lower():
                            owner_permission = {
                                'displayName': current_account.name.display_name,
                                'emailAddress': current_account.email,
                                'role': 'owner',
                                'modified_by': modifier_info  # Add modifier info
                            }
                            permissions.append(owner_permission)
                    except Exception as e:
                        log_event(f"Error getting current account info: {str(e)}")

        except dropbox.exceptions.ApiError as e:
            if 'sharing.read' in str(e):
                log_event(f"Missing sharing.read permission for file {file_path}")
            else:
                log_event(f"Error getting sharing metadata: {str(e)}")

        # If still no permissions found, try to get file metadata
        if not permissions:
            try:
                metadata = dbx.files_get_metadata(file_path)

                # Try to get owner from file metadata
                if hasattr(metadata, 'sharing_info'):
                    sharing_info = metadata.sharing_info
                    owner_team = getattr(sharing_info, 'owner_team', None)
                    owner_display = getattr(sharing_info, 'owner_display_name', None)

                    if owner_display or owner_team:
                        owner_permission = {
                            'displayName': owner_display or f"Team: {owner_team.name}",
                            'emailAddress': owner_team.team_id if owner_team else 'Unknown',
                            'role': 'owner',
                            'modified_by': modifier_info  # Add modifier info
                        }
                        permissions.append(owner_permission)

                # If still no permissions, try to get the current user's info
                if not permissions:
                    try:
                        current_account = dbx.users_get_current_account()
                        if current_account:
                            owner_permission = {
                                'displayName': current_account.name.display_name,
                                'emailAddress': current_account.email,
                                'role': 'owner',
                                'modified_by': modifier_info  # Add modifier info
                            }
                            permissions.append(owner_permission)
                    except Exception as e:
                        log_event(f"Error getting current account info: {str(e)}")

            except Exception as e:
                log_event(f"Error getting file metadata: {str(e)}")

        return permissions

    except Exception as e:
        log_event(f"Error in get_dropbox_file_permissions: {str(e)}")
        return []


def prepare_pinecone_metadata(file_data, permissions):
    """Prepare metadata for Pinecone storage by converting complex structures to strings."""
    metadata = {
        "id": file_data.get("id", ""),
        "name": file_data.get("name", ""),
        "path": file_data.get("path", ""),
        "size": str(file_data.get("size", "0")),
        "client_modified": str(file_data.get("client_modified", "")),
        "server_modified": str(file_data.get("server_modified", "")),
        "permissions_json": json.dumps(permissions),  # Store permissions as JSON string
        "owner_name": "",  # Initialize owner fields
        "owner_email": "",
        "owner_timestamp": ""
    }

    # Extract owner from permissions and store as separate fields
    owner = next((p for p in permissions if p.get('role') == 'owner'), {})
    if owner:
        metadata["owner_name"] = owner.get('displayName', 'Unknown Owner')
        metadata["owner_email"] = owner.get('emailAddress', 'Unknown')
        metadata["owner_timestamp"] = str(datetime.datetime.now())

    return metadata


def detect_dropbox_change(file_id, new_metadata):
    """Detect and alert on Dropbox metadata changes."""
    try:
        old_metadata = get_existing_dropbox_metadata(file_id)
        alerts_log = load_dropbox_alerts()

        if old_metadata != new_metadata:
            # Parse old and new permissions
            old_permissions = json.loads(old_metadata.get("permissions_json", "[]")) if old_metadata else []
            new_permissions = json.loads(new_metadata.get("permissions_json", "[]"))

            # Create sets of permission tuples for comparison
            old_perm_set = {(p.get('emailAddress', ''), p.get('role', ''), p.get('status', 'active'))
                            for p in old_permissions}
            new_perm_set = {(p.get('emailAddress', ''), p.get('role', ''), p.get('status', 'active'))
                            for p in new_permissions}

            # Detect permission changes
            added_perms = new_perm_set - old_perm_set
            removed_perms = old_perm_set - new_perm_set

            change_details = {
                "file_id": file_id,
                "file_name": new_metadata.get("name", "Unknown"),
                "path": new_metadata.get("path", ""),
                "timestamp": str(datetime.datetime.now()),
                "source": "dropbox",
                "changes": {
                    "metadata_changes": {
                        key: {"old": old_metadata.get(key), "new": new_metadata.get(key)}
                        for key in new_metadata
                        if key != "permissions_json" and old_metadata and old_metadata.get(key) != new_metadata.get(key)
                    },
                    "permission_changes": {
                        "added": [{"email": email, "role": role, "status": status}
                                  for email, role, status in added_perms],
                        "removed": [{"email": email, "role": role, "status": status}
                                    for email, role, status in removed_perms]
                    }
                }
            }

            # Only send alert if there are actual changes
            if (change_details["changes"]["metadata_changes"] or
                    change_details["changes"]["permission_changes"]["added"] or
                    change_details["changes"]["permission_changes"]["removed"]):

                send_dropbox_alert(file_id, change_details)

                # Update Pinecone with new metadata
                try:
                    vector = model.encode(json.dumps(new_metadata)).tolist()
                    dropbox_index.upsert(vectors=[{
                        "id": file_id,
                        "values": vector,
                        "metadata": new_metadata
                    }])
                except Exception as e:
                    log_event(f"Error updating Pinecone with Dropbox metadata: {e}")

                alerts_log[file_id] = change_details["changes"]
                save_dropbox_alerts(alerts_log)
                log_event(f"Dropbox change detected and alerts sent for {file_id}")
                return change_details

        return None
    except Exception as e:
        log_event(f"Error in detect_dropbox_change: {e}")
        return None


def send_dropbox_alert(file_id, change_details):
    """Send rich formatted alerts for Dropbox changes, using the same structure as Google Drive."""
    try:
        file_name = change_details.get("file_name", "Unknown")
        details = change_details.get("details", {})
        path = details.get("path", "")
        event_type = details.get("type", "changes")
        changes = details.get("changes", [])
        owner = change_details.get("owner", {})
        timestamp = change_details.get("timestamp", str(datetime.datetime.now()))

        # Format email content
        subject = f"Dropbox Change: {file_name}"
        body = [
            f"Changes detected in Dropbox file '{file_name}':",
            f"Path: {path}",
            f"File ID: {file_id}",
            f"Owner: {owner.get('name', 'Unknown Owner')} ({owner.get('email', 'Unknown')})",
            f"Timestamp: {timestamp}",
            "\nChanges:"
        ]

        if event_type == "new_file":
            body.append("- New file created")
            perms = details.get("permissions", [])
            if perms:
                body.append("Initial permissions:")
                for perm in perms:
                    status = f" ({perm.get('status')})" if perm.get('status') == 'pending' else ""
                    body.append(
                        f"- {perm.get('displayName', 'Unknown')} ({perm.get('emailAddress', 'Unknown')}): {perm.get('role', 'Unknown')}{status}")
        elif event_type == "file_deleted":
            body.append("- File deleted")
        else:
            for change in changes:
                modifier = change.get("modified_by", {})
                modifier_text = f" by {modifier.get('name', 'Unknown')} ({modifier.get('email', 'Unknown')})" if modifier else ""
                status = f" ({change.get('status')})" if change.get('status') == 'pending' else ""
                if change["type"] == "permission_added":
                    body.append(
                        f"- Added {change['role']} permission for {change['user_name']} ({change['user']}){status}{modifier_text}")
                elif change["type"] == "permission_changed":
                    body.append(
                        f"- Changed permission for {change['user_name']} ({change['user']}) from {change['old_role']} to {change['new_role']}{status}{modifier_text}")
                elif change["type"] == "permission_removed":
                    body.append(
                        f"- Removed {change['role']} permission for {change['user_name']} ({change['user']}){status}{modifier_text}")

        message = "\n".join(body)
        log_event(f"Alert content:\n{message}")
        # TODO: Implement your alert sending mechanism here
        # send_alert(subject, message)
    except Exception as e:
        log_event(f"Error sending Dropbox alert: {e}")


def monitor_dropbox():
    """Main function to monitor Dropbox files."""
    try:
        log_event("Real-time Dropbox Metadata Monitoring Started...")
        dropbox_files = get_dropbox_files()
        try:
            query_response = dropbox_index.query(
                vector=[0.1] * 384,
                top_k=1000,
                include_metadata=True
            )
            stored_files = {}
            for match in query_response.matches:
                if match.id and match.metadata:
                    stored_files[match.id] = match.metadata
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for file in dropbox_files:
                try:
                    file_id = file['id']
                    file_path = file['path']
                    permissions = get_dropbox_file_permissions(file_path)
                    owner_info = next((p for p in permissions if p.get('role') == 'owner'), {})
                    owner_details = {
                        'name': owner_info.get('displayName', 'Unknown Owner'),
                        'email': owner_info.get('emailAddress', 'Unknown'),
                        'timestamp': current_time
                    }
                    pinecone_metadata = prepare_pinecone_metadata(file, permissions)
                    if file_id in stored_files:
                        old_metadata = stored_files[file_id]
                        try:
                            old_permissions = json.loads(old_metadata.get('permissions_json', '[]'))
                        except:
                            old_permissions = []
                        old_perms_dict = {p.get('emailAddress'): p for p in old_permissions if p.get('emailAddress')}
                        new_perms_dict = {p.get('emailAddress'): p for p in permissions if p.get('emailAddress')}
                        changes = []
                        for email, new_perm in new_perms_dict.items():
                            modifier_info = new_perm.get('modified_by') or {
                                'name': owner_info.get('displayName', 'Unknown Owner'),
                                'email': owner_info.get('emailAddress', 'Unknown'),
                                'is_owner': True,
                                'timestamp': current_time
                            }
                            if 'timestamp' not in modifier_info:
                                modifier_info['timestamp'] = current_time
                            if email not in old_perms_dict:
                                changes.append({
                                    "type": "permission_added",
                                    "user": email,
                                    "user_name": new_perm.get('displayName', 'Unknown'),
                                    "role": new_perm.get('role', 'Unknown'),
                                    "status": new_perm.get('status', 'active'),
                                    "modified_by": modifier_info,
                                    "timestamp": current_time
                                })
                            elif old_perms_dict[email].get('role') != new_perm.get('role'):
                                changes.append({
                                    "type": "permission_changed",
                                    "user": email,
                                    "user_name": new_perm.get('displayName', 'Unknown'),
                                    "old_role": old_perms_dict[email].get('role', 'Unknown'),
                                    "new_role": new_perm.get('role', 'Unknown'),
                                    "status": new_perm.get('status', 'active'),
                                    "modified_by": modifier_info,
                                    "timestamp": current_time
                                })
                        for email, old_perm in old_perms_dict.items():
                            if email not in new_perms_dict:
                                modifier_info = old_perm.get('modified_by') or {
                                    'name': owner_info.get('displayName', 'Unknown Owner'),
                                    'email': owner_info.get('emailAddress', 'Unknown'),
                                    'is_owner': True,
                                    'timestamp': current_time
                                }
                                if 'timestamp' not in modifier_info:
                                    modifier_info['timestamp'] = current_time
                                changes.append({
                                    "type": "permission_removed",
                                    "user": email,
                                    "user_name": old_perm.get('displayName', 'Unknown'),
                                    "role": old_perm.get('role', 'Unknown'),
                                    "modified_by": modifier_info,
                                    "timestamp": current_time
                                })
                        if changes:
                            change_event = {
                                "file_id": file_id,
                                "file_name": file['name'],
                                "timestamp": current_time,
                                "owner": owner_details,
                                "details": {
                                    "type": "changes",
                                    "source": "dropbox",
                                    "file_name": file['name'],
                                    "path": file_path,
                                    "changes": changes,
                                    "timestamp": current_time
                                }
                            }
                            log_event(change_event)
                            send_dropbox_alert(file_id, change_event)
                    else:
                        new_file_event = {
                            "file_id": file_id,
                            "file_name": file['name'],
                            "timestamp": current_time,
                            "owner": owner_details,
                            "details": {
                                "type": "new_file",
                                "source": "dropbox",
                                "file_name": file['name'],
                                "path": file_path,
                                "permissions": permissions,
                                "owner": owner_details,
                                "timestamp": current_time
                            }
                        }
                        log_event(new_file_event)
                        send_dropbox_alert(file_id, new_file_event)
                    dropbox_index.upsert(vectors=[{
                        "id": file_id,
                        "metadata": pinecone_metadata,
                        "values": clean_vector(model.encode(file['name'] + " " + file['path']).tolist())
                    }])
                except Exception as e:
                    log_event(f"Error processing Dropbox file {file.get('id', 'Unknown')}: {e}")
                    continue
            current_file_ids = {f['id'] for f in dropbox_files}
            for stored_id in stored_files:
                if stored_id not in current_file_ids:
                    stored_file = stored_files[stored_id]
                    owner_details = {
                        'name': stored_file.get('owner_name', 'Unknown Owner'),
                        'email': stored_file.get('owner_email', 'Unknown'),
                        'timestamp': current_time
                    }
                    deleted_event = {
                        "file_id": stored_id,
                        "file_name": stored_file.get('name', 'Unknown'),
                        "timestamp": current_time,
                        "owner": owner_details,
                        "details": {
                            "type": "file_deleted",
                            "source": "dropbox",
                            "file_name": stored_file.get('name', 'Unknown'),
                            "path": stored_file.get('path', ''),
                            "owner": owner_details,
                            "timestamp": current_time
                        }
                    }
                    log_event(deleted_event)
                    send_dropbox_alert(stored_id, deleted_event)
                    try:
                        dropbox_index.delete(ids=[stored_id])
                    except Exception as e:
                        log_event(f"Error removing deleted Dropbox file from Pinecone: {e}")
        except Exception as e:
            log_event(f"Error checking Dropbox files: {e}")
        except Exception as e:
            log_event(f"Error in Dropbox monitoring: {e}")
    finally:
        log_event("Dropbox monitoring completed.")


# Add a test call after the main function
if __name__ == "__main__":
    # Call the main function if no arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test-log":
        test_logging()
    else:
        main()