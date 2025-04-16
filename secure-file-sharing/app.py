import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, jsonify, render_template
import re
import dropbox

app = Flask(__name__)

# Configure basic logging without rotation to avoid file access issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only
    ]
)
app.logger.setLevel(logging.INFO)
app.logger.info('Flask app startup')


def read_syslog(max_lines=1000):
    """Read and parse the syslog file."""
    logs = []
    try:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syslog.log')
        if not os.path.exists(log_path):
            app.logger.warning(f"Syslog file not found at {log_path}")
            return logs

        line_count = 0
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue

                    # Skip header lines and non-JSON lines
                    if line.startswith('===') or line.startswith('Real-time') or \
                            line.startswith('Successfully') or line.startswith('Monitoring') or \
                            line.startswith('Error') or line.startswith('Would send email') or \
                            line.startswith('Google Drive') or line.startswith('All monitoring') or \
                            line.startswith('Subject:') or line.startswith('Body:') or \
                            line.startswith('The following') or line.startswith('File') or \
                            line.startswith('New file') or line.startswith('Owner:') or \
                            line.startswith('Initial permissions') or line.startswith('Timestamp:') or \
                            line.startswith('-') or line.startswith('removed') or line.startswith('  '):

                        # Check specifically for Dropbox email notifications that contain detailed changes
                        if "Subject: Dropbox Change:" in line:
                            # This is a Dropbox change email notification - capture file info
                            filename = line.replace("Subject: Dropbox Change:", "").strip()
                            path = ""
                            file_id = ""
                            changes = {}
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Read the next few lines to get details
                            next_lines = []
                            for _ in range(10):  # Read up to 10 lines to find relevant info
                                try:
                                    next_line = next(f).strip()
                                    next_lines.append(next_line)

                                    if "Path:" in next_line:
                                        path = next_line.replace("Path:", "").strip()
                                    elif "File ID:" in next_line:
                                        file_id = next_line.replace("File ID:", "").strip()
                                    elif "Timestamp:" in next_line:
                                        timestamp = next_line.replace("Timestamp:", "").strip()
                                    elif "Changes:" in next_line:
                                        # The changes details follow
                                        change_lines = []
                                        for _ in range(5):  # Read up to 5 change lines
                                            try:
                                                change_line = next(f).strip()
                                                if change_line.startswith('-'):
                                                    change_parts = change_line.lstrip('- ').split(': Changed from ')
                                                    if len(change_parts) >= 2:
                                                        key = change_parts[0].strip()
                                                        values = change_parts[1].replace("'", "").split(' to ')
                                                        old_val = values[0].strip()
                                                        new_val = values[1].strip() if len(values) > 1 else None
                                                        changes[key] = {"old": old_val, "new": new_val}
                                            except StopIteration:
                                                break
                                except StopIteration:
                                    break

                            # Create a log entry for this Dropbox change
                            if file_id:
                                log_entry = {
                                    "file_id": file_id,
                                    "file_name": filename,
                                    "owner": "Dropbox User",
                                    "timestamp": timestamp,
                                    "details": {
                                        "source": "dropbox",
                                        "file_name": filename,
                                        "path": path,
                                        "type": "file_deleted" if "status" in changes and changes["status"][
                                            "new"] == "deleted" else "changes",
                                        "changes": changes,
                                        "modified_by": {
                                            "name": "Dropbox System",
                                            "email": "notifications@dropbox.com"
                                        }
                                    }
                                }
                                logs.append(log_entry)
                                line_count += 1

                        continue

                    # Special handling for Dropbox notification lines embedded in logs
                    if "Subject: Dropbox Change:" in line:
                        continue  # Skip, this will be handled above

                    # Parse JSON logs
                    if line.startswith('{') and line.endswith('}'):
                        log_data = json.loads(line)

                        # Ensure all logs have details object and source field
                        if 'details' not in log_data:
                            log_data['details'] = {}

                        # Determine the source based on the structure
                        if 'details' in log_data:
                            # Set default source as google_drive if not specified
                            if 'source' not in log_data['details']:
                                # Check if it's a Dropbox log based on content
                                if ('file_id' in log_data and log_data['file_id'].startswith('id:')) or \
                                        ('path' in log_data['details'] and 'dropbox' in log_data['details'].get('path',
                                                                                                                '').lower()):
                                    log_data['details']['source'] = 'dropbox'
                                else:
                                    log_data['details']['source'] = 'google_drive'

                            # Ensure there's a 'modified_by' field
                            if 'modified_by' not in log_data['details']:
                                # Try to extract from changes if it exists
                                if 'changes' in log_data['details'] and isinstance(log_data['details']['changes'],
                                                                                   list):
                                    for change in log_data['details']['changes']:
                                        if 'modified_by' in change:
                                            log_data['details']['modified_by'] = change['modified_by']
                                            break
                                    else:
                                        # Default if not found in changes
                                        log_data['details']['modified_by'] = {
                                            'name': 'Unknown',
                                            'email': 'unknown@example.com'
                                        }
                                else:
                                    # Default if no changes field
                                    log_data['details']['modified_by'] = {
                                        'name': 'Unknown',
                                        'email': 'unknown@example.com'
                                    }

                        logs.append(log_data)
                        line_count += 1

                        if max_lines and line_count >= max_lines:
                            break
                except json.JSONDecodeError as json_err:
                    app.logger.warning(f"Failed to parse JSON in line: {line[:100]}... Error: {json_err}")
                    continue
                except Exception as e:
                    app.logger.warning(f"Error parsing log line: {str(e)}")
                    continue

        # Sort logs by timestamp (newest first)
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return logs
    except Exception as e:
        app.logger.error(f"Error reading syslog: {str(e)}\n{traceback.format_exc()}")
        return []


@app.route('/')
def index():
    """Render the dashboard."""
    return render_template('index.html')


@app.route('/file-based')
def file_based():
    """Render the file-based view."""
    return render_template('file_based.html')


@app.route('/user-based')
def user_based():
    """Render the user-based view."""
    return render_template('user_based.html')


@app.route('/api/files')
def get_all_files():
    """Get list of all files from logs."""
    try:
        logs = read_syslog(max_lines=None)

        # Track unique files and their latest state
        files = {}

        for log in logs:
            file_id = log.get('file_id')
            if not file_id:
                continue

            details = log.get('details', {})

            # Skip if this is a deletion event
            if details.get('type') == 'file_deleted':
                if file_id in files:
                    del files[file_id]
                continue

            # Update or add file info
            files[file_id] = {
                "id": file_id,
                "name": details.get('file_name', log.get('file_name', 'Unknown')),
                "source": details.get('source', 'google_drive'),
                "path": details.get('path', ''),
                "timestamp": log.get('timestamp', '')
            }

        # Convert to list and sort by name
        file_list = sorted(files.values(), key=lambda x: x['name'].lower())
        return jsonify(file_list)
    except Exception as e:
        app.logger.error(f"Error getting file list: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/file-permissions/<file_id>')
def get_file_permissions(file_id):
    """Get permissions for a specific file."""
    try:
        logs = read_syslog(max_lines=None)
        file_logs = [log for log in logs if log.get('file_id') == file_id]
        if not file_logs:
            return jsonify({"error": "File not found"}), 404
        file_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        latest_log = file_logs[0]
        roles = {"owner": [], "reader": [], "writer": [], "commenter": []}
        details = latest_log.get('details', {})
        # Unified permission extraction for both sources
        processed_users = set()
        for log in file_logs:
            details = log.get('details', {})
            if details.get('type') == 'new_file':
                permissions = details.get('permissions', [])
                for perm in permissions:
                    email = perm.get('emailAddress', '').lower()
                    if email not in processed_users:
                        role = perm.get('role', '').lower()
                        if role in roles:
                            roles[role].append({
                                "name": perm.get('displayName', 'Unknown'),
                                "email": email
                            })
                            processed_users.add(email)
            elif details.get('type') == 'changes':
                changes = details.get('changes', [])
                for change in changes:
                    change_type = change.get('type')
                    email = change.get('user', '').lower()
                    if change_type == 'permission_added' and email not in processed_users:
                        role = change.get('role', '').lower()
                        if role in roles:
                            roles[role].append({
                                "name": change.get('user_name', 'Unknown'),
                                "email": email
                            })
                            processed_users.add(email)
                    elif change_type == 'permission_changed' and email not in processed_users:
                        new_role = change.get('new_role', '').lower()
                        if new_role in roles:
                            roles[new_role].append({
                                "name": change.get('user_name', 'Unknown'),
                                "email": email
                            })
                            processed_users.add(email)
                    elif change_type == 'permission_removed':
                        for role_users in roles.values():
                            role_users[:] = [u for u in role_users if u.get('email', '').lower() != email]
        file_info = {
            "name": latest_log.get('file_name') or details.get('file_name', 'Unknown'),
            "source": details.get('source', 'google_drive')
        }
        return jsonify({"file": file_info, "permissions": roles})
    except Exception as e:
        app.logger.error(f"Error getting file permissions: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/user-files/<email>')
def get_user_files(email):
    """Get all files and their permissions for a specific user."""
    try:
        # Get all logs
        logs = read_syslog(max_lines=None)

        # Initialize result structure
        files = {
            "owned": [],
            "can_edit": [],
            "can_comment": [],
            "can_view": []
        }

        # Keep track of processed files to avoid duplicates
        processed_files = set()

        # Process logs in reverse chronological order
        for log in sorted(logs, key=lambda x: x.get('timestamp', ''), reverse=True):
            file_id = log.get('file_id')
            if not file_id or file_id in processed_files:
                continue

            details = log.get('details', {})

            # Skip deleted files
            if details.get('type') == 'file_deleted':
                continue

            file_info = {
                "id": file_id,
                "name": details.get('file_name', log.get('file_name', 'Unknown')),
                "source": details.get('source', 'google_drive'),  # Default to google_drive if not specified
                "path": details.get('path', '')
            }

            # Check permissions
            permissions = []
            if details.get('type') == 'new_file':
                permissions = details.get('permissions', [])
            elif details.get('type') == 'changes':
                # For changes, collect all permission changes
                changes = details.get('changes', [])
                for change in changes:
                    if change.get('type') in ['permission_added', 'permission_changed']:
                        permissions.append({
                            'emailAddress': change.get('user'),
                            'role': change.get('new_role', change.get('role')),
                            'displayName': change.get('user_name', 'Unknown')
                        })

            # Check user's permissions
            for perm in permissions:
                if perm.get('emailAddress', '').lower() == email.lower():
                    role = perm.get('role', '').lower()
                    if role == 'owner':
                        files['owned'].append(file_info)
                    elif role == 'writer':
                        files['can_edit'].append(file_info)
                    elif role == 'commenter':
                        files['can_comment'].append(file_info)
                    elif role == 'reader':
                        files['can_view'].append(file_info)
                    break

            processed_files.add(file_id)

        return jsonify(files)
    except Exception as e:
        app.logger.error(f"Error getting user files: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/logs')
def get_logs():
    """Get logs from the syslog file."""
    try:
        logs = read_syslog(max_lines=None)  # Read all logs
        return jsonify(logs)
    except Exception as e:
        app.logger.error(f"Error reading logs: {str(e)}\n{traceback.format_exc()}")
        return jsonify([])


@app.route('/api/stats')
def get_stats():
    """Get statistics of the logs."""
    try:
        # Read logs
        logs = read_syslog(max_lines=None)  # Read all logs

        if not logs:
            # Return empty stats if no logs found - no sample data
            return jsonify({
                'google_drive': {
                    'total_changes': 0,
                    'new_files': 0,
                    'deleted_files': 0,
                    'permission_changes': 0,
                    'last_24h': 0
                },
                'dropbox': {
                    'total_changes': 0,
                    'new_files': 0,
                    'deleted_files': 0,
                    'last_24h': 0
                },
                'total': {
                    'total_changes': 0,
                    'last_24h': 0
                },
                'gdrive_logs': [],
                'dropbox_logs': []
            })

        # Separate logs by source
        gdrive_logs = []
        dropbox_logs = []

        for log in logs:
            try:
                source = log.get('details', {}).get('source', 'google_drive')
                if source == 'dropbox':
                    dropbox_logs.append(log)
                else:
                    gdrive_logs.append(log)
            except Exception as e:
                app.logger.warning(f"Error sorting log by source: {e}")

        # Google Drive stats - only from actual logs
        gdrive_total = len(gdrive_logs)
        gdrive_new_files = sum(1 for log in gdrive_logs if log.get('details', {}).get('type') == 'new_file')
        gdrive_deleted_files = sum(1 for log in gdrive_logs if log.get('details', {}).get('type') == 'file_deleted')
        gdrive_permission_changes = sum(1 for log in gdrive_logs
                                        if log.get('details', {}).get('type') == 'changes'
                                        and log.get('details', {}).get('changes', []))

        # Dropbox stats - only from actual logs
        dropbox_total = len(dropbox_logs)
        dropbox_new_files = sum(1 for log in dropbox_logs
                                if (log.get('details', {}).get('type') == 'changes'
                                    and 'status' not in log.get('details', {}).get('changes', {})))
        dropbox_deleted_files = sum(1 for log in dropbox_logs if log.get('details', {}).get('type') == 'file_deleted')

        # Recent activity (last 24 hours) - only from actual logs
        try:
            last_24h = datetime.now() - timedelta(hours=24)

            # Count Google Drive logs in the last 24 hours
            gdrive_recent = 0
            for log in gdrive_logs:
                try:
                    if 'timestamp' in log:
                        log_time = datetime.strptime(log['timestamp'].split('.')[0], '%Y-%m-%d %H:%M:%S')
                        if log_time > last_24h:
                            gdrive_recent += 1
                except Exception as e:
                    app.logger.warning(f"Error parsing timestamp in Google Drive log: {e}")

            # Count Dropbox logs in the last 24 hours
            dropbox_recent = 0
            for log in dropbox_logs:
                try:
                    if 'timestamp' in log:
                        log_time = datetime.strptime(log['timestamp'].split('.')[0], '%Y-%m-%d %H:%M:%S')
                        if log_time > last_24h:
                            dropbox_recent += 1
                except Exception as e:
                    app.logger.warning(f"Error parsing timestamp in Dropbox log: {e}")

        except Exception as e:
            # Default to zero if calculation fails
            app.logger.error(f"Error calculating recent logs: {e}")
            gdrive_recent = 0
            dropbox_recent = 0

        # Combined stats - only from actual logs
        total_changes = gdrive_total + dropbox_total
        total_recent = gdrive_recent + dropbox_recent

        # Return stats with logs for chart rendering - no sample data
        response_data = {
            'google_drive': {
                'total_changes': gdrive_total,
                'new_files': gdrive_new_files,
                'deleted_files': gdrive_deleted_files,
                'permission_changes': gdrive_permission_changes,
                'last_24h': gdrive_recent
            },
            'dropbox': {
                'total_changes': dropbox_total,
                'new_files': dropbox_new_files,
                'deleted_files': dropbox_deleted_files,
                'last_24h': dropbox_recent
            },
            'total': {
                'total_changes': total_changes,
                'last_24h': total_recent
            },
            'gdrive_logs': gdrive_logs,
            'dropbox_logs': dropbox_logs
        }

        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error calculating stats: {str(e)}\n{traceback.format_exc()}")
        # Return empty structure on error - no sample data
        return jsonify({
            'error': str(e),
            'google_drive': {'total_changes': 0, 'new_files': 0, 'deleted_files': 0, 'permission_changes': 0,
                             'last_24h': 0},
            'dropbox': {'total_changes': 0, 'new_files': 0, 'deleted_files': 0, 'last_24h': 0},
            'total': {'total_changes': 0, 'last_24h': 0},
            'gdrive_logs': [],
            'dropbox_logs': []
        })


@app.route('/api/users')
def get_users():
    """Get list of all users from logs."""
    try:
        logs = read_syslog(max_lines=None)
        users = set()

        # Extract users from logs
        for log in logs:
            details = log.get('details', {})

            # Get owner
            if 'owner' in log and '@' in log['owner']:
                users.add((log['owner'], details.get('source', 'google_drive')))

            # Get users from initial permissions for new files
            if details.get('type') == 'new_file':
                permissions = details.get('permissions', [])
                for perm in permissions:
                    email = perm.get('emailAddress')
                    if email and '@' in email:
                        users.add((email, details.get('source', 'google_drive')))

            # Get users from permission changes
            changes = details.get('changes', {})
            if isinstance(changes, dict):
                # Handle permission changes in key-value format
                for key, value in changes.items():
                    if isinstance(value, dict):
                        # Check both old and new values for emails
                        for val in [value.get('old', ''), value.get('new', '')]:
                            if isinstance(val, str) and '@' in val:
                                users.add((val, details.get('source', 'google_drive')))
            elif isinstance(changes, list):
                # Handle permission changes in list format
                for change in changes:
                    if isinstance(change, dict):
                        # Check various fields that might contain user emails
                        email_fields = ['email', 'user', 'emailAddress']
                        for field in email_fields:
                            email = change.get(field)
                            if email and '@' in str(email):
                                users.add((email, details.get('source', 'google_drive')))

            # Get modified_by user
            modified_by = details.get('modified_by', {})
            if isinstance(modified_by, dict):
                email = modified_by.get('email')
                if email and '@' in email:
                    users.add((email, details.get('source', 'google_drive')))

            # Get users from direct permission lists
            permissions = details.get('permissions', [])
            if isinstance(permissions, list):
                for perm in permissions:
                    if isinstance(perm, dict):
                        email = perm.get('emailAddress')
                        if email and '@' in email:
                            users.add((email, details.get('source', 'google_drive')))

        # Convert set to list of dictionaries and sort by email
        user_list = sorted([{"email": email, "source": source} for email, source in users],
                           key=lambda x: x['email'].lower())
        return jsonify(user_list)
    except Exception as e:
        app.logger.error(f"Error getting users: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/user-permissions/<email>')
def get_user_permissions(email):
    """Get all file permissions for a specific user by leveraging file-based permissions."""
    try:
        # Get all files first
        all_files_response = get_all_files()
        if not isinstance(all_files_response.json, list):
            return jsonify({"error": "Failed to get files"}), 500

        owned_files = []
        editable_files = []
        commentable_files = []
        viewable_files = []

        # Process each file's permissions
        for file_info in all_files_response.json:
            file_id = file_info.get('id')
            if not file_id:
                continue

            # Get permissions for this file
            file_perms_response = get_file_permissions(file_id)
            if not isinstance(file_perms_response.json, dict):
                continue

            permissions = file_perms_response.json.get('permissions', {})

            # Check each permission category
            if any(user.get('email', '').lower() == email.lower() for user in permissions.get('owner', [])):
                owned_files.append(file_info)
            elif any(user.get('email', '').lower() == email.lower() for user in permissions.get('writer', [])):
                editable_files.append(file_info)
            elif any(user.get('email', '').lower() == email.lower() for user in permissions.get('commenter', [])):
                commentable_files.append(file_info)
            elif any(user.get('email', '').lower() == email.lower() for user in permissions.get('reader', [])):
                viewable_files.append(file_info)

        # Sort files by name
        for file_list in [owned_files, editable_files, commentable_files, viewable_files]:
            file_list.sort(key=lambda x: x['name'].lower())

        return jsonify({
            'owned_files': owned_files,
            'editable_files': editable_files,
            'commentable_files': commentable_files,
            'viewable_files': viewable_files
        })
    except Exception as e:
        app.logger.error(f"Error getting user permissions: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
