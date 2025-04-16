import os.path
import pprint
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
#SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
SCOPES = ["https://www.googleapis.com/auth/drive"]

def main():
    """
    Shows basic usage of the Drive v3 API.
    Prints the file meta data information from google drive
    """
    creds = None

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token1.json"):
        creds = Credentials.from_authorized_user_file("token1.json", SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    # --- This would prompt you authorize the login on the browser ---
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                                                "credentials.json", SCOPES)

            creds = flow.run_local_server(port=52166)

        # authorization complete on the browser, then a token1.json would be
        # created for the next run
        # Save the credentials for the next run
        with open("token1.json", "w") as token:
            token.write(creds.to_json())

    # Fetch the files & its properties from the drive
    try:
        service = build("drive", "v3", credentials=creds)

        # Call the Drive v3 API
        results = (
                    service.files()
                    .list(pageSize=100, fields="nextPageToken, files(id, name, description, mimeType,md5Checksum, parents, trashed, permissions)")
                    .execute()
                  )
        items = results.get("files", [])

        if not items:
            print("No files found.")
            return

        print("File properties:")
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(items)
        with open("drive_files2.json", "w") as f:
            f.write(json.dumps(items, indent=4))
    except HttpError as error:
        # Handle errors from drive API.
        print(f"An error occurred: {error}")

if __name__ == "__main__":
  main()
