import os
import base64
from email.mime.text import MIMEText
from functools import lru_cache
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopes: read & send mail
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly",
          "https://www.googleapis.com/auth/gmail.send"]

def gmail_service():
    creds = None
    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except Exception as e:
            print(f"⚠️ Warning: Could not load token.json: {e}")
            print("🔄 Will attempt to re-authenticate...")
            # Remove the corrupted token file
            if os.path.exists("token.json"):
                os.remove("token.json")
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("✅ Successfully refreshed expired token")
            except Exception as e:
                print(f"❌ Failed to refresh token: {e}")
                print("🔄 Will re-authenticate...")
                creds = None
        
        if not creds:
            # Local OAuth flow with browser
            if not os.path.exists("credentials.json"):
                print("❌ Gmail Authentication Error:")
                print("=" * 50)
                print("Missing credentials.json file!")
                print("\nTo fix this:")
                print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
                print("2. Create a new project or select existing one")
                print("3. Enable Gmail API")
                print("4. Create OAuth 2.0 credentials (Desktop application)")
                print("5. Download the JSON file and save it as 'credentials.json' in this directory")
                print("6. Run the script again")
                print("=" * 50)
                raise FileNotFoundError(
                    "credentials.json not found in project root. Please follow the setup instructions above."
                )
            
            print("🔐 Starting Gmail authentication...")
            print("A browser window will open for Google OAuth consent.")
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
            )
            # This will open a browser window for Google consent and capture token locally
            creds = flow.run_local_server(port=0)
            print("✅ Authentication successful!")

        # Save credentials
        with open("token.json", "w") as token:
            token.write(creds.to_json())
        print("💾 Credentials saved to token.json")

    return build("gmail", "v1", credentials=creds)

@lru_cache(maxsize=1)
def get_service():
    svc = gmail_service()
    return svc


def send_email(to_address: str, subject: str, body_text: str, *, service_override=None) -> dict:
    """Send a plain-text email using Gmail API.

    Returns the API response dict from messages.send
    """
    message = MIMEText(body_text)
    message["to"] = to_address
    message["subject"] = subject

    raw_bytes = base64.urlsafe_b64encode(message.as_bytes())
    raw_str = raw_bytes.decode()
    send_body = {"raw": raw_str}
    svc = service_override or get_service()
    return svc.users().messages().send(userId="me", body=send_body).execute()

