from mcp.server.fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

mcp = FastMCP("Gmail Server")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose"
]


def get_gmail_service():
    credentials = Credentials.from_authorized_user_file(
        "token.json",
        SCOPES
    )

    return build("gmail", "v1", credentials=credentials)


@mcp.tool()
def get_recent_emails(limit: int = 5):
    """Get the user's most recent Gmail messages."""

    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        maxResults=limit
    ).execute()

    messages = results.get("messages", [])

    emails = []

    for message in messages:
        email = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        headers = email["payload"]["headers"]

        header_dict = {
            header["name"]: header["value"]
            for header in headers
        }

        emails.append({
            "id": message["id"],
            "from": header_dict.get("From", ""),
            "subject": header_dict.get("Subject", ""),
            "date": header_dict.get("Date", "")
        })

    return emails




@mcp.tool()
def search_emails(query: str, limit: int = 5):
    """Search the user's Gmail messages."""

    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=limit
    ).execute()

    messages = results.get("messages", [])

    emails = []

    for message in messages:
        email = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        headers = email["payload"]["headers"]

        header_dict = {
            header["name"]: header["value"]
            for header in headers
        }

        emails.append({
            "id": message["id"],
            "from": header_dict.get("From", ""),
            "subject": header_dict.get("Subject", ""),
            "date": header_dict.get("Date", "")
        })

    return emails


@mcp.tool()
def get_email(message_id: str):
    """Get the full content of a Gmail message."""

    service = get_gmail_service()

    email = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute()

    headers = email["payload"]["headers"]

    header_dict = {
        header["name"]: header["value"]
        for header in headers
    }

    body = ""

    payload = email["payload"]

    if "body" in payload and payload["body"].get("data"):
        import base64

        body = base64.urlsafe_b64decode(
            payload["body"]["data"]
        ).decode("utf-8", errors="ignore")

    elif "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                if part["body"].get("data"):
                    import base64

                    body = base64.urlsafe_b64decode(
                        part["body"]["data"]
                    ).decode("utf-8", errors="ignore")
                    break

    return {
        "id": message_id,
        "from": header_dict.get("From", ""),
        "to": header_dict.get("To", ""),
        "subject": header_dict.get("Subject", ""),
        "date": header_dict.get("Date", ""),
        "body": body
    }


@mcp.tool()
def create_draft(to: str, subject: str, body: str):
    """Create a Gmail draft without sending it."""

    service = get_gmail_service()

    if to.lower() == "me":
        profile = service.users().getProfile(userId="me").execute()
        to = profile["emailAddress"]

    import base64
    from email.mime.text import MIMEText

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    draft = service.users().drafts().create(
        userId="me",
        body={
            "message": {
                "raw": encoded_message
            }
        }
    ).execute()

    return {
        "status": "draft_created",
        "draft_id": draft["id"],
        "to": to,
        "subject": subject
    }



if __name__ == "__main__":
    mcp.settings.port = 8001
    mcp.run(transport="streamable-http")