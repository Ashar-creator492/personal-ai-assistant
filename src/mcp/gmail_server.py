from mcp.server.fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

mcp = FastMCP("Gmail Server")

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


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


if __name__ == "__main__":
    mcp.settings.port = 8001
    mcp.run(transport="streamable-http")