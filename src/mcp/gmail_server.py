from mcp.server.fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import base64
import os
import json


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

mcp = FastMCP("Gmail Server")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.readonly"
]


def get_gmail_service():
    credentials = Credentials.from_authorized_user_file(
        "token.json",
        SCOPES
    )

    return build("gmail", "v1", credentials=credentials)


def decode_body(data):
    """Decode Gmail's URL-safe base64 body data."""

    if not data:
        return ""

    try:
        return base64.urlsafe_b64decode(data).decode(
            "utf-8",
            errors="ignore"
        )
    except Exception:
        return ""


def extract_text_from_payload(payload):
    """Extract useful text from a Gmail MIME payload."""

    # Direct body
    if payload.get("body", {}).get("data"):
        body = decode_body(payload["body"]["data"])

        if payload.get("mimeType") == "text/html":
            soup = BeautifulSoup(body, "html.parser")

            # Remove things that add lots of useless text
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            body = soup.get_text(" ", strip=True)

        return body

    # Multipart email
    parts = payload.get("parts", [])

    # Prefer plain text
    for part in parts:
        if part.get("mimeType") == "text/plain":
            body = decode_body(
                part.get("body", {}).get("data")
            )

            if body:
                return body

    # Fall back to HTML
    for part in parts:
        if part.get("mimeType") == "text/html":
            body = decode_body(
                part.get("body", {}).get("data")
            )

            if body:
                soup = BeautifulSoup(body, "html.parser")

                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

                return soup.get_text(" ", strip=True)

    # Handle nested multipart messages
    for part in parts:
        if "parts" in part:
            body = extract_text_from_payload(part)

            if body:
                return body

    return ""


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
def search_emails(query: str, limit: int = 10):
    """
    Search the user's Gmail mailbox.

    Use this when you need to discover emails relevant to the
    user's request. The query should describe what information
    you are looking for.

    Search results contain email metadata. If a result appears
    relevant, use get_email with its message ID to inspect the
    actual email contents before drawing conclusions.
    """

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
    """Get a cleaned and limited version of a Gmail message."""

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

    body = extract_text_from_payload(email["payload"])

    # Normalize whitespace
    body = " ".join(body.split())

    # Prevent huge emails from blowing up the LLM context
    MAX_BODY_LENGTH = 5000

    if len(body) > MAX_BODY_LENGTH:
        body = body[:MAX_BODY_LENGTH]
        body += "\n[Email body truncated]"

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
        profile = service.users().getProfile(
            userId="me"
        ).execute()

        to = profile["emailAddress"]

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


@mcp.tool()
def send_email(to: str, subject: str, body: str):
    """Send an email through Gmail."""

    service = get_gmail_service()

    if to.lower() == "me":
        profile = service.users().getProfile(
            userId="me"
        ).execute()

        to = profile["emailAddress"]

    from email.mime.text import MIMEText

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    sent = service.users().messages().send(
        userId="me",
        body={"raw": encoded_message}
    ).execute()

    return {
        "status": "email_sent",
        "message_id": sent["id"],
        "to": to,
        "subject": subject
    }


@mcp.tool()
def search_contacts(name: str):
    """Find email addresses from previous Gmail messages involving a person."""

    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        q=f'"{name}"',
        maxResults=10
    ).execute()

    messages = results.get("messages", [])

    contacts = []

    for message in messages:
        email = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="metadata",
            metadataHeaders=["From", "To"]
        ).execute()

        headers = {
            h["name"]: h["value"]
            for h in email["payload"]["headers"]
        }

        contacts.append({
            "from": headers.get("From", ""),
            "to": headers.get("To", "")
        })

    return contacts


@mcp.tool()
def find_upcoming_events(limit: int = 15):
    """
    Find actual upcoming events or plans from recent personal emails.

    The tool uses an LLM to extract structured event information
    from email content. It does not rely on keyword matching.
    """

    service = get_gmail_service()

    query = (
        "newer_than:90d "
        "-category:promotions "
        "-category:social "
        "-category:forums "
        "-category:updates"
    )

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
            format="full"
        ).execute()

        headers = {
            header["name"]: header["value"]
            for header in email["payload"]["headers"]
        }

        body = extract_text_from_payload(email["payload"])
        body = " ".join(body.split())

        if not body:
            continue

        if len(body) > 2500:
            body = body[:2500] + "\n[Email body truncated]"

        emails.append({
            "id": message["id"],
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "body": body
        })

    if not emails:
        return {
            "events": []
        }

    email_text = json.dumps(emails, ensure_ascii=False)

    prompt = f"""
You are an event extraction system.

Analyze these emails and identify ONLY genuine upcoming
personal events, plans, appointments, meetings, classes,
trips, reservations, sports activities, or other commitments.

Do NOT treat normal conversations, emotional messages,
password resets, security notifications, test emails,
news, promotions, or unrelated messages as events.

Do not guess missing information.

Return ONLY valid JSON in this exact format:

[
  {{
    "message_id": "email id",
    "title": "event name",
    "date": "YYYY-MM-DD or null",
    "time": "HH:MM or null",
    "location": "location or null",
    "description": "short description"
  }}
]

Emails:

{email_text}
"""

    response = llm.invoke(prompt)

    try:
        events = json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "events": [],
            "error": "Could not parse event extraction result."
        }

    return {
        "events": events
    }

if __name__ == "__main__":
    mcp.settings.port = 8001
    mcp.run(transport="streamable-http")