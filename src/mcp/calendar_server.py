# MCP boundary for reading and creating Google Calendar events.
from mcp.server.fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timezone

mcp = FastMCP("Calendar Server")

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events"
]


def get_calendar_service():
    credentials = Credentials.from_authorized_user_file(
        "token.json",
        SCOPES
    )

    return build(
        "calendar",
        "v3",
        credentials=credentials
    )

@mcp.tool()
def get_upcoming_events(limit: int = 10):
    """
    Get the user's upcoming Google Calendar events.
    """

    service = get_calendar_service()

    now = datetime.now(timezone.utc).isoformat()

    events_result = service.events().list(
        calendarId="primary",
        timeMin=now,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])

    results = []

    seen_recurring = set()

    for event in events:

        start = event.get("start", {})

        start_time = (
            start.get("dateTime")
            or start.get("date")
        )

        title = event.get("summary", "")

        # Avoid filling the results with repeated yearly
        # occurrences of the same all-day event.
        if "recurringEventId" in event:
            recurring_id = event["recurringEventId"]

            if recurring_id in seen_recurring:
                continue

            seen_recurring.add(recurring_id)

        results.append({
            "id": event.get("id"),
            "title": title,
            "start": start_time,
            "location": event.get("location", ""),
            "description": event.get("description", "")
        })

        if len(results) >= limit:
            break

    return results

@mcp.tool()
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    location: str = "",
    description: str = ""
):
    """
    Create a new event in the user's primary Google Calendar.
    """

    service = get_calendar_service()

    event = {
        "summary": title,
        "start": {
            "dateTime": start_time,
            "timeZone": "Asia/Karachi",
        },
        "end": {
            "dateTime": end_time,
            "timeZone": "Asia/Karachi",
        },
    }

    if location:
        event["location"] = location

    if description:
        event["description"] = description

    created_event = service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

    return {
        "id": created_event.get("id"),
        "title": created_event.get("summary"),
        "start": created_event.get("start", {}).get("dateTime"),
        "end": created_event.get("end", {}).get("dateTime"),
        "location": created_event.get("location", ""),
        "description": created_event.get("description", ""),
    }


if __name__ == "__main__":
    mcp.settings.port = 8002
    mcp.run(transport="streamable-http")
