"""Streamlit interface for the existing Aether LangGraph agent."""

import asyncio
import ast
import base64
import json
import re
from datetime import datetime, timedelta
from email.utils import parseaddr, parsedate_to_datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.assistant.agent import conversation_state, invoke_agent
from src.assistant.conversations import ConversationManager

ROOT = Path(__file__).parent
SUGGESTIONS = (
    "Check my unread emails",
    "What's on my calendar this week?",
    "Weather in Rawalpindi",
    "Draft an email",
)
EMOJI = re.compile("[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF\ufe0f\u200d\u20e3]")
LOCAL_TIME = ZoneInfo("Asia/Karachi")


def safe(value):
    return escape(str(value))


def logo_image():
    logo = ROOT / "assets" / ("logo_dark.svg" if st.session_state.get("dark_mode") else "logo.svg")
    return "data:image/svg+xml;base64," + base64.b64encode(logo.read_bytes()).decode("ascii")


def setup_page():
    logo = ROOT / "assets" / ("logo_dark.svg" if st.session_state.get("dark_mode") else "logo.svg")
    st.set_page_config(page_title="Aether", page_icon=str(logo),
                       layout="wide", initial_sidebar_state="expanded")
    marker = '<span class="aether-dark-marker" hidden></span>' if st.session_state.get("dark_mode") else ""
    st.markdown(f"<style>{(ROOT / 'styles.css').read_text(encoding='utf-8')}</style>{marker}",
                unsafe_allow_html=True)


def unique_name(conversations):
    base = "New conversation"
    number = 1
    while (base if number == 1 else f"{base} {number}") in conversations:
        number += 1
    return base if number == 1 else f"{base} {number}"


def group_for(snapshot, name):
    created_at = getattr(snapshot, "created_at", None)
    if not created_at:
        return "Today" if name.startswith("New conversation") else "Earlier"
    when = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    date = when.astimezone(LOCAL_TIME).date()
    today = datetime.now(LOCAL_TIME).date()
    if date == today:
        return "Today"
    if date == today - timedelta(days=1):
        return "Yesterday"
    return "Earlier"


def relative_time(snapshot):
    created_at = getattr(snapshot, "created_at", None)
    if not created_at:
        return ""
    when = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    elapsed = max(0, int((datetime.now(LOCAL_TIME) - when.astimezone(LOCAL_TIME)).total_seconds()))
    if elapsed < 3600:
        return f"{max(1, elapsed // 60)}m"
    if elapsed < 86400:
        return f"{elapsed // 3600}h"
    return f"{elapsed // 86400}d"


def sidebar(manager, conversations, snapshots):
    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand"><img src="{logo_image()}" alt=""/>'
                    '<span class="wordmark">Aether</span></div>', unsafe_allow_html=True)
        if st.button("New conversation", use_container_width=True, type="secondary"):
            name = unique_name(conversations)
            manager.create(name)
            st.session_state.active_conversation = name
            st.rerun()
        query = st.text_input("Search conversations", placeholder="Search conversations",
                              label_visibility="collapsed", key="conversation_search").casefold().strip()
        with st.container(key="conversation-list"):
            st.markdown('<div class="sidebar-label">CONVERSATIONS</div>', unsafe_allow_html=True)
            for group in ("Today", "Yesterday", "Earlier"):
              items = [(name, thread_id) for name, thread_id in conversations.items()
                       if query in name.casefold() and group_for(snapshots[name], name) == group]
              if not items:
                  continue
              st.markdown(f'<div class="conversation-group">{group}</div>', unsafe_allow_html=True)
              for name, thread_id in items:
                  active = name == st.session_state.active_conversation
                  with st.container(key="active-conversation" if active else f"conversation-{thread_id}"):
                      row, time_col, menu = st.columns([5, 1.1, .7], gap=None, vertical_alignment="center")
                      with row:
                          if st.button(name, key=f"select-{thread_id}", icon=":material/chat_bubble_outline:",
                                       use_container_width=True):
                              st.session_state.active_conversation = name
                              st.rerun()
                      with time_col:
                          st.markdown(f'<span class="row-time">{relative_time(snapshots[name])}</span>',
                                      unsafe_allow_html=True)
                      with menu:
                          with st.popover("...", key=f"menu-{thread_id}"):
                              with st.form(f"rename-{thread_id}"):
                                  new_name = st.text_input("Rename", value=name)
                                  if st.form_submit_button("Save", use_container_width=True):
                                      try:
                                          manager.rename(name, new_name)
                                      except ValueError as error:
                                          st.error(str(error))
                                      else:
                                          if active:
                                              st.session_state.active_conversation = new_name.strip()
                                          st.rerun()
                              if st.button("Delete conversation", key=f"delete-{thread_id}"):
                                  st.session_state.delete_target = name
                              if st.session_state.get("delete_target") == name:
                                  st.warning("Remove this conversation from the list? Saved checkpoint data remains.")
                                  if st.button("Confirm delete", key=f"confirm-delete-{thread_id}", type="secondary"):
                                      manager.delete(name)
                                      st.session_state.delete_target = None
                                      if active:
                                          st.session_state.active_conversation = next(iter(manager.list()), None)
                                      st.rerun()
                                  if st.button("Cancel", key=f"cancel-delete-{thread_id}"):
                                      st.session_state.delete_target = None
                                      st.rerun()
        with st.container(key="services-footer"):
            with st.popover("Account", icon=":material/account_circle:"):
                st.toggle("Dark mode", key="dark_mode")
                st.markdown('<div class="connected-services"><span class="dot"></span> Gmail'
                            '<span class="dot"></span> Calendar<span class="dot"></span> Weather</div>',
                            unsafe_allow_html=True)


def header(name):
    st.markdown(f'<div class="chat-header"><h1 class="conversation-title" title="{safe(name)}">'
                f'{safe(name)}</h1><span class="header-services">'
                '<span class="status-chip"><i></i>Gmail</span><span class="status-chip"><i></i>Calendar</span>'
                '<span class="status-chip"><i></i>Weather</span><span class="status-chip groq-chip">Groq</span>'
                '</span></div>',
                unsafe_allow_html=True)


def strip_emojis(text):
    return EMOJI.sub("", text)


def clean(value):
    return safe(strip_emojis(str(value)))


def readable(value):
    if isinstance(value, dict):
        return "; ".join(f"{clean(str(key).replace('_', ' ').title())}: {readable(item)}"
                         for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(readable(item) for item in value)
    return clean(value)


def parse_payload(content):
    try:
        data = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return None
    if isinstance(data, list) and data and all(isinstance(block, dict) and "text" in block for block in data):
        values = []
        for block in data:
            try:
                values.append(json.loads(block["text"]))
            except (ValueError, TypeError):
                return None
        return values[0] if len(values) == 1 else values
    return data


def local_datetime(value):
    if not value:
        return None
    try:
        when = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(value).replace(" (UTC)", ""))
        except (TypeError, ValueError, IndexError):
            return None
    return (when if when.tzinfo else when.replace(tzinfo=LOCAL_TIME)).astimezone(LOCAL_TIME)


def short_time(when):
    return when.strftime("%I:%M %p").lstrip("0")


def email_date(value):
    when = local_datetime(value)
    if not when:
        return clean(value)
    days = (datetime.now(LOCAL_TIME).date() - when.date()).days
    if days == 0:
        return f"Today, {short_time(when)}"
    if days == 1:
        return f"Yesterday, {short_time(when)}"
    return when.strftime("%a %-d %b")


def event_parts(start, end=""):
    when = local_datetime(start)
    if not when:
        return "—", "", "", clean(start or "Time unavailable")
    finish = local_datetime(end)
    time = short_time(when) + (f" to {short_time(finish)}" if finish else "") if "T" in str(start) else "All day"
    return when.strftime("%a"), when.strftime("%-d"), when.strftime("%b"), time


def result_card(service, icon, body, detail=""):
    st.markdown(f'<div class="result-card"><div class="result-header"><span class="service-tile '
                f'{service.lower()}-tile material-symbols-outlined">{icon}</span>'
                f'<span>{service}{(" · " + clean(detail)) if detail else ""}</span></div>{body}</div>',
                unsafe_allow_html=True)


def render_structured(kind, data):
    if kind in {"get_recent_emails", "search_emails"} and isinstance(data, (list, dict)):
        rows = []
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and "subject" in item:
                sender = str(item.get("from", ""))
                name, address = parseaddr(sender)
                name = name or address or sender
                initial = next((character.upper() for character in name if character.isalpha()), "?")
                rows.append(f'<div class="email-row"><span class="sender-tile">{clean(initial)}</span>'
                            f'<span class="result-title">{clean(item.get("subject") or "Untitled")}</span>'
                            f'<span class="result-meta email-date">{email_date(item.get("date", ""))}</span>'
                            f'<span class="result-meta sender-name" title="{safe(address)}">{clean(name)}</span></div>')
        if rows:
            result_card("Gmail", "mail", '<div class="result-list">' + "".join(rows) + "</div>",
                        f'{len(rows)} email{"s" if len(rows) != 1 else ""}')
            return True
    if kind in {"get_upcoming_events", "find_upcoming_events", "create_calendar_event"}:
        events = data.get("events", []) if isinstance(data, dict) and "events" in data else data
        events = events if isinstance(events, list) else [events]
        rows = []
        for event in events:
            if not isinstance(event, dict) or "title" not in event:
                continue
            start = event.get("start") or event.get("start_time") or event.get("date", "")
            weekday, day, month, time = event_parts(start, event.get("end") or event.get("end_time", ""))
            location = event.get("location")
            location_html = f'<span class="result-meta">{clean(location)}</span>' if location else ""
            success = ('<span class="success-chip"><span class="material-symbols-outlined">check</span>'
                       'Event created</span>') if kind == "create_calendar_event" else ""
            zone_label = " PKT" if time != "All day" else ""
            rows.append(f'<div class="event-row"><span class="event-date"><small>{clean(weekday)}</small>'
                        f'{clean(day)}<small>{clean(month)}</small></span>'
                        f'<span class="event-info"><span class="result-title">{clean(event["title"])}</span>'
                        f'<span class="result-meta">{clean(time)}{zone_label}</span>{location_html}</span>{success}</div>')
        if rows:
            result_card("Calendar", "calendar_month", '<div class="result-list">' + "".join(rows) + "</div>")
            return True
    if kind == "get_weather" and isinstance(data, dict) and "temperature" in data:
        details = []
        if "humidity" in data:
            details.append(f'{clean(data["humidity"])}% humidity')
        if "wind_speed" in data:
            details.append(f'{clean(data["wind_speed"])} km/h wind')
        for field, label in (("feels_like", "Feels like"), ("max_temperature", "High"), ("min_temperature", "Low")):
            if data.get(field) is not None:
                details.append(f'{label} {clean(data[field])}°C')
        condition = data.get("condition") or data.get("description") or "Current conditions"
        body = f'<div class="weather-row"><div class="weather-main"><span class="weather-city">{clean(data.get("city", ""))}</span>' \
               f'<strong>{clean(data["temperature"])}°C</strong><span>{clean(condition)}</span></div>' \
               f'<div class="weather-details">{" · ".join(details)}</div></div>'
        result_card("Weather", "partly_cloudy_day", body, data.get("city", ""))
        return True
    if kind == "get_forecast" and isinstance(data, dict) and "dates" in data:
        rows = []
        for day, high, low in zip(data["dates"], data["max_temperature"], data["min_temperature"]):
            rows.append(f'<div class="weather-row">{clean(day)} <strong>{clean(high)}°C</strong>'
                        f'<span class="result-meta">Low {clean(low)}°C</span></div>')
        result_card("Weather", "partly_cloudy_day", '<div class="result-list">' + "".join(rows) + '</div>')
        return True
    return False


def render_content(content, remove_emoji=True):
    if not isinstance(content, str):
        content = str(content)
    if remove_emoji:
        content = strip_emojis(content)
    data = parse_payload(content)
    if data is None:
        st.markdown(content)
        return
    if isinstance(data, dict):
        kind = "get_weather" if "temperature" in data else "get_forecast" if "dates" in data else "get_upcoming_events"
        if render_structured(kind, data):
            return
    if isinstance(data, list):
        kind = "get_recent_emails" if data and "subject" in data[0] else "get_upcoming_events"
        if render_structured(kind, data):
            return
    st.markdown(readable(data) or "No results")


def service_for_tool(name):
    if not name:
        return None
    if any(part in name for part in ("email", "contact", "draft")):
        return "Gmail"
    if any(part in name for part in ("calendar", "event")):
        return "Calendar"
    if any(part in name for part in ("weather", "forecast")):
        return "Weather"
    return None


def message_list(messages):
    visible = False
    tool_names = {}
    tool_args = {}
    results = []
    action_records = []
    used_services = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_names.update({call["id"]: call["name"] for call in message.tool_calls})
            tool_args.update({call["id"]: call.get("args", {}) for call in message.tool_calls})
            continue
        if isinstance(message, ToolMessage):
            kind = tool_names.get(message.tool_call_id)
            data = parse_payload(message.content)
            if kind and data is not None:
                results.append((kind, data))
            service = service_for_tool(kind)
            if service and service not in used_services:
                used_services.append(service)
            if kind in {"send_email", "create_calendar_event"}:
                if "cancelled" in str(message.content).lower():
                    action_records.append("Cancelled")
                elif kind == "send_email":
                    action_records.append("Confirmed: Email sent" if isinstance(data, dict) and data.get("status") == "email_sent"
                                          else "Email request finished")
                else:
                    event = tool_args.get(message.tool_call_id, {})
                    when = local_datetime(event.get("start_time"))
                    detail = f', {when.strftime("%a %-d %b")}' if when else ""
                    action_records.append(f'Confirmed: {event.get("title", "Event")}{detail}'
                                          if isinstance(data, dict) and data.get("id") else "Event request finished")
            continue
        if not isinstance(message, (HumanMessage, AIMessage)) or not message.content:
            continue
        visible = True
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        logo = ROOT / "assets" / ("logo_dark.svg" if st.session_state.get("dark_mode") else "logo.svg")
        with st.chat_message(role, avatar=str(logo) if role == "assistant" else None):
            rendered_result = False
            if role == "assistant":
                if used_services:
                    st.markdown(f'<div class="activity-summary"><span class="material-symbols-outlined">check</span>'
                                f'Used {safe(" · ".join(used_services))}</div>', unsafe_allow_html=True)
                    used_services.clear()
                for kind, data in results:
                    rendered_result = render_structured(kind, data) or rendered_result
                results.clear()
            if message.content and not rendered_result:
                render_content(message.content, remove_emoji=role == "assistant")
            if role == "assistant":
                for record in action_records:
                    st.markdown(f'<div class="action-result">{safe(record)}</div>', unsafe_allow_html=True)
                action_records.clear()
            if message.additional_kwargs.get("aether_time"):
                st.markdown(f'<div class="message-time">{safe(message.additional_kwargs["aether_time"])}</div>',
                            unsafe_allow_html=True)
            if role == "assistant":
                copy_control(strip_emojis(str(message.content)))
    return visible


def copy_control(content):
    value = base64.b64encode(content.encode("utf-8")).decode("ascii")
    st.iframe(f"""
      <style>body {{ margin:0; }}
      button {{ border:0; padding:2px; background:transparent; color:#8A8A90; cursor:pointer; }}
      svg {{ width:16px; height:16px; stroke:currentColor; fill:none; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }}</style>
      <button id="copy" aria-label="Copy response" title="Copy response"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg></button>
      <script>
        const value = new TextDecoder().decode(Uint8Array.from(atob('{value}'), c => c.charCodeAt(0)));
        document.getElementById('copy').onclick = async () => {{
          try {{ await navigator.clipboard.writeText(value); document.getElementById('copy').title = 'Copied'; }}
          catch {{ const input = document.createElement('textarea'); input.value = value;
            document.body.appendChild(input); input.select(); const copied = document.execCommand('copy'); input.remove();
            document.getElementById('copy').title = copied ? 'Copied' : 'Copy unavailable'; }}
        }};
      </script>""", width=24, height=24)


def empty_state():
    hour = datetime.now(LOCAL_TIME).hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    st.markdown(f'<div class="empty-state"><img src="{logo_image()}" alt=""/>'
                f'<div class="empty-title">{greeting}</div>'
                '<div class="empty-subtitle">Ask about your inbox, calendar, or the weather.</div></div>',
                unsafe_allow_html=True)
    cards = (("Unread emails", "See what needs your attention", "mail", "gmail"),
             ("This week’s calendar", "Find your upcoming plans", "calendar_month", "calendar"),
             ("Weather in Rawalpindi", "Check today’s conditions", "partly_cloudy_day", "weather"),
             ("Draft an email", "Start a message", "edit_square", "gmail"))
    with st.container(key="suggestion-cards"):
        for row in ((0, 1), (2, 3)):
            columns = st.columns(2)
            for column, index in zip(columns, row):
                title, description, icon, service = cards[index]
                with column:
                    with st.container(key=f"suggestion-{index}"):
                        st.markdown(f'<span class="service-tile {service}-tile material-symbols-outlined">{icon}</span>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<div class="suggestion-title">{safe(title)}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="suggestion-description">{safe(description)}</div>', unsafe_allow_html=True)
                        if st.button(title, key=f"suggest-{SUGGESTIONS[index]}", type="secondary",
                                     use_container_width=True):
                            return SUGGESTIONS[index]
    return None


def pending_actions(snapshot):
    for task in snapshot.tasks:
        for paused in task.interrupts:
            return paused.value
    return []


def confirmation_card(actions):
    decision_key = f"approvals_{st.session_state.thread_id}"
    decisions = st.session_state.setdefault(decision_key, {})
    for action in actions:
        action_id, kind, args = action["id"], action["name"], action["args"]
        logo = ROOT / "assets" / ("logo_dark.svg" if st.session_state.get("dark_mode") else "logo.svg")
        with st.chat_message("assistant", avatar=str(logo)):
            with st.container(border=True, key=f"approval-{action_id}"):
                heading = "Send email" if kind == "send_email" else "Create calendar event"
                st.markdown(f'<div class="approval-heading">{heading}</div>', unsafe_allow_html=True)
                fields = [("To", args.get("to")), ("Subject", args.get("subject"))] if kind == "send_email" else [
                    ("Title", args.get("title"))]
                if kind == "create_calendar_event":
                    start, end = local_datetime(args.get("start_time")), local_datetime(args.get("end_time"))
                    if start:
                        value = f'{start.strftime("%a, %-d %b")} · {short_time(start)}'
                        if end:
                            value += f' to {short_time(end)}'
                        fields.append(("When", value + " PKT"))
                        if end:
                            minutes = round((end - start).total_seconds() / 60)
                            duration = f'{minutes // 60} hour{"s" if minutes // 60 != 1 else ""}' if minutes % 60 == 0 else f'{minutes} minutes'
                            fields.append(("Duration", duration))
                    fields.append(("Location", args.get("location")))
                details = "".join(f'<div class="approval-label">{safe(label)}</div><div class="approval-value">{clean(value)}</div>'
                                  for label, value in fields if value)
                if details:
                    st.markdown(f'<div class="approval-details">{details}</div>', unsafe_allow_html=True)
                if kind == "send_email" and args.get("body"):
                    st.markdown(f'<blockquote class="approval-body">{clean(args["body"])}</blockquote>',
                                unsafe_allow_html=True)
                if action_id in decisions:
                    st.markdown('<div class="action-result">' + ("Confirmed" if decisions[action_id] else "Cancelled")
                                + '</div>', unsafe_allow_html=True)
                else:
                    _, confirm, cancel = st.columns([4, 1, 1], gap="small")
                    if confirm.button("Confirm", key=f"approve-{action_id}", type="primary"):
                        decisions[action_id] = True
                        st.rerun()
                    if cancel.button("Cancel", key=f"reject-{action_id}"):
                        decisions[action_id] = False
                        st.rerun()
    if all(action["id"] in decisions for action in actions):
        activity = []
        status_slot = st.empty()
        try:
            asyncio.run(invoke_agent(st.session_state.thread_id, approvals=decisions,
                                     progress=lambda name: update_activity(status_slot, name, activity)))
        except Exception as error:
            st.error(f"Could not complete the request: {error}")
            return
        st.session_state[decision_key] = {}
        st.rerun()


def title_from_prompt(prompt, conversations):
    words = " ".join(strip_emojis(prompt).split())
    base = words if len(words) <= 30 else words[:29].rstrip() + "…"
    if not base:
        base = "Conversation"
    title = base
    number = 2
    while title in conversations:
        suffix = f" ({number})"
        title = base[:30 - len(suffix)].rstrip() + suffix
        number += 1
    return title


def send_message(thread_id, prompt, first_turn=False, activity_slot=None):
    if first_turn and re.fullmatch(r"New conversation(?: \d+)?", st.session_state.active_conversation):
        manager = ConversationManager(ROOT / "conversations.json")
        title = title_from_prompt(prompt, manager.list())
        manager.rename(st.session_state.active_conversation, title)
        st.session_state.active_conversation = title
    activity = []
    with (activity_slot.container() if activity_slot else st.container()):
        status_slot = st.empty()
        try:
            asyncio.run(invoke_agent(thread_id, message=prompt,
                                     progress=lambda name: update_activity(status_slot, name, activity)))
        except Exception as error:
            st.error(f"Could not reach Aether services: {error}")
            return
    st.rerun()


def update_activity(status_slot, tool_name, activity):
    service = service_for_tool(tool_name)
    if not service:
        return
    if service not in activity:
        activity.append(service)
    label = {"Gmail": "Checking Gmail...", "Calendar": "Checking calendar...",
             "Weather": "Getting weather..."}[service]
    if status_slot:
        status_slot.status(label, expanded=False)


def main():
    setup_page()
    manager = ConversationManager(ROOT / "conversations.json")
    conversations = manager.list()
    if st.session_state.get("active_conversation") not in conversations:
        st.session_state.active_conversation = next(iter(conversations), None)
    snapshots = {name: asyncio.run(conversation_state(thread_id))
                 for name, thread_id in conversations.items()}
    for name, thread_id in list(conversations.items()):
        if not re.fullmatch(r"New conversation(?: \d+)?", name):
            continue
        first = next((item.content for item in snapshots[name].values.get("messages", [])
                      if isinstance(item, HumanMessage) and isinstance(item.content, str)), None)
        if first:
            title = title_from_prompt(first, conversations)
            manager.rename(name, title)
            conversations[title] = conversations.pop(name)
            snapshots[title] = snapshots.pop(name)
            if st.session_state.active_conversation == name:
                st.session_state.active_conversation = title
    sidebar(manager, conversations, snapshots)
    if not st.session_state.active_conversation:
        st.markdown('<div class="empty-title">Aether</div>', unsafe_allow_html=True)
        st.write("Create a conversation to begin.")
        return
    name = st.session_state.active_conversation
    thread_id = conversations[name]
    st.session_state.thread_id = thread_id
    snapshot = snapshots[name]
    actions = pending_actions(snapshot)
    with st.container(key="chat-content"):
        header(name)
        messages = snapshot.values.get("messages", [])
        has_messages = message_list(messages)
        activity_slot = st.empty()
        if actions:
            confirmation_card(actions)
        elif not has_messages:
            suggestion = empty_state()
            if suggestion:
                send_message(thread_id, suggestion, first_turn=True, activity_slot=activity_slot)
    with st.container(key="input-chips"):
        for column, (label, starter) in zip(st.columns(3, gap="small"), (
            ("Email", "Show my unread emails"), ("Calendar", "What's on my calendar this week?"),
            ("Weather", "What's the weather in Rawalpindi?"))):
            if column.button(label, key=f"starter-{label}", disabled=bool(actions)):
                st.session_state.composer = starter
    prompt = st.chat_input("Message Aether", disabled=bool(actions), key="composer")
    if prompt:
        first_turn = not any(isinstance(message, HumanMessage) for message in messages)
        send_message(thread_id, prompt, first_turn=first_turn, activity_slot=activity_slot)


if __name__ == "__main__":
    main()
