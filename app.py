"""Streamlit interface for the existing Aether LangGraph agent."""

import asyncio
import ast
import base64
import json
import re
from datetime import datetime, timedelta
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
            st.markdown('<div class="connected-services"><span class="dot"></span> Gmail'
                        '<span class="dot"></span> Calendar<span class="dot"></span> Weather</div>',
                        unsafe_allow_html=True)
            st.toggle("Dark mode", key="dark_mode")


def header(name):
    st.markdown(f'<div class="chat-header"><h1 class="conversation-title" title="{safe(name)}">'
                f'{safe(name)}</h1><span class="service-label">Gmail · Calendar · Weather</span></div>',
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


def event_parts(start, end=""):
    try:
        when = datetime.fromisoformat(start)
        day = when.strftime("%d")
        month = when.strftime("%b")
        time = when.strftime("%-I:%M %p") if "T" in start else "All day"
        if end:
            time += " – " + datetime.fromisoformat(end).strftime("%-I:%M %p")
        return day, month, time
    except (ValueError, TypeError):
        return "—", "", start or "Time unavailable"


def render_structured(kind, data):
    if kind in {"get_recent_emails", "search_emails"} and isinstance(data, (list, dict)):
        rows = []
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and "subject" in item:
                sender = str(item.get("from", ""))
                initial = next((character.upper() for character in sender if character.isalpha()), "?")
                rows.append(f'<div class="email-row"><span class="sender-tile">{clean(initial)}</span>'
                            f'<span class="result-title">{clean(item.get("subject") or "Untitled")}</span>'
                            f'<span class="result-meta">{clean(item.get("date", ""))}</span>'
                            f'<span class="result-meta sender-name">{clean(sender)}</span></div>')
        if rows:
            st.markdown('<div class="result-list">' + "".join(rows) + "</div>", unsafe_allow_html=True)
            return True
    if kind in {"get_upcoming_events", "find_upcoming_events", "create_calendar_event"}:
        events = data.get("events", []) if isinstance(data, dict) and "events" in data else data
        events = events if isinstance(events, list) else [events]
        rows = []
        for event in events:
            if not isinstance(event, dict) or "title" not in event:
                continue
            start = event.get("start") or event.get("start_time") or event.get("date", "")
            day, month, time = event_parts(start, event.get("end") or event.get("end_time", ""))
            rows.append(f'<div class="event-row"><span class="event-date">{clean(day)}<small>{clean(month)}</small></span>'
                        f'<span><span class="result-title">{clean(event["title"])}</span>'
                        f'<br><span class="result-meta">{clean(time)}</span></span></div>')
        if rows:
            st.markdown('<div class="result-list">' + "".join(rows) + "</div>", unsafe_allow_html=True)
            return True
    if kind == "get_weather" and isinstance(data, dict) and "temperature" in data:
        details = [f'{clean(data.get("city", ""))}', f'<strong>{clean(data["temperature"])}°C</strong>']
        if "humidity" in data:
            details.append(f'{clean(data["humidity"])}% humidity')
        if "wind_speed" in data:
            details.append(f'{clean(data["wind_speed"])} km/h wind')
        st.markdown('<div class="weather-row"><span class="service-tile weather-tile material-symbols-outlined">partly_cloudy_day</span>'
                    + "<span>·</span>".join(details) + '</div>', unsafe_allow_html=True)
        return True
    if kind == "get_forecast" and isinstance(data, dict) and "dates" in data:
        rows = []
        for day, high, low in zip(data["dates"], data["max_temperature"], data["min_temperature"]):
            rows.append(f'<div class="weather-row">{clean(day)} <strong>{clean(high)}°C</strong>'
                        f'<span class="result-meta">Low {clean(low)}°C</span></div>')
        st.markdown('<div class="result-list">' + "".join(rows) + '</div>', unsafe_allow_html=True)
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


def message_list(messages):
    visible = False
    tool_names = {}
    results = []
    action_records = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_names.update({call["id"]: call["name"] for call in message.tool_calls})
            continue
        if isinstance(message, ToolMessage):
            kind = tool_names.get(message.tool_call_id)
            data = parse_payload(message.content)
            if kind and data is not None:
                results.append((kind, data))
            if kind in {"send_email", "create_calendar_event"}:
                if "cancelled" in str(message.content).lower():
                    action_records.append("Email cancelled" if kind == "send_email" else "Event cancelled")
                elif kind == "send_email":
                    action_records.append("Email sent" if isinstance(data, dict) and data.get("status") == "email_sent"
                                          else "Email request finished")
                else:
                    action_records.append("Event created" if isinstance(data, dict) and data.get("id")
                                          else "Event request finished")
            continue
        if not isinstance(message, (HumanMessage, AIMessage)) or not message.content:
            continue
        visible = True
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        logo = ROOT / "assets" / ("logo_dark.svg" if st.session_state.get("dark_mode") else "logo.svg")
        with st.chat_message(role, avatar=str(logo) if role == "assistant" else None):
            rendered_result = False
            if role == "assistant":
                for kind, data in results:
                    rendered_result = render_structured(kind, data) or rendered_result
                results.clear()
            if message.content and not rendered_result:
                render_content(message.content, remove_emoji=role == "assistant")
            if role == "assistant":
                for record in action_records:
                    st.markdown(f'<div class="action-result">✓ {safe(record)}</div>', unsafe_allow_html=True)
                action_records.clear()
            if message.additional_kwargs.get("aether_time"):
                st.markdown(f'<div class="message-time">{safe(message.additional_kwargs["aether_time"])}</div>',
                            unsafe_allow_html=True)
            if role == "assistant":
                copy_control(strip_emojis(str(message.content)))
    for record in action_records:
        st.markdown(f'<div class="action-result">✓ {safe(record)}</div>', unsafe_allow_html=True)
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
        with st.container(border=True, key=f"approval-{action_id}"):
            st.markdown('<div class="approval-heading">' + ("Send email" if kind == "send_email" else "Create calendar event")
                        + '</div>', unsafe_allow_html=True)
            if kind == "send_email":
                st.write(f"To: {args.get('to', '')}")
                st.write(f"Subject: {args.get('subject', '')}")
                st.write(args.get("body", ""))
            else:
                st.write(f"Title: {args.get('title', '')}")
                st.write(f"Start: {args.get('start_time', '')}")
                st.write(f"End: {args.get('end_time', '')}")
                try:
                    duration = datetime.fromisoformat(args["end_time"]) - datetime.fromisoformat(args["start_time"])
                    st.write(f"Duration: {duration}")
                except (KeyError, ValueError):
                    pass
                if args.get("location"):
                    st.write(f"Location: {args['location']}")
            if action_id in decisions:
                st.caption("Approved" if decisions[action_id] else "Cancelled")
            else:
                confirm, cancel = st.columns(2)
                if confirm.button("Confirm", key=f"approve-{action_id}", type="primary"):
                    decisions[action_id] = True
                    st.rerun()
                if cancel.button("Cancel", key=f"reject-{action_id}"):
                    decisions[action_id] = False
                    st.rerun()
    if all(action["id"] in decisions for action in actions):
        activity = []
        with st.status("Completing request...", expanded=True) as status:
            try:
                asyncio.run(invoke_agent(st.session_state.thread_id, approvals=decisions,
                                         progress=lambda name: update_activity(status, name, activity)))
            except Exception as error:
                status.update(state="error", expanded=False)
                st.error(f"Could not complete the request: {error}")
                return
            status.update(label="Request finished", state="complete", expanded=False)
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
        with st.status("Aether is working...", expanded=True) as status:
            try:
                result = asyncio.run(invoke_agent(thread_id, message=prompt,
                                                  progress=lambda name: update_activity(status, name, activity)))
            except Exception as error:
                status.update(state="error", expanded=False)
                st.error(f"Could not reach Aether services: {error}")
                return
            if isinstance(result, dict) and "__interrupt__" in result:
                status.update(label="Awaiting confirmation", state="complete", expanded=False)
                st.session_state.setdefault("activity_summaries", {}).pop(thread_id, None)
            else:
                summary = " · ".join(activity) if activity else "Response ready"
                status.update(label=f"Completed · {summary}", state="complete", expanded=False)
                st.session_state.setdefault("activity_summaries", {})[thread_id] = f"✓ {summary}"
    st.rerun()


def update_activity(status, tool_name, activity):
    if "email" in tool_name or "contact" in tool_name or "draft" in tool_name:
        service = "Gmail"
    elif "calendar" in tool_name or "event" in tool_name:
        service = "Calendar"
    elif "weather" in tool_name or "forecast" in tool_name:
        service = "Weather"
    else:
        return
    if service not in activity:
        activity.append(service)
        icon = {"Gmail": "mail", "Calendar": "calendar_month", "Weather": "partly_cloudy_day"}[service]
        detail = ("sending email" if tool_name == "send_email" else
                  "drafting email" if "draft" in tool_name else
                  "searching inbox" if service == "Gmail" else
                  "creating event" if tool_name == "create_calendar_event" else
                  "checking events" if service == "Calendar" else "checking conditions")
        status.markdown(f'<div class="tool-activity"><span class="service-tile {service.lower()}-tile '
                        f'material-symbols-outlined">{icon}</span>{service}: {detail}</div>', unsafe_allow_html=True)
    status.update(label=f"Working with {service}...")


def main():
    setup_page()
    manager = ConversationManager(ROOT / "conversations.json")
    conversations = manager.list()
    if st.session_state.get("active_conversation") not in conversations:
        st.session_state.active_conversation = next(iter(conversations), None)
    snapshots = {name: asyncio.run(conversation_state(thread_id))
                 for name, thread_id in conversations.items()}
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
        if st.session_state.get("activity_summaries", {}).get(thread_id):
            st.markdown(f'<div class="activity-summary">{safe(st.session_state.activity_summaries[thread_id])}</div>',
                        unsafe_allow_html=True)
        activity_slot = st.empty()
        if actions:
            confirmation_card(actions)
        elif not has_messages:
            suggestion = empty_state()
            if suggestion:
                send_message(thread_id, suggestion, first_turn=True, activity_slot=activity_slot)
    prompt = st.chat_input("Message Aether", disabled=bool(actions))
    if prompt:
        first_turn = not any(isinstance(message, HumanMessage) for message in messages)
        send_message(thread_id, prompt, first_turn=first_turn, activity_slot=activity_slot)


if __name__ == "__main__":
    main()
