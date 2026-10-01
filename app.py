"""Streamlit interface for the existing Aether LangGraph agent."""

import asyncio
import json
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.assistant.agent import conversation_state, invoke_agent
from src.assistant.conversations import ConversationManager

ROOT = Path(__file__).parent
SUGGESTIONS = ("Check my unread emails", "What's on my calendar this week?", "Weather in Rawalpindi")


def safe(value):
    return escape(str(value))


def setup_page():
    st.set_page_config(page_title="Aether", page_icon=str(ROOT / "assets/logo.svg"),
                       layout="wide", initial_sidebar_state="expanded")
    st.logo(str(ROOT / "assets/logo.svg"), size="small")
    st.markdown(f"<style>{(ROOT / 'styles.css').read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


def unique_name(conversations):
    base = "New conversation"
    number = 1
    while (base if number == 1 else f"{base} {number}") in conversations:
        number += 1
    return base if number == 1 else f"{base} {number}"


def sidebar(manager, conversations):
    with st.sidebar:
        st.markdown('<div class="wordmark">Aether</div>', unsafe_allow_html=True)
        if st.button("New conversation", use_container_width=True, type="secondary"):
            name = unique_name(conversations)
            manager.create(name)
            st.session_state.active_conversation = name
            st.rerun()
        st.markdown('<div class="sidebar-label">CONVERSATIONS</div>', unsafe_allow_html=True)
        for name, thread_id in conversations.items():
            active = name == st.session_state.active_conversation
            with st.container(key="active-conversation" if active else f"conversation-{thread_id}"):
                row, menu = st.columns([5, 1], gap="small", vertical_alignment="center")
                with row:
                    if st.button(name, key=f"select-{thread_id}", help=name, use_container_width=True):
                        st.session_state.active_conversation = name
                        st.rerun()
                with menu:
                    with st.popover("⋯", help=f"Options for {name}"):
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


def header(name):
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    with left:
        st.markdown(f'<h1 class="conversation-title">{safe(name)}</h1>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="service-label">Gmail · Calendar · Weather</div>',
                    unsafe_allow_html=True)
    st.divider()


def render_content(content):
    if not isinstance(content, str):
        content = str(content)
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        st.markdown(content)
        return
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        for item in data:
            if "subject" in item:
                st.markdown(f"**{safe(item.get('subject', 'Untitled'))}**  \n"
                            f"{safe(item.get('from', ''))} · {safe(item.get('date', ''))}")
            elif "title" in item:
                st.markdown(f"**{safe(item.get('title', 'Untitled'))}** · "
                            f"{safe(item.get('start', ''))}")
            else:
                st.markdown(" · ".join(safe(value) for value in item.values() if value is not None))
    elif isinstance(data, dict) and "temperature" in data:
        st.markdown(f"{safe(data.get('city', ''))} · {safe(data['temperature'])}°C")
    elif isinstance(data, dict) and "dates" in data:
        for day, high, low in zip(data["dates"], data["max_temperature"], data["min_temperature"]):
            st.markdown(f"{safe(day)} · {safe(high)}° / {safe(low)}°C")
    elif isinstance(data, dict) and "events" in data:
        for event in data["events"]:
            st.markdown(f"**{safe(event.get('title', 'Untitled'))}** · "
                        f"{safe(event.get('date', event.get('start', '')))}")
    else:
        st.markdown("  \n".join(f"**{safe(key.replace('_', ' ').title())}:** {safe(value)}"
                                 for key, value in data.items()) if isinstance(data, dict) else safe(data))


def message_list(messages):
    visible = [message for message in messages if isinstance(message, (HumanMessage, AIMessage))
               and message.content and not getattr(message, "tool_calls", None)]
    for message in visible:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            render_content(message.content)
            if message.additional_kwargs.get("aether_time"):
                st.markdown(f'<div class="message-time">{safe(message.additional_kwargs["aether_time"])}</div>',
                            unsafe_allow_html=True)
    return bool(visible)


def empty_state():
    st.markdown('<div class="empty-title">How can I help?</div>', unsafe_allow_html=True)
    st.markdown('<div class="empty-subtitle">Ask about your inbox, calendar, or the weather.</div>',
                unsafe_allow_html=True)
    for suggestion in SUGGESTIONS:
        if st.button(suggestion, key=f"suggest-{suggestion}", type="secondary"):
            return suggestion
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
            st.markdown("**Send email**" if kind == "send_email" else "**Create calendar event**")
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
        with st.status("Completing request...", expanded=False) as status:
            try:
                asyncio.run(invoke_agent(st.session_state.thread_id, approvals=decisions,
                                         progress=lambda name: update_activity(status, name)))
            except Exception as error:
                st.error(f"Could not complete the request: {error}")
                return
        st.session_state[decision_key] = {}
        st.rerun()


def send_message(thread_id, prompt):
    with st.status("Aether is working...", expanded=False) as status:
        try:
            asyncio.run(invoke_agent(thread_id, message=prompt,
                                     progress=lambda name: update_activity(status, name)))
        except Exception as error:
            st.error(f"Could not reach Aether services: {error}")
            return
    st.rerun()


def update_activity(status, tool_name):
    if "email" in tool_name or "contact" in tool_name or "draft" in tool_name:
        status.update(label="Checking Gmail...")
    elif "calendar" in tool_name or "event" in tool_name:
        status.update(label="Checking calendar...")
    elif "weather" in tool_name or "forecast" in tool_name:
        status.update(label="Getting weather...")


def main():
    setup_page()
    manager = ConversationManager(ROOT / "conversations.json")
    conversations = manager.list()
    if st.session_state.get("active_conversation") not in conversations:
        st.session_state.active_conversation = next(iter(conversations), None)
    sidebar(manager, conversations)
    if not st.session_state.active_conversation:
        st.markdown('<div class="empty-title">Aether</div>', unsafe_allow_html=True)
        st.write("Create a conversation to begin.")
        return
    name = st.session_state.active_conversation
    thread_id = conversations[name]
    st.session_state.thread_id = thread_id
    snapshot = asyncio.run(conversation_state(thread_id))
    actions = pending_actions(snapshot)
    with st.container(key="chat-content"):
        header(name)
        has_messages = message_list(snapshot.values.get("messages", []))
        if actions:
            confirmation_card(actions)
        elif not has_messages:
            suggestion = empty_state()
            if suggestion:
                send_message(thread_id, suggestion)
    prompt = st.chat_input("Message Aether", disabled=bool(actions))
    if prompt:
        send_message(thread_id, prompt)


if __name__ == "__main__":
    main()
