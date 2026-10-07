# UI behavior tests use Streamlit's AppTest runner with isolated conversation registries.
import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.assistant.conversations import ConversationManager
from app import (checkpoint_pending_turn, clean_conversation_title, event_parts,
                 bind_delete_target, clear_delete_target, delete_bound_conversation,
                 generate_conversation_title, is_default_name, is_substantive_message,
                 local_datetime, pending_in_history, phase_label, service_for_tool,
                 should_auto_title, working_label, THINKING_LABELS, TURN_TIMEOUT_SECONDS,
                 run_pending_turn)


class AppTests(unittest.TestCase):
    def setUp(self):
        title_model = patch("src.assistant.agent.create_chat_model", side_effect=RuntimeError("offline test"))
        title_model.start()
        self.addCleanup(title_model.stop)

    def test_delete_target_requires_explicit_confirmation(self):
        class Manager:
            def __init__(self):
                self.deleted = []

            def delete(self, name):
                self.deleted.append(name)

            def list(self):
                return {"Other": "thread-2"}

        manager = Manager()
        state = {"active_conversation": "Other"}
        bind_delete_target(state, "Clicked row")
        self.assertEqual(state["delete_target"], "Clicked row")
        self.assertFalse(delete_bound_conversation(manager, state, confirmed=False))
        self.assertEqual(manager.deleted, [])
        self.assertIsNone(state["delete_target"])

        bind_delete_target(state, "Clicked row")
        clear_delete_target(state)
        self.assertEqual(manager.deleted, [])
        self.assertIsNone(state["delete_target"])

        bind_delete_target(state, "Clicked row")
        self.assertTrue(delete_bound_conversation(manager, state, confirmed=True))
        self.assertEqual(manager.deleted, ["Clicked row"])
        self.assertEqual(state["active_conversation"], "Other")

    def test_conversation_title_cleaner(self):
        cases = {
            "i have a cricket game tomorrow, 6AM to 8Am ,set it up in calendars": "I have a cricket game tomorrow",
            "check my calendar and , at 5pm": "Check my calendar at 5pm",
            "give me last 3 mails": "Last 3 mails",
            "can you check weather in islamabad please": "Check weather in islamabad",
            "Draft an email": "Draft an email",
            "send email": "Send an email",
            "send an email to my friend": "Send an email to my friend",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(clean_conversation_title(message), expected)

    def test_default_name_greeting_deferral_and_manual_name(self):
        self.assertTrue(is_default_name("New conversation"))
        self.assertTrue(is_default_name("New conversation 4"))
        self.assertFalse(is_default_name("Project plan"))
        for greeting in ("hi", "hello", "hey aether"):
            self.assertFalse(is_substantive_message(greeting))
            self.assertFalse(should_auto_title("New conversation", greeting))
        self.assertTrue(should_auto_title("New conversation", "send email"))
        self.assertTrue(should_auto_title("New conversation 2", "check weather in Lahore"))
        self.assertFalse(should_auto_title("My Lahore trip", "check weather in Lahore"))

    def test_generated_title_failure_and_timeout_leave_fallback_available(self):
        fallback = clean_conversation_title("check weather in Islamabad please")

        class FailingModel:
            async def ainvoke(self, messages):
                raise RuntimeError("offline")

        class SlowModel:
            async def ainvoke(self, messages):
                await asyncio.sleep(.05)
                return AIMessage(content="Islamabad weather")

        for factory, timeout in ((FailingModel, 5), (SlowModel, .001)):
            with self.subTest(factory=factory.__name__):
                with self.assertRaises((RuntimeError, TimeoutError)):
                    asyncio.run(generate_conversation_title(
                        "check weather in Islamabad please", factory, timeout=timeout))
                self.assertEqual(fallback, "Check weather in Islamabad")

    def test_progress_labels_use_real_phase_and_tool_arguments(self):
        self.assertEqual(phase_label({"phase": "thinking"}), "Thinking...")
        self.assertEqual(phase_label({"phase": "writing"}), "Writing the reply...")
        self.assertEqual(phase_label({"phase": "waiting"}), "Waiting for your confirmation")
        self.assertEqual(phase_label({"phase": "tool", "tool": "get_recent_emails", "args": {}}),
                         "Checking Gmail...")
        self.assertEqual(phase_label({"phase": "tool", "tool": "get_upcoming_events", "args": {}}),
                         "Checking calendar...")
        self.assertEqual(phase_label({"phase": "tool", "tool": "get_weather",
                                      "args": {"city": "Lahore"}}),
                         "Getting weather for Lahore...")
        self.assertIsNone(phase_label({"phase": "tool", "tool": "unknown", "args": {}}))

    def test_thinking_timeline_and_tool_override(self):
        self.assertEqual(TURN_TIMEOUT_SECONDS, 30)
        for elapsed, expected in ((0, THINKING_LABELS[0]), (2, THINKING_LABELS[0]),
                                  (3, THINKING_LABELS[1]), (6, THINKING_LABELS[2]),
                                  (18, THINKING_LABELS[-1]), (29, THINKING_LABELS[-1])):
            self.assertEqual(working_label("Thinking...", elapsed), expected)
            self.assertEqual(working_label("Writing the reply...", elapsed), expected)
        self.assertEqual(working_label("Checking Gmail...", 12), "Checking Gmail...")
        self.assertEqual(working_label("Waiting for your confirmation", 90),
                         "Waiting for your confirmation")

    def test_timed_out_turn_keeps_prompt_and_releases_input(self):
        pending = {"prompt": "Check my inbox", "turn_id": "same-turn"}
        state = SimpleNamespace(pending_turn=pending, run_active=True, run_error=None)
        cancelled = []

        async def stalled(thread_id, message, progress, turn_id):
            self.assertEqual((thread_id, message, turn_id),
                             ("thread-1", "Check my inbox", "same-turn"))
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(True)

        with patch("app.st.session_state", state), patch("app.st.rerun"), \
             patch("app.invoke_agent", stalled), patch("app.TURN_TIMEOUT_SECONDS", .01):
            run_pending_turn("thread-1", Mock())
        self.assertIs(state.pending_turn, pending)
        self.assertFalse(state.run_active)
        self.assertEqual(state.run_error, "The request timed out after 30 seconds.")
        self.assertEqual(cancelled, [True])

    def test_pending_turn_is_not_rendered_twice_after_checkpoint_save(self):
        pending = {"prompt": "hello", "turn_id": "turn-1"}
        messages = [HumanMessage(content="hello", additional_kwargs={"aether_turn_id": "turn-1"})]
        self.assertTrue(pending_in_history(messages, pending))
        self.assertFalse(pending_in_history([HumanMessage(content="hello")], pending))

        snapshot = SimpleNamespace(next=("aether",), values={"messages": messages})
        recovered = checkpoint_pending_turn(snapshot, [])
        self.assertEqual(recovered["prompt"], "hello")
        self.assertEqual(recovered["turn_id"], "turn-1")
        self.assertTrue(recovered["resume"])
        self.assertTrue(pending_in_history(messages, recovered))
        self.assertIsNone(checkpoint_pending_turn(snapshot, [{"id": "approval"}]))

    def test_display_dates_and_tool_labels(self):
        self.assertEqual(local_datetime("Fri, 2 Oct 2026 02:08:04 +0000 (UTC)").strftime("%I:%M %p"), "07:08 AM")
        self.assertEqual(event_parts("2026-10-03T01:00:00Z", "2026-10-03T03:00:00Z"),
                         ("Sat", "3", "Oct", "6:00 AM to 8:00 AM"))
        self.assertEqual(service_for_tool("get_recent_emails"), "Gmail")
        self.assertEqual(service_for_tool("get_weather"), "Weather")
        self.assertIsNone(service_for_tool(None))

    def test_sidebar_groups_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            threads = {name: manager.create(name) for name in ("Today chat", "Yesterday chat", "Earlier chat")}
            now = datetime.now(ZoneInfo("Asia/Karachi"))
            dates = {threads["Today chat"]: now, threads["Yesterday chat"]: now - timedelta(days=1),
                     threads["Earlier chat"]: now - timedelta(days=4)}

            async def state(thread_id):
                return SimpleNamespace(values={"messages": [HumanMessage(content="hi")]}, tasks=[], created_at=dates[thread_id].isoformat())

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
                labels = " ".join(item.value for item in app.markdown)
                self.assertIn('conversation-group">Today', labels)
                self.assertIn('conversation-group">Yesterday', labels)
                self.assertIn('conversation-group">Earlier', labels)
                app.text_input(key="conversation_search").set_value("Yesterday").run(timeout=15)
                self.assertEqual([button.label for button in app.button if button.key and button.key.startswith("select-")],
                                 ["Yesterday chat"])
                self.assertFalse(app.exception)

    def test_structured_results_render_without_raw_json(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            manager.create("Results")

            async def state(thread_id):
                return SimpleNamespace(values={"messages": [
                    AIMessage(content='{"city":"Rawalpindi","temperature":26} 😀'),
                    AIMessage(content='[{"title":"Meeting","start":"2026-10-01T10:00:00+05:00"}]'),
                    AIMessage(content='[{"from":"Team","subject":"Update","date":"Today"}]'),
                    AIMessage(content="", tool_calls=[{"id": "mail-1", "name": "get_recent_emails", "args": {}}]),
                    ToolMessage(content=str([{"type": "text", "text": '{"from":"Alice","subject":"Hello","date":"Today"}'}]),
                                tool_call_id="mail-1"),
                    AIMessage(content="Here is your mail 😊"),
                    AIMessage(content="", tool_calls=[{"id": "weather-1", "name": "get_weather", "args": {}}]),
                    ToolMessage(content=str([{"type": "text", "text": '{"city":"Lahore","temperature":30,"humidity":50}'}]),
                                tool_call_id="weather-1"),
                    AIMessage(content="Here is the weather"),
                    AIMessage(content="", tool_calls=[{"id": "send-1", "name": "send_email", "args": {}}]),
                    ToolMessage(content='{"status":"email_sent"}', tool_call_id="send-1"),
                    AIMessage(content="Done"),
                    AIMessage(content="", tool_calls=[{"id": "event-1", "name": "create_calendar_event",
                                                       "args": {"title": "Test"}}]),
                    ToolMessage(content="The user cancelled this action. Do not perform it.", tool_call_id="event-1"),
                    AIMessage(content="Cancelled."),
                ]}, tasks=[])

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
                self.assertFalse(app.exception)
                app.button(key=f"select-{manager.get('Results')}").click().run(timeout=15)
                output = " ".join(item.value for item in app.markdown)
                self.assertIn("<strong>26°C</strong>", output)
                self.assertIn("Meeting", output)
                self.assertIn("Update", output)
                self.assertIn("Alice", output)
                self.assertIn("Lahore", output)
                self.assertIn("Email sent", output)
                self.assertIn("Used Gmail", output)
                self.assertIn("Used Weather", output)
                self.assertEqual(output.count("Cancelled"), 1)
                self.assertNotIn("Used Calendar", output)
                self.assertNotIn("Response ready", output)
                self.assertNotIn("Here is the weather", output)
                self.assertNotIn("Here is your mail", output)
                self.assertNotIn('{"city"', output)
                self.assertNotIn("😊", output)

    def test_sidebar_and_suggestion_use_stored_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            existing = manager.create("Existing")
            calls = []
            history = {existing: [HumanMessage(content="hi"), AIMessage(content="Hello")]}

            async def state(thread_id):
                return SimpleNamespace(values={"messages": history.get(thread_id, [])}, tasks=[])

            async def invoke(thread_id, message=None, approvals=None, progress=None, turn_id=None):
                calls.append((thread_id, message, approvals))
                history.setdefault(thread_id, []).extend([HumanMessage(content=message), AIMessage(content="Done")])
                if progress:
                    progress({"phase": "tool", "tool": "get_recent_emails", "args": {}})

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state), \
                 patch("src.assistant.agent.invoke_agent", invoke):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
                self.assertFalse(app.exception)
                app.button(key="starter-Email").click().run(timeout=15)
                self.assertEqual(calls, [])
                app.button(key=f"select-{existing}").click().run(timeout=15)
                self.assertEqual(app.session_state["active_conversation"], "Existing")

                next(button for button in app.button if button.label == "New conversation").click().run(timeout=15)
                for _ in range(4):
                    next(button for button in app.button if button.label == "New conversation").click().run(timeout=15)
                self.assertIsNone(app.session_state["active_conversation"])
                self.assertEqual(manager.list(), {"Existing": existing})
                app.button(key="suggest-Check my unread emails").click().run(timeout=15)
                app.run(timeout=15)
                new_thread = manager.get("Check my unread emails")
                self.assertEqual(len(manager.list()), 2)
                self.assertEqual(calls, [(new_thread, "Check my unread emails", None)])
                self.assertEqual(manager.get("Check my unread emails"), new_thread)
                self.assertEqual(app.session_state["active_conversation"], "Check my unread emails")

                app.text_input[-1].set_value("Renamed").run(timeout=15)
                app.button(key=f"FormSubmitter:rename-{new_thread}-Save").click().run(timeout=15)
                self.assertEqual(manager.get("Renamed"), new_thread)
                app.button(key=f"delete-{new_thread}").click().run(timeout=15)
                app.button(key=f"confirm-dialog-delete-{new_thread}").click().run(timeout=15)
                self.assertEqual(manager.list(), {"Existing": existing})
                self.assertFalse(app.exception)
                self.assertEqual(len(app.radio), 0)
                self.assertNotIn("dark_mode", app.session_state)

    def test_draft_first_send_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            empty = manager.create("New conversation")
            existing = manager.create("Existing")
            history = {existing: [HumanMessage(content="hi")]}
            calls = []

            async def state(thread_id):
                return SimpleNamespace(values={"messages": history.get(thread_id, [])}, tasks=[])

            async def invoke(thread_id, message=None, approvals=None, progress=None, turn_id=None):
                calls.append(thread_id)
                history.setdefault(thread_id, []).extend([HumanMessage(content=message), AIMessage(content="Done")])

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state), \
                 patch("src.assistant.agent.invoke_agent", invoke):
                path = str(Path(__file__).parents[1] / "app.py")
                app = AppTest.from_file(path).run(timeout=15)
                self.assertIsNone(app.session_state["active_conversation"])
                self.assertNotIn(f"select-{empty}", [button.key for button in app.button])
                self.assertEqual(len(manager.list()), 2)
                app.chat_input[0].set_value("Check the weather in Lahore").run(timeout=15)
                app.run(timeout=15)
                self.assertFalse(app.exception)
                self.assertEqual(len(manager.list()), 3)
                thread_id = calls[0]
                self.assertEqual(app.query_params["conversation"], [thread_id])
                restored = AppTest.from_file(path)
                restored.query_params["conversation"] = thread_id
                restored.run(timeout=15)
                self.assertEqual(restored.session_state["thread_id"], thread_id)
                restored.chat_input[0].set_value("hi").run(timeout=15)
                restored.run(timeout=15)
                self.assertEqual(calls, [thread_id, thread_id])
                self.assertEqual(len(manager.list()), 3)
                draft = AppTest.from_file(path).run(timeout=15)
                self.assertIsNone(draft.session_state["active_conversation"])
                self.assertNotIn("conversation", draft.query_params)

    def test_confirmation_uses_saved_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            thread_id = manager.create("Approval")
            action = {"id": "send-1", "name": "send_email",
                      "args": {"to": "test@example.com", "subject": "Hi", "body": "Hello"}}
            pending = [action]
            calls = []

            async def state(selected_thread):
                tasks = [SimpleNamespace(interrupts=[SimpleNamespace(value=pending)])] if pending else []
                return SimpleNamespace(values={"messages": [HumanMessage(content="hi")]}, tasks=tasks)

            async def invoke(selected_thread, message=None, approvals=None, progress=None, turn_id=None):
                calls.append((selected_thread, approvals))
                pending.clear()

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state), \
                 patch("src.assistant.agent.invoke_agent", invoke):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
                self.assertFalse(app.exception)
                app.button(key=f"select-{thread_id}").click().run(timeout=15)
                app.button(key="reject-send-1").click().run(timeout=15)
                self.assertEqual(calls, [(thread_id, {"send-1": False})])
                self.assertFalse(app.exception)
                pending.append({"id": "event-1", "name": "create_calendar_event",
                                "args": {"title": "Test", "start_time": "2026-10-01T10:00:00+05:00",
                                         "end_time": "2026-10-01T12:00:00+05:00"}})
                app.run(timeout=15)
                app.button(key="approve-event-1").click().run(timeout=15)
                self.assertEqual(calls[-1], (thread_id, {"event-1": True}))
                self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
