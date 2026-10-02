import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage, ToolMessage

from src.assistant.conversations import ConversationManager
from app import event_parts, local_datetime, service_for_tool


class AppTests(unittest.TestCase):
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
                return SimpleNamespace(values={}, tasks=[], created_at=dates[thread_id].isoformat())

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

            async def state(thread_id):
                return SimpleNamespace(values={}, tasks=[])

            async def invoke(thread_id, message=None, approvals=None, progress=None):
                calls.append((thread_id, message, approvals))
                if progress:
                    progress("get_recent_emails")

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
                new_thread = manager.get("New conversation")
                self.assertEqual(app.session_state["active_conversation"], "New conversation")
                app.button(key="suggest-Check my unread emails").click().run(timeout=15)
                self.assertEqual(calls, [(new_thread, "Check my unread emails", None)])
                self.assertEqual(manager.get("Check my unread emails"), new_thread)
                self.assertEqual(app.session_state["active_conversation"], "Check my unread emails")

                app.text_input[-1].set_value("Renamed").run(timeout=15)
                app.button(key=f"FormSubmitter:rename-{new_thread}-Save").click().run(timeout=15)
                self.assertEqual(manager.get("Renamed"), new_thread)
                app.button(key=f"delete-{new_thread}").click().run(timeout=15)
                app.button(key=f"confirm-delete-{new_thread}").click().run(timeout=15)
                self.assertEqual(manager.list(), {"Existing": existing})
                self.assertFalse(app.exception)
                app.toggle[0].set_value(True).run(timeout=15)
                self.assertTrue(app.session_state["dark_mode"])
                self.assertTrue(any("aether-dark-marker" in item.value for item in app.markdown))

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
                return SimpleNamespace(values={}, tasks=tasks)

            async def invoke(selected_thread, message=None, approvals=None, progress=None):
                calls.append((selected_thread, approvals))
                pending.clear()

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state), \
                 patch("src.assistant.agent.invoke_agent", invoke):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
                self.assertFalse(app.exception)
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
