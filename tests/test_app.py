import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage

from src.assistant.conversations import ConversationManager


class AppTests(unittest.TestCase):
    def test_structured_results_render_without_raw_json(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            manager.create("Results")

            async def state(thread_id):
                return SimpleNamespace(values={"messages": [
                    AIMessage(content='{"city":"Rawalpindi","temperature":26}'),
                    AIMessage(content='[{"title":"Meeting","start":"2026-10-01T10:00:00+05:00"}]'),
                    AIMessage(content='[{"from":"Team","subject":"Update","date":"Today"}]'),
                ]}, tasks=[])

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
                self.assertFalse(app.exception)
                output = " ".join(item.value for item in app.markdown)
                self.assertIn("Rawalpindi · 26°C", output)
                self.assertIn("Meeting", output)
                self.assertIn("Update", output)
                self.assertNotIn('{"city"', output)

    def test_sidebar_and_suggestion_use_stored_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            existing = manager.create("Existing")
            calls = []

            async def state(thread_id):
                return SimpleNamespace(values={}, tasks=[])

            async def invoke(thread_id, message=None, approvals=None, progress=None):
                calls.append((thread_id, message, approvals))

            with patch("src.assistant.conversations.ConversationManager", return_value=manager), \
                 patch("src.assistant.agent.conversation_state", state), \
                 patch("src.assistant.agent.invoke_agent", invoke):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=15)
                self.assertFalse(app.exception)
                app.button(key=f"select-{existing}").click().run(timeout=15)
                self.assertEqual(app.session_state["active_conversation"], "Existing")

                app.button[3].click().run(timeout=15)
                new_thread = manager.get("New conversation")
                self.assertEqual(app.session_state["active_conversation"], "New conversation")
                app.button(key="suggest-Check my unread emails").click().run(timeout=15)
                self.assertEqual(calls, [(new_thread, "Check my unread emails", None)])

                app.text_input[-1].set_value("Renamed").run(timeout=15)
                app.button(key=f"FormSubmitter:rename-{new_thread}-Save").click().run(timeout=15)
                self.assertEqual(manager.get("Renamed"), new_thread)
                app.button(key=f"delete-{new_thread}").click().run(timeout=15)
                app.button(key=f"confirm-delete-{new_thread}").click().run(timeout=15)
                self.assertEqual(manager.list(), {"Existing": existing})
                self.assertFalse(app.exception)

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
