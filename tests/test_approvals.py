import asyncio
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from src.assistant.agent import build_graph


class FakeLLM:
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.last_messages = None

    async def ainvoke(self, messages):
        self.last_messages = messages
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=messages[-1].content)
        return AIMessage(content="", tool_calls=[{
            "id": "write-1", "name": self.name, "args": self.args,
        }])


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.args = []

    async def ainvoke(self, args):
        self.calls += 1
        self.args.append(args)
        return "Email sent"


class ApprovalTests(unittest.TestCase):
    def test_write_requires_explicit_approval(self):
        for name, args in (
            ("send_email", {"to": "test@example.com", "subject": "Test", "body": "Hello"}),
            ("create_calendar_event", {"title": "Test", "start_time": "2026-10-01T10:00:00+05:00",
                                       "end_time": "2026-10-01T12:00:00+05:00"}),
        ):
            with self.subTest(name=name):
                asyncio.run(self._exercise(name, args))

    async def _exercise(self, name, args):
        with tempfile.TemporaryDirectory() as directory:
            tool = FakeTool(name)
            progress = []
            async with AsyncSqliteSaver.from_conn_string(str(Path(directory) / "state.db")) as saver:
                app = build_graph(FakeLLM(name, args), [tool], saver, progress.append)
                denied_config = {"configurable": {"thread_id": "denied"}}
                result = await app.ainvoke({"messages": [HumanMessage(content="Send it")]}, denied_config)
                self.assertIn("__interrupt__", result)
                snapshot = await app.aget_state(denied_config)
                self.assertEqual(snapshot.tasks[0].interrupts[0].value[0]["id"], "write-1")
                self.assertEqual(tool.calls, 0)
                self.assertEqual(result["__interrupt__"][0].value[0]["name"], name)
                result = await app.ainvoke(Command(resume={"write-1": False}), denied_config)
                self.assertEqual(tool.calls, 0)
                self.assertEqual(progress, [])
                self.assertEqual(result["messages"][-1].content, "Cancelled.")

                allowed_config = {"configurable": {"thread_id": "allowed"}}
                result = await app.ainvoke({"messages": [HumanMessage(content="Send it")]}, allowed_config)
                self.assertIn("__interrupt__", result)
                result = await app.ainvoke(Command(resume={"write-1": True}), allowed_config)
                self.assertEqual(tool.calls, 1)
                self.assertEqual(progress, [name])
                self.assertEqual(result["messages"][-1].content, "Email sent")

    def test_calendar_approval_and_execution_keep_local_wall_time(self):
        asyncio.run(self._calendar_uses_local_time())

    async def _calendar_uses_local_time(self):
        args = {"title": "Morning event", "start_time": "2026-10-03T09:00:00-04:00",
                "end_time": "2026-10-03T09:15:00-04:00"}
        with tempfile.TemporaryDirectory() as directory:
            tool = FakeTool("create_calendar_event")
            llm = FakeLLM("create_calendar_event", args)
            async with AsyncSqliteSaver.from_conn_string(str(Path(directory) / "state.db")) as saver:
                app = build_graph(llm, [tool], saver)
                config = {"configurable": {"thread_id": "calendar"}}
                result = await app.ainvoke({"messages": [HumanMessage(content="Add it at 9 AM PKT")]}, config)
                self.assertIsInstance(llm.last_messages[0], SystemMessage)
                self.assertIn("Asia/Karachi", llm.last_messages[0].content)
                pending = result["__interrupt__"][0].value[0]
                self.assertEqual(pending["args"]["start_time"], "2026-10-03T09:00:00+05:00")
                self.assertEqual(pending["args"]["end_time"], "2026-10-03T09:15:00+05:00")
                await app.ainvoke(Command(resume={"write-1": True}), config)
                self.assertEqual(tool.args[0]["start_time"], pending["args"]["start_time"])
                self.assertEqual(tool.args[0]["end_time"], pending["args"]["end_time"])


if __name__ == "__main__":
    unittest.main()
