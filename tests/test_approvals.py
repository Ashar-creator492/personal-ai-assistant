import asyncio
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from src.assistant.agent import build_graph


class FakeLLM:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    async def ainvoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=messages[-1].content)
        return AIMessage(content="", tool_calls=[{
            "id": "write-1", "name": self.name, "args": self.args,
        }])


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
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
            async with AsyncSqliteSaver.from_conn_string(str(Path(directory) / "state.db")) as saver:
                app = build_graph(FakeLLM(name, args), [tool], saver)
                denied_config = {"configurable": {"thread_id": "denied"}}
                result = await app.ainvoke({"messages": [HumanMessage(content="Send it")]}, denied_config)
                self.assertIn("__interrupt__", result)
                snapshot = await app.aget_state(denied_config)
                self.assertEqual(snapshot.tasks[0].interrupts[0].value[0]["id"], "write-1")
                self.assertEqual(tool.calls, 0)
                self.assertEqual(result["__interrupt__"][0].value[0]["name"], name)
                result = await app.ainvoke(Command(resume={"write-1": False}), denied_config)
                self.assertEqual(tool.calls, 0)
                self.assertIn("cancelled", result["messages"][-1].content)

                allowed_config = {"configurable": {"thread_id": "allowed"}}
                result = await app.ainvoke({"messages": [HumanMessage(content="Send it")]}, allowed_config)
                self.assertIn("__interrupt__", result)
                result = await app.ainvoke(Command(resume={"write-1": True}), allowed_config)
                self.assertEqual(tool.calls, 1)
                self.assertEqual(result["messages"][-1].content, "Email sent")


if __name__ == "__main__":
    unittest.main()
