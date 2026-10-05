# Persistence tests keep the conversation registry aligned with LangGraph checkpoints.
import asyncio
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.assistant.agent import AgentState
from src.assistant.conversations import ConversationManager


class ConversationTests(unittest.TestCase):
    def test_registry_and_checkpoint_persistence(self):
        asyncio.run(self._exercise())

    async def _exercise(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationManager(Path(directory) / "conversations.json")
            database = str(Path(directory) / "checkpoints.db")
            first = manager.create("First")
            second = manager.create("Second")
            with self.assertRaises(ValueError):
                manager.create("First")
            with self.assertRaises(ValueError):
                manager.rename("Second", "First")

            async def reply(state):
                human_messages = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
                return {"messages": [AIMessage(content=" | ".join(human_messages))]}

            graph = StateGraph(AgentState)
            graph.add_node("reply", reply)
            graph.add_edge(START, "reply")
            graph.add_edge("reply", END)

            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                app = graph.compile(checkpointer=saver)
                await app.ainvoke({"messages": [HumanMessage(content="hello")]},
                                  {"configurable": {"thread_id": first}})
                await app.ainvoke({"messages": [HumanMessage(content="other")]},
                                  {"configurable": {"thread_id": second}})

            manager.rename("First", "Renamed")
            self.assertEqual(manager.get("Renamed"), first)
            self.assertEqual(list(manager.list()), ["Renamed", "Second"])

            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                app = graph.compile(checkpointer=saver)
                result = await app.ainvoke({"messages": [HumanMessage(content="again")]},
                                           {"configurable": {"thread_id": manager.get("Renamed")}})
                self.assertEqual(result["messages"][-1].content, "hello | again")
                other = await app.aget_state({"configurable": {"thread_id": second}})
                self.assertEqual(other.values["messages"][-1].content, "other")

            manager.delete("Renamed")
            self.assertEqual(manager.list(), {"Second": second})
            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                app = graph.compile(checkpointer=saver)
                preserved = await app.aget_state({"configurable": {"thread_id": first}})
                self.assertEqual(preserved.values["messages"][-1].content, "hello | again")


if __name__ == "__main__":
    unittest.main()
