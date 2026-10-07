"""The pruning dry run reads history without changing registry or checkpoints."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from scripts.prune_empty_conversations import empty_conversations


class PruneEmptyConversationsTests(unittest.TestCase):
    def test_empty_history_and_pending_writes_without_database_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.db"
            serializer = JsonPlusSerializer()
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
                                   "checkpoint_id TEXT, type TEXT, checkpoint BLOB)")
                connection.execute("CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, "
                                   "checkpoint_id TEXT, channel TEXT, type TEXT, value BLOB)")
                for thread_id, messages in (("empty", []), ("saved", [HumanMessage(content="hi")]),
                                            ("pending", [])):
                    kind, content = serializer.dumps_typed({"channel_values": {"messages": messages}})
                    connection.execute("INSERT INTO checkpoints VALUES (?, '', '1', ?, ?)",
                                       (thread_id, kind, content))
                kind, content = serializer.dumps_typed([HumanMessage(content="pending")])
                connection.execute("INSERT INTO writes VALUES ('pending', '', '1', 'messages', ?, ?)",
                                   (kind, content))
            before = database.read_bytes()
            self.assertEqual(empty_conversations({"Empty": "empty", "Missing": "missing",
                                                  "Saved": "saved", "Pending": "pending"}, database),
                             ["Empty", "Missing"])
            self.assertEqual(database.read_bytes(), before)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.db"
            self.assertEqual(empty_conversations({"Empty": "thread"}, database), ["Empty"])
            self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
