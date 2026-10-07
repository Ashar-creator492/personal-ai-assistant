"""List empty conversation names; remove registry entries only with --apply."""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from src.assistant.conversations import ConversationManager


def empty_conversations(conversations, checkpoints):
    """Read checkpoints without creating, updating, or deleting database data."""
    if not checkpoints.exists():
        return list(conversations)
    serializer = JsonPlusSerializer()
    empty = []
    with sqlite3.connect(checkpoints.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for name, thread_id in conversations.items():
            row = connection.execute(
                "SELECT checkpoint_id, type, checkpoint FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = '' ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
            if row is None:
                empty.append(name)
                continue
            checkpoint_id, kind, content = row
            checkpoint = serializer.loads_typed((kind, content))
            if checkpoint.get("channel_values", {}).get("messages"):
                continue
            # A message may already be saved as a pending graph write.
            writes = connection.execute(
                "SELECT type, value FROM writes WHERE thread_id = ? "
                "AND checkpoint_ns = '' AND checkpoint_id = ? AND channel = 'messages'",
                (thread_id, checkpoint_id),
            ).fetchall()
            if not any(serializer.loads_typed(write) for write in writes):
                empty.append(name)
    return empty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Remove only empty registry entries")
    parser.add_argument("--registry", type=Path, default=ROOT / "conversations.json")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints.db")
    args = parser.parse_args()
    manager = ConversationManager(args.registry)
    names = empty_conversations(manager.list(), args.checkpoints)
    print(f'{"Apply" if args.apply else "Dry run"}: {len(names)} empty conversation(s)')
    for name in names:
        print(f"- {name}")
        if args.apply:
            manager.delete(name)
    if not args.apply:
        print("No files changed. Use --apply to remove these registry entries; checkpoints are never modified.")


if __name__ == "__main__":
    main()
