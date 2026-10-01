"""Names for LangGraph threads. Checkpoint data stays in SQLite."""

import json
import os
import tempfile
import uuid
from pathlib import Path


class ConversationManager:
    def __init__(self, path="conversations.json"):
        self.path = Path(path)

    def list(self):
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as file:
            conversations = json.load(file)
        if not isinstance(conversations, dict) or not all(
            isinstance(name, str) and isinstance(thread_id, str)
            for name, thread_id in conversations.items()
        ):
            raise ValueError("Invalid conversation registry")
        return conversations

    def get(self, name):
        return self.list()[name]

    def create(self, name):
        name = self._name(name)
        conversations = self.list()
        if name in conversations:
            raise ValueError("A conversation with this name already exists")
        thread_id = str(uuid.uuid4())
        conversations[name] = thread_id
        self._save(conversations)
        return thread_id

    def rename(self, old_name, new_name):
        new_name = self._name(new_name)
        conversations = self.list()
        if old_name not in conversations:
            raise KeyError(old_name)
        if new_name != old_name and new_name in conversations:
            raise ValueError("A conversation with this name already exists")
        if new_name != old_name:
            items = [(new_name if name == old_name else name, thread_id)
                     for name, thread_id in conversations.items()]
            self._save(dict(items))

    def delete(self, name):
        conversations = self.list()
        if name not in conversations:
            raise KeyError(name)
        del conversations[name]
        self._save(conversations)
        # Keep checkpoints: deleting a name must never erase saved history.

    @staticmethod
    def _name(name):
        name = name.strip()
        if not name:
            raise ValueError("Conversation name cannot be empty")
        return name

    def _save(self, conversations):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".conversations-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(conversations, file, indent=4)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
