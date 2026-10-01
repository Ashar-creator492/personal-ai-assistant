# Aether AI — Project Status

## Architecture

Streamlit UI (planned) → conversation registry → LangGraph agent → Groq → MCP client → Gmail, Calendar, Weather servers.

`conversations.json` maps names to thread IDs. `checkpoints.db` stores LangGraph state by thread ID. Both are local data files and are excluded from Git.

## Completed

* [x] Gmail search, reading, contacts, drafts, sending, and event extraction
* [x] Calendar retrieval and creation with Asia/Karachi time handling
* [x] Current weather and forecasts
* [x] LangGraph orchestration and MCP tool calling
* [x] Confirmation before email sending and calendar creation in the terminal
* [x] SQLite checkpoint persistence across restarts
* [x] Named conversation creation, listing, selection, and resume in the terminal
* [x] Duplicate name protection, rename, and safe registry deletion
* [x] Python test for persistence, isolation, rename, duplicates, and deletion

## Remaining

* [ ] Streamlit frontend and conversation sidebar
* [ ] Streamlit confirmation interaction for email and calendar writes
* [ ] Rendered Gmail, Calendar, and Weather results
* [ ] Visual and integration verification of the frontend

Checkpoint rows are retained when a conversation name is renamed or deleted. Deletion removes only the name mapping.
