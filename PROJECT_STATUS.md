# Aether AI — Project Status

## Architecture

Streamlit UI → conversation registry → LangGraph agent → Groq → MCP client → Gmail, Calendar, Weather servers.

`conversations.json` maps names to thread IDs. `checkpoints.db` stores LangGraph state by thread ID. Both are local data files and are excluded from Git.

## Completed

* [x] Gmail search, reading, contacts, drafts, sending, and event extraction
* [x] Calendar retrieval and creation with Asia/Karachi time handling
* [x] Current weather and forecasts
* [x] LangGraph orchestration and MCP tool calling
* [x] Confirmation before email sending and calendar creation in terminal and Streamlit
* [x] SQLite checkpoint persistence across restarts
* [x] Named conversation creation, listing, selection, and resume in the terminal
* [x] Duplicate name protection, rename, and safe registry deletion
* [x] Python test for persistence, isolation, rename, duplicates, and deletion
* [x] Streamlit sidebar, chat, empty-state suggestions, activity status, and confirmation cards
* [x] Streamlit interactions tested with its test harness
* [x] Read-only end-to-end Groq and MCP checks for Gmail, Calendar, and Weather

## Remaining

* [ ] Visual browser review at desktop and narrow widths
* [ ] Live Gmail sending and Calendar creation checks (write actions intentionally not exercised)

Checkpoint rows are retained when a conversation name is renamed or deleted. Deletion removes only the name mapping.
