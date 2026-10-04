# Aether AI

**A personal AI assistant built with LangGraph, Groq, MCP, and Streamlit.**

Aether is an agentic AI assistant that can reason about a user's request, decide which external tools are needed, execute those tools through the Model Context Protocol (MCP), maintain persistent conversation memory, and safely handle actions such as sending emails and creating calendar events.

Instead of being a simple chatbot, Aether is designed as an **agent that can interact with real-world services**.

---

## ✨ Features

### 🤖 Agentic Reasoning

Aether uses **LangGraph** to manage the agent's execution flow.

It can:

- Understand the user's intent
- Decide when tools are required
- Select the appropriate MCP tool
- Process tool results
- Call additional tools when necessary
- Stop when enough information has been gathered

---

### 🔌 MCP Integration

Aether communicates with external services through the **Model Context Protocol (MCP)**.

Current MCP servers:

- 📧 Gmail
- 📅 Google Calendar
- 🌤️ Weather

This keeps external-service integrations separated from the core agent logic.

---

### 📧 Gmail

Aether can interact with Gmail for tasks such as:

- Reading recent emails
- Searching emails
- Opening specific emails
- Searching contacts
- Creating email drafts
- Sending emails

Email-sending actions require explicit confirmation before execution.

---

### 📅 Google Calendar

Aether can:

- Find upcoming events
- Extract upcoming commitments from relevant emails
- Create calendar events
- Specify event title, time, location, and description

Calendar-writing actions require explicit confirmation.

---

### 🌤️ Weather

Aether can retrieve:

- Current weather
- Weather forecasts

Weather information is provided through a dedicated MCP weather server.

---

### 🧠 Persistent Memory

Aether uses **LangGraph checkpoints with SQLite** to maintain conversation state.

This allows conversations to:

- Persist across application restarts
- Maintain their own independent state
- Resume from where they previously stopped

Different conversations use separate `thread_id`s, preventing their histories from being mixed together.

---

### 💬 Conversation Management

Aether supports multiple conversations.

Users can:

- Create conversations
- Switch between conversations
- Resume previous conversations
- Rename conversations
- Delete conversations
- Maintain separate conversation histories

Conversation metadata is maintained separately from LangGraph's checkpoint database.

---

### 🛡️ Action Confirmation

Aether does not blindly execute potentially consequential actions.

Before actions such as:

- Sending an email
- Creating a calendar event

the user is shown what will happen and must explicitly confirm it.

This provides an additional safety layer between the AI's reasoning and real-world actions.

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Streamlit     │
                         │       UI         │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │    Conversation Manager  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         ┌──────────────────┐
                         │    LangGraph     │
                         │      Agent       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Groq LLM     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    MCP Client    │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
       │    Gmail    │     │   Calendar  │     │   Weather   │
       │ MCP Server  │     │ MCP Server  │     │ MCP Server  │
       └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
              │                   │                   │
              ▼                   ▼                   ▼
         Gmail API          Google Calendar API   Weather API
```

---

## 🧩 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| LangGraph | Agent orchestration and state management |
| LangChain | LLM/tool integration |
| Groq | LLM inference |
| MCP | External tool/service communication |
| Streamlit | User interface |
| SQLite | Persistent LangGraph checkpoints |
| Gmail API | Email integration |
| Google Calendar API | Calendar integration |
| Weather API | Weather information |

---

## 🔄 How Aether Handles a Request

For a simple question:

```text
User
 ↓
Aether Agent
 ↓
Groq
 ↓
Final Answer
```

For a request requiring external information:

```text
User
 ↓
Aether Agent
 ↓
Groq
 ↓
Tool Decision
 ↓
MCP Client
 ↓
MCP Server
 ↓
External API
 ↓
Tool Result
 ↓
Aether Agent
 ↓
Final Answer
```

For an action requiring confirmation:

```text
User Request
      ↓
Agent Reasoning
      ↓
Tool Call Prepared
      ↓
Confirmation Required
      ↓
┌───────────────┐
│ User confirms │
└───────┬───────┘
        ↓
MCP Tool
        ↓
External Service
```

---

## 🔐 Safety

Aether separates **information retrieval** from **real-world actions**.

Read-only operations can be performed automatically when appropriate.

Actions that change external state require confirmation.

For example:

```text
"Show me my recent emails"
        ↓
       Read
        ↓
    No approval
```

Whereas:

```text
"Send an email to John"
        ↓
Prepare email
        ↓
Ask for confirmation
        ↓
User confirms
        ↓
Send email
```

This prevents the agent from independently performing consequential actions.

---

## 🧠 Memory Architecture

Aether uses two separate concepts:

### Conversation Registry

Stores conversation names and their associated `thread_id`s.

```text
Conversation Name
       ↓
   thread_id
```

### LangGraph Checkpoints

SQLite stores the actual conversation state associated with each thread.

```text
thread_id
   ↓
SQLite checkpoint
   ↓
Conversation state
```

Keeping these responsibilities separate allows conversations to be renamed or deleted without unnecessarily modifying the underlying checkpoint architecture.

---

## 📁 Project Structure

```text
personal-ai-assistant/
│
├── app.py
├── PROJECT_STATUS.md
├── conversations.json
├── checkpoints.db
├── requirements.txt
│
├── src/
│   ├── assistant/
│   │   ├── agent.py
│   │   └── conversations.py
│   │
│   └── mcp/
│       ├── client.py
│       ├── gmail_server.py
│       ├── calendar_server.py
│       └── weather_server.py
│
├── .streamlit/
│   ├── config.toml
│   └── styles.css
│
└── ...
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ashar-creator492/personal-ai-assistant.git
cd personal-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file containing the required credentials.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

Additional credentials are required for the Gmail and Google Calendar integrations.

**Never commit API keys, OAuth tokens, or secrets to GitHub.**

---

## 🔑 Google API Setup

Aether requires Google authentication for Gmail and Calendar functionality.

You need to configure:

- Google Cloud project
- Gmail API
- Google Calendar API
- OAuth credentials

The first authentication may require completing Google's OAuth consent flow in your browser.

---

## 🚀 Running Aether

Start the MCP servers in separate terminals.

### Weather

```bash
python -m src.mcp.weather_server
```

Runs on:

```text
http://127.0.0.1:8000
```

### Gmail

```bash
python -m src.mcp.gmail_server
```

Runs on:

```text
http://127.0.0.1:8001
```

### Calendar

```bash
python -m src.mcp.calendar_server
```

Runs on:

```text
http://127.0.0.1:8002
```

### Start the Streamlit application

```bash
streamlit run app.py
```

Aether will then be available through the Streamlit interface.

---

## 💡 Example Requests

### Gmail

```text
Show me my recent emails.
```

```text
Find emails from my professor about assignments.
```

```text
Draft an email to my professor asking for an extension.
```

### Calendar

```text
What events do I have coming up?
```

```text
Add a meeting with Ali tomorrow at 3 PM.
```

### Weather

```text
What's the weather today?
```

```text
Will it rain tomorrow?
```

### Multi-service reasoning

Aether can combine information from different services when required.

For example:

```text
Check my calendar for tomorrow and tell me whether
the weather will be suitable for my outdoor meeting.
```

The agent can determine that both Calendar and Weather information are required instead of treating the request as a simple keyword search.

---

## 🧪 Testing

The project includes checks for:

- Conversation persistence
- Conversation isolation
- Conversation renaming
- Duplicate conversation names
- Conversation deletion
- Gmail read operations
- Calendar read operations
- Weather operations
- Confirmation/cancellation behavior
- Streamlit application health

Write operations should always be tested carefully to avoid unintended external actions.

---

## 🛣️ Future Development

Planned improvements include:

- WhatsApp integration
- Incoming WhatsApp messages through webhooks
- Automated workflows with n8n
- More MCP servers and tools
- Improved long-term memory
- Additional agent capabilities
- Deployment
- Production authentication and infrastructure
- Expanded observability and monitoring

---

## 🎯 Why I Built This

I built Aether to explore what happens when an LLM moves beyond generating text and becomes an **agent capable of interacting with real-world services**.

The project allowed me to work with:

- Agentic AI
- LangGraph
- MCP
- Tool calling
- LLM orchestration
- Persistent state
- OAuth
- External APIs
- Safety/confirmation systems
- Streamlit application development

The goal was not simply to build another chatbot, but to understand how an AI system can **reason, use tools, maintain state, and safely perform actions**.

---

## 👨‍💻 Author

**Ashar**

Computer Science Student  
FAST-NUCES Islamabad

GitHub: [Ashar-creator492](https://github.com/Ashar-creator492)

---

## 📄 License

This project is intended primarily as a learning and portfolio project.
