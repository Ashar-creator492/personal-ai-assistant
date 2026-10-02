import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
)


from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.mcp.client import client
from src.assistant.conversations import ConversationManager

load_dotenv()


SYSTEM_PROMPT = """
You are Aether, a personal AI assistant.

You have access to external services through MCP tools.

Your job is to understand the user's intent and decide which tools
are necessary to answer the request.

GENERAL RULES:

1. Understand the user's actual goal before choosing a tool.
2. Use the tool whose purpose best matches the information you need.
3. You may call multiple tools when a task requires information
   from multiple services.
4. After receiving a tool result, inspect it and decide whether
   another tool is actually necessary.
5. Do not repeatedly search for the same information.
6. Do not search for literal keywords when a better tool strategy
   exists.
7. Do not call unrelated tools.
8. Do not invent information that was not returned by a tool.
9. When information from multiple tools is relevant, reason across
   the results before answering.
10. Once you have enough information, stop using tools and answer
    the user directly.

GMAIL:

Use Gmail tools when the user's request involves their emails,
messages, contacts, meetings, appointments, classes, events,
or information contained in their mailbox.

When a search result identifies a potentially relevant email,
use get_email to inspect that email before making conclusions
about its contents.

Do not assume that an email is relevant merely because a search
keyword appears in its subject.

When find_upcoming_events returns a genuine upcoming event and
the user has asked you to add, schedule, or put that event on
their calendar, use create_calendar_event with the extracted
title, date, time, location, and description.

Do not create a calendar event merely because an event was found
in an email. Only create it when the user's request indicates
that they want it added to the calendar.

Before creating the event, use the information returned by
find_upcoming_events. Do not invent missing event details.


CALENDAR:

When creating calendar events, all date and time values must use
the user's local timezone, Asia/Karachi.

Use ISO 8601 datetime values with the +05:00 offset.

Never convert calendar event times to UTC (+00:00).

For example, 6:00 PM in Pakistan should be represented as:
2026-09-14T18:00:00+05:00

If an event has a start time but no end time or duration,
automatically use a default duration of 2 hours.

If the email provides an explicit end time or duration, always
use the information from the email instead of the default.

Never ask the user for the duration when it is missing. Use the
2-hour default automatically.

WEATHER:

Use weather tools when the user's request requires current or
forecast weather information.

If the user asks a question requiring both Gmail information and
weather information, you may use both services and combine their
results.

IMPORTANT:

You are an agent, not a simple keyword search system.

Do not interpret the user's request as:
"find emails containing the word X."

Instead determine:
"What information does the user actually need?"

Keep tool usage efficient because external tool results consume
context.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    approvals: dict


def calendar_args_in_local_time(args):
    """Calendar tool times are local wall times, even if the model adds another offset."""
    local_args = dict(args)
    for field in ("start_time", "end_time"):
        value = local_args.get(field)
        if not value:
            continue
        try:
            local_time = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        local_args[field] = local_time.replace(tzinfo=ZoneInfo("Asia/Karachi")).isoformat()
    return local_args


def create_aether_node(llm_with_tools):

    async def aether_node(state: AgentState):

        tool_results = []
        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                break
            tool_results.append(message)
        if tool_results and all(message.content.startswith("The user cancelled this action.")
                                for message in tool_results):
            return {"messages": [AIMessage(content="Cancelled.")]}

        response = await llm_with_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        response.additional_kwargs["aether_time"] = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%I:%M %p")

        return {
            "messages": [response]
        }

    return aether_node


def create_tool_node(tools, progress=None):

    async def tool_node(state: AgentState):

        last_message = state["messages"][-1]

        tool_messages = []

        for tool_call in last_message.tool_calls:

            tool_name = tool_call["name"]
            tool_args = (calendar_args_in_local_time(tool_call["args"])
                         if tool_name == "create_calendar_event" else tool_call["args"])

            if tool_name in {"send_email", "create_calendar_event"} and not state.get("approvals", {}).get(tool_call["id"]):
                tool_messages.append(ToolMessage(
                    content="The user cancelled this action. Do not perform it.",
                    tool_call_id=tool_call["id"],
                ))
                continue

            if progress:
                progress(tool_name)

            # --------------------------------------------------
            # FIND MCP TOOL
            # --------------------------------------------------

            tool = next(
                (
                    tool
                    for tool in tools
                    if tool.name == tool_name
                ),
                None,
            )

            if tool is None:

                tool_messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' was not found.",
                        tool_call_id=tool_call["id"],
                    )
                )

                continue

            # --------------------------------------------------
            # EXECUTE TOOL
            # --------------------------------------------------

            try:

                tool_result = await tool.ainvoke(tool_args)

                result_text = str(tool_result)

                MAX_TOOL_RESULT = 6000

                if len(result_text) > MAX_TOOL_RESULT:

                    result_text = (
                        result_text[:MAX_TOOL_RESULT]
                        + "\n[Tool result truncated]"
                    )

                tool_messages.append(
                    ToolMessage(
                        content=result_text,
                        tool_call_id=tool_call["id"],
                    )
                )

            except Exception as e:

                tool_messages.append(
                    ToolMessage(
                        content=f"Tool execution failed: {str(e)}",
                        tool_call_id=tool_call["id"],
                    )
                )

        return {
            "messages": tool_messages
        }

    return tool_node


def approval_node(state: AgentState):
    pending = [
        {"id": call["id"], "name": call["name"],
         "args": calendar_args_in_local_time(call["args"])
         if call["name"] == "create_calendar_event" else call["args"]}
        for call in state["messages"][-1].tool_calls
        if call["name"] in {"send_email", "create_calendar_event"}
    ]
    return {"approvals": interrupt(pending) if pending else {}}


def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


async def run_agent(question, llm_with_tools, tools):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    max_steps = 8

    for step in range(max_steps):

        response = await llm_with_tools.ainvoke(messages)

        # --------------------------------------------------
        # FINAL ANSWER
        # --------------------------------------------------

        if not response.tool_calls:
            return response.content

        # Keep the assistant's tool-call message
        messages.append(response)

        # --------------------------------------------------
        # TOOL CALLS
        # --------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = (calendar_args_in_local_time(tool_call["args"])
                         if tool_name == "create_calendar_event" else tool_call["args"])

            print(f"\nUsing tool: {tool_name}")
            print(f"Arguments: {tool_args}")

            # --------------------------------------------------
            # SEND EMAIL CONFIRMATION
            # --------------------------------------------------

            if tool_name == "send_email":

                print("\nEmail ready to send:")
                print(f"To: {tool_args['to']}")
                print(f"Subject: {tool_args['subject']}")
                print(f"Body: {tool_args['body']}")

                confirmation = input(
                    "\nSend this email? (yes/no): "
                ).strip().lower()

                if confirmation != "yes":

                    print("Email cancelled.")

                    messages.append(
                        ToolMessage(
                            content=(
                                "The user cancelled the email. "
                                "Do not send it."
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )

                    continue

            # --------------------------------------------------
            # CALENDAR EVENT CONFIRMATION
            # --------------------------------------------------

            if tool_name == "create_calendar_event":

                print("\nCalendar event ready to create:")
                print(f"Title: {tool_args['title']}")
                print(f"Start: {tool_args['start_time']}")
                print(f"End: {tool_args['end_time']}")

                if tool_args.get("location"):
                    print(f"Location: {tool_args['location']}")

                if tool_args.get("description"):
                    print(f"Description: {tool_args['description']}")

                confirmation = input(
                    "\nAdd this event to your calendar? (yes/no): "
                ).strip().lower()

                if confirmation != "yes":

                    print("Calendar event cancelled.")

                    messages.append(
                        ToolMessage(
                            content=(
                                "The user cancelled the calendar event. "
                                "Do not create it."
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )

                    continue

            # --------------------------------------------------
            # FIND MCP TOOL
            # --------------------------------------------------

            tool = next(
                (
                    tool
                    for tool in tools
                    if tool.name == tool_name
                ),
                None,
            )

            if tool is None:

                messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' was not found.",
                        tool_call_id=tool_call["id"],
                    )
                )

                continue

            # --------------------------------------------------
            # EXECUTE TOOL
            # --------------------------------------------------

            try:

                tool_result = await tool.ainvoke(tool_args)

                print("Tool result:")
                print(tool_result)

                # Keep tool output reasonably sized so the model
                # does not hit Groq's context/TPM limits.
                result_text = str(tool_result)

                MAX_TOOL_RESULT = 6000

                if len(result_text) > MAX_TOOL_RESULT:

                    result_text = (
                        result_text[:MAX_TOOL_RESULT]
                        + "\n[Tool result truncated]"
                    )

                messages.append(
                    ToolMessage(
                        content=result_text,
                        tool_call_id=tool_call["id"],
                    )
                )

            except Exception as e:

                print(f"Tool error: {e}")

                messages.append(
                    ToolMessage(
                        content=f"Tool execution failed: {str(e)}",
                        tool_call_id=tool_call["id"],
                    )
                )

    return (
        "I could not complete the request within the allowed "
        "number of tool steps."
    )



def create_conversation(name):
    return ConversationManager().create(name)


def build_graph(llm_with_tools, tools, checkpointer, progress=None):
    graph = StateGraph(AgentState)
    graph.add_node("aether", create_aether_node(llm_with_tools))
    graph.add_node("approval", approval_node)
    graph.add_node("tools", create_tool_node(tools, progress))
    graph.add_edge(START, "aether")
    graph.add_conditional_edges("aether", should_continue, {"tools": "approval", END: END})
    graph.add_edge("approval", "tools")
    graph.add_edge("tools", "aether")
    return graph.compile(checkpointer=checkpointer)


async def invoke_agent(thread_id, message=None, approvals=None, progress=None):
    """Run one turn or resume a paused write operation."""
    tools = await client.get_tools()
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0,
                   api_key=os.getenv("GROQ_API_KEY"))
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = build_graph(llm.bind_tools(tools), tools, checkpointer, progress)
        config = {"configurable": {"thread_id": thread_id}}
        if approvals is not None:
            return await app.ainvoke(Command(resume=approvals), config=config)
        human = HumanMessage(content=message, additional_kwargs={
            "aether_time": datetime.now(ZoneInfo("Asia/Karachi")).strftime("%I:%M %p")
        })
        return await app.ainvoke({"messages": [human]}, config=config)


async def conversation_state(thread_id):
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        # History loading does not require live MCP servers or an LLM.
        app = build_graph(None, [], checkpointer)
        return await app.aget_state({"configurable": {"thread_id": thread_id}})




async def main():

    # --------------------------------------------------
    # LOAD MCP TOOLS
    # --------------------------------------------------

    tools = await client.get_tools()

    print("Available MCP tools:")

    for tool in tools:
        print(f"- {tool.name}")

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------
    # LANGGRAPH
    # --------------------------------------------------

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = build_graph(llm_with_tools, tools, checkpointer)

        # --------------------------------------------------
        # USER INPUT
        # --------------------------------------------------

        conversations = ConversationManager()
        while True:
            saved = conversations.list()
            print("\nConversations:")
            for name in saved:
                print(f"- {name}")
            choice = input("Name to open, or 'new', 'rename', 'delete', 'quit': ").strip()
            if choice.lower() in {"quit", "exit"}:
                return
            try:
                if choice.lower() == "new":
                    conversation_name = input("New conversation name: ").strip()
                    thread_id = conversations.create(conversation_name)
                    break
                if choice.lower() == "rename":
                    conversations.rename(input("Existing name: ").strip(), input("New name: ").strip())
                    continue
                if choice.lower() == "delete":
                    name = input("Conversation to delete: ").strip()
                    if input(f"Delete '{name}' from the list? (yes/no): ").strip().lower() == "yes":
                        conversations.delete(name)
                    continue
                thread_id = conversations.get(choice)
                break
            except (KeyError, ValueError) as error:
                print(f"Conversation error: {error}")

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        while True:

            question = input("\nYou: ")

            if question.lower() in {"exit", "quit"}:
                break

            result = await app.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=question)
                    ]
                },
                config=config
            )

            while "__interrupt__" in result:
                pending = result["__interrupt__"][0].value
                approvals = {}
                for action in pending:
                    print(f"\nConfirm {action['name']}: {action['args']}")
                    approvals[action["id"]] = input("Proceed? (yes/no): ").strip().lower() == "yes"
                result = await app.ainvoke(Command(resume=approvals), config=config)

            print("\nAssistant:")
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
