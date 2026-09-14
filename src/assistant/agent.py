import asyncio
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    ToolMessage,
    SystemMessage,
    HumanMessage,
)

from src.mcp.client import client

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
            tool_args = tool_call["args"]

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
    # USER INPUT
    # --------------------------------------------------

    question = input("\nYou: ")

    answer = await run_agent(
        question,
        llm_with_tools,
        tools,
    )

    print("\nAssistant:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
