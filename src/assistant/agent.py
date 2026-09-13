
import asyncio
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

from src.mcp.client import client

load_dotenv()


async def run_agent(question, llm_with_tools, tools):

    messages = [
        SystemMessage(
            content=(
                "You are Aether, a personal AI assistant. "
                "Use the available tools to complete the user's request. "
                "If you need another tool after receiving a tool result, "
                "call it. Continue until the task is complete. "
                "When you have enough information, directly answer the user."
            )
        ),
        HumanMessage(content=question)
    ]

    while True:

        response = await llm_with_tools.ainvoke(messages)

        if not response.tool_calls:
            return response.content

        messages.append(response)

        for tool_call in response.tool_calls:

            print(f"Using tool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

            # Ask for confirmation before sending an email
            if tool_call["name"] == "send_email":

                args = tool_call["args"]

                print("\nEmail ready to send:")
                print(f"To: {args['to']}")
                print(f"Subject: {args['subject']}")
                print(f"Body: {args['body']}")

                confirmation = input("\nSend this email? (yes/no): ")

                if confirmation.lower() != "yes":

                    print("Email cancelled.")

                    messages.append(
                        ToolMessage(
                            content="The user cancelled the email. Do not send it.",
                            tool_call_id=tool_call["id"],
                        )
                    )

                    continue

            tool = next(
                tool for tool in tools
                if tool.name == tool_call["name"]
            )

            tool_result = await tool.ainvoke(
                tool_call["args"]
            )

            print("Tool result:", tool_result)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )


async def main():

    tools = await client.get_tools()

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(tools)

    question = input("You: ")

    answer = await run_agent(
        question,
        llm_with_tools,
        tools
    )

    print("\nAssistant:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())

