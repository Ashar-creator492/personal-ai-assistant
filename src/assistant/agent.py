import asyncio
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.mcp.client import client

from langchain_core.messages import ToolMessage

load_dotenv()


async def run_agent(question, llm_with_tools, tools):

    response = await llm_with_tools.ainvoke(question)

    if response.tool_calls:

        tool_messages = []

        for tool_call in response.tool_calls:

            tool = next(
                tool for tool in tools
                if tool.name == tool_call["name"]
            )

            tool_result = await tool.ainvoke(
                tool_call["args"]
            )

            tool_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

        final_response = await llm_with_tools.ainvoke(
            [response] + tool_messages
        )

        return final_response.content

    return response.content



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