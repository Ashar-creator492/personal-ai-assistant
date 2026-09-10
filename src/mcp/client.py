import asyncio
from unittest import result

from langchain_mcp_adapters.client import MultiServerMCPClient


client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8000/mcp",
        },
        "gmail": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8001/mcp",
        }
    }
)


async def main():
    tools = await client.get_tools()

    print("Available MCP tools:")

    for tool in tools:
        print(tool.name)

    gmail_tool = next(
    tool for tool in tools
    if tool.name == "get_recent_emails"
)

    result = await gmail_tool.ainvoke({"limit": 5})

    print("\nGmail Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())