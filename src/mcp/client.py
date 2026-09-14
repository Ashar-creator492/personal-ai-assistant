
import asyncio

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
        },
        "calendar": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8002/mcp",
        },
    }
)


async def main():
    tools = await client.get_tools()

    print("Available MCP tools:")

    for tool in tools:
        print(tool.name)

    calendar_tool = next(
        tool
        for tool in tools
        if tool.name == "get_upcoming_events"
    )

    result = await calendar_tool.ainvoke({"limit": 10})

    print("\nCalendar Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

