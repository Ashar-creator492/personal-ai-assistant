import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    }
)


async def main():
    tools = await client.get_tools()

    print("Available MCP tools:")

    for tool in tools:
        print(tool.name)

    result = await tools[0].ainvoke({
        "city": "Islamabad"
    })

    print("\nResult:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())