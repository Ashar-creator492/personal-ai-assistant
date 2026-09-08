from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Server")

@mcp.tool()
def get_weather(city: str):
    """Get the current weather for a city."""
    
    return f"Weather requested for {city}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    