from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Weather Server")


@mcp.tool()
def get_weather(city: str):
    """Get the current weather for a city."""

    # 1. Convert city name to coordinates
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }
    )

    geo_data = geo_response.json()

    if "results" not in geo_data:
        return f"Could not find city: {city}"

    latitude = geo_data["results"][0]["latitude"]
    longitude = geo_data["results"][0]["longitude"]

    # 2. Get current weather
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh"
        }
    )

    weather_data = weather_response.json()
    current = weather_data["current"]

    # 3. Return useful information
    return {
        "city": city,
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "wind_speed": current["wind_speed_10m"]
    }



@mcp.tool()
def get_forecast(city: str, days: int = 3):
    """Get the weather forecast for a city."""

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }
    )

    geo_data = geo_response.json()

    if "results" not in geo_data:
        return f"Could not find city: {city}"

    latitude = geo_data["results"][0]["latitude"]
    longitude = geo_data["results"][0]["longitude"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": days,
            "temperature_unit": "celsius"
        }
    )

    weather_data = weather_response.json()

    return {
        "city": city,
        "dates": weather_data["daily"]["time"],
        "max_temperature": weather_data["daily"]["temperature_2m_max"],
        "min_temperature": weather_data["daily"]["temperature_2m_min"],
        "rain_probability": weather_data["daily"]["precipitation_probability_max"]
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")