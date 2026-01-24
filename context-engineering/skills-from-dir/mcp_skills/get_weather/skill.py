def _get_weather(city: str) -> str:
    return f"Weather in {city}: 25°C, sunny"

TOOL_METADATA = {
    "name": "get_weather",
    "func": _get_weather,
    "description": "Get current weather for a city",
    "parameters": {"city": "string"}
}