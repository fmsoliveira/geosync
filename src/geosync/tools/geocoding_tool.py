import requests
from pydantic import BaseModel
from typing import Type
from crewai.tools import BaseTool

class GeocodeInput(BaseModel):
    address: str

class GeoapifyTool(BaseTool):
    name: str = "Geoapify Geocoder"
    description: str = "Converts a physical address into latitude and longitude coordinates using the Geoapify API."
    args_schema: Type[BaseModel] = GeocodeInput

    def _run(self, address: str) -> str:
        import os
        api_key = os.environ.get("GEOAPIFY_KEY")
        if not api_key:
            return "Geoapify API key not found. Please set GEOAPIFY_KEY environment variable."

        url = "https://api.geoapify.com/v1/geocode/search"
        params = {
            "text": address,
            "apiKey": api_key,
            "format": "json"
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"Error contacting Geoapify: {response.text}"

        data = response.json()
        if not data.get("features"):
            return "Address not found."

        coords = data["features"][0]["geometry"]["coordinates"]
        formatted = data["features"][0]["properties"]["formatted"]

        return f"Address: {formatted}\nLatitude: {coords[1]}\nLongitude: {coords[0]}"