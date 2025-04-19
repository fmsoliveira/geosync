import requests
from pydantic import BaseModel
from typing import Type
from crewai.tools import BaseTool
import json

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
            return json.dumps({"error": "Geoapify API key not found."})
        
        url = "https://api.geoapify.com/v1/geocode/search"
        params = {
            "text": address,
            "apiKey": api_key,
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            return json.dumps({"error": "Geoapify API error."})
        
        data = response.json()
        
        # Verifica se há resultados em vez de features
        if not data.get("results") or len(data["results"]) == 0:
            return json.dumps({"error": "Address not found."})
        
        # Extrai as coordenadas do primeiro resultado
        first_result = data["results"][0]
        lat = first_result["lat"]
        lon = first_result["lon"]
        formatted = first_result["formatted"]

        return {
            "lat": lat,
            "lon": lon#,
            #"formatted_address": formatted
        }