from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import datetime
import os
import requests
import ee
import logging
import json
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("meu_log.log"),
        logging.StreamHandler()  # Mostra no terminal também
    ]
)

class SatelliteImageFetcherInput(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    first_date: str = Field(..., description="Start date for image acquisition (YYYY-MM-DD)")
    second_date: str = Field(..., description="End date for image acquisition (YYYY-MM-DD)")

class EarthEngineImageFetcherTool(BaseTool):
    name: str = "EarthEngineImageFetcher"
    description: str = "Obtém imagens de satélite utilizando coordenadas (latitude e longitude)"
    args_schema: Type[BaseModel] = SatelliteImageFetcherInput

    def download_image_if_not_exists(self, url: str, filename: str) -> str:
        """Download an image if it doesn't exist locally."""
        # Check if the directory exists, create if not
        mkdir_result = os.makedirs("raw_images", exist_ok=True)
        print("Directory created: raw_images", mkdir_result, file=sys.stderr)

        # Format the full filepath
        filepath = os.path.join("raw_images", filename)
        
        # Check if the image already exists
        if os.path.exists(filepath):
            return f"Image already exists: {filepath}"

        # Download the image
        print("Downloading image...", file=sys.stderr)
        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                return f"Failed to download image: {response.status_code}"

            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        except Exception as e:
            return f"Error downloading image: {str(e)}"
    
    def get_image_collection(self, ROI: ee.Geometry) -> ee.ImageCollection:
        # collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        #         .filterBounds(ROI) \
        #         .filterDate(image_date.strftime('%Y-%m-%d'), (image_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')) \
        #         .sort("CLOUDY_PIXEL_PERCENTAGE")
        collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(ROI)
        return collection
        
    def get_image_url(self, image: ee.Image, ROI: ee.Geometry, scale: int = 100) -> str:
        return image.getDownloadURL({
            'scale': scale,
            'region': ROI,
            #'format': 'GEO_TIFF'
            'crs': 'EPSG:4326'
        })
    
    def get_nearest_image(self, collection: ee.ImageCollection, date: datetime.datetime, window: int = 7) -> ee.Image:
        for i in range(window):
            for delta in [-i, i]:
                candidate_date = date + datetime.timedelta(days=delta)
                filtered = collection.filterDate(
                    candidate_date.strftime('%Y-%m-%d'),
                    (candidate_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                )
                if filtered.size().getInfo() > 0:
                    return filtered.sort("CLOUDY_PIXEL_PERCENTAGE").first()
        return None

    def create_safe_roi(self, lat: float, lon: float, scale: int, max_pixels: int = 100):
        """
        Cria uma ROI dinâmica centrada nas coordenadas dadas, com dimensões adaptadas à escala,
        para que a imagem final não exceda o tamanho máximo em pixels.
        
        - lat, lon: centro da ROI
        - scale: resolução desejada em metros/pixel (ex: 10, 30, 100)
        - max_pixels: número máximo de pixels por lado (ex: 5000 * 5000 = 25M pixels máx)

        Returns:
            ee.Geometry.Rectangle
        """
        # Calcula o tamanho do lado da área segura em metros
        max_side_meters = scale * max_pixels  # Ex: 10 m * 5000 = 50 km lado máximo

        # Aproximação: 1 grau de latitude ~ 111.32 km
        degrees_per_meter = 1 / 111320.0  # Aproximadamente

        half_side_deg = (max_side_meters / 2.0) * degrees_per_meter

        return ee.Geometry.Rectangle([
            lon - half_side_deg,
            lat - half_side_deg,
            lon + half_side_deg,
            lat + half_side_deg
        ])

    def _run(self, lat: float, lon: float, first_date: str, second_date: str) -> str:
        """
        Process input data and fetch satellite images.
        
        Expected input formats:
        1. Direct format: "lat,lon[,start_date][,end_date]"
        2. JSON object: {"input_data": "lat,lon[,start_date][,end_date]"}
        3. Array of objects: [{"latlon": "lat,lon", "date": "YYYY-MM-DD"}, ...]
        """
        
        #print(f"Resultado depois do parse: {lat}, {lon}, {formatted_address}", file=sys.stderr)
        print(f"Resultado depois do parse: {lat}, {lon}", file=sys.stderr)
        start_date = datetime.datetime.strptime(first_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(second_date, "%Y-%m-%d")

        # # Define o intervalo de datas
        # date_now = datetime.datetime.now()
        # start_date = date_now - datetime.timedelta(days=7)
        # end_date = date_now

        GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if GOOGLE_APPLICATION_CREDENTIALS is None:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS not found in environment variables"

        try:
            #ee.Initialize(project_id="ee-geosync-crewai")
            ee.Initialize(project="ee-syncearth")
            logging.debug("Autenticação GEE bem-sucedida via Conta de Serviço (env).")
        except Exception as e:
            return f"Failed to initialize Earth Engine: {str(e)}"
        
        try:
            # Define a resolução desejada
            scale = 100  # metros por pixel

            # Define o ROI com base na escala
            ROI = self.create_safe_roi(lat, lon, scale)

            # Get the image collections on the start and end dates
            #first_collection = self.get_image_collection(ROI, start_date)
            #second_collection = self.get_image_collection(ROI, end_date)
            collection = self.get_image_collection(ROI)

            first_image = self.get_nearest_image(collection, start_date)
            second_image = self.get_nearest_image(collection, end_date)
            
            if not first_image:
                return f"No image found for start date: {start_date.date()}"
            
            if not second_image:
                return f"No image found for end date: {end_date.date()}"

            print("Trying to get images...")
            
            # Get the image URLs
            first_image_url = self.get_image_url(first_image, ROI, scale)
            second_image_url = self.get_image_url(second_image, ROI, scale)
            
            print("First image URL:", first_image_url, file=sys.stderr)
            print("Second image URL:", second_image_url, file=sys.stderr)

            # Get the image filenames
            first_image_filename = f"satellite_image_{lat}_{lon}_{start_date.strftime('%Y-%m-%d')}.zip"
            second_image_filename = f"satellite_image_{lat}_{lon}_{end_date.strftime('%Y-%m-%d')}.zip"
            
            # Download the images
            first_image_result = self.download_image_if_not_exists(first_image_url, first_image_filename)
            second_image_result = self.download_image_if_not_exists(second_image_url, second_image_filename)
            
            return {
                "sat_image_path_1": first_image_result,
                "sat_image_path_2": second_image_result
            }
        except Exception as e:
            return f"Error getting download URL or fetching image: {str(e)}"