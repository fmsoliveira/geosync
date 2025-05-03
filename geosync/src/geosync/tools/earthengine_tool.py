from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict
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
    
    def get_nearest_image(self, collection: ee.ImageCollection, date: datetime.datetime, window: int = 30, after=True) -> ee.Image:
        if after:
            filtered = collection.filterDate(
                date.strftime('%Y-%m-%d'),
                (date + datetime.timedelta(days=window)).strftime('%Y-%m-%d')
            )
        else:
            filtered = collection.filterDate(
                (date - datetime.timedelta(days=window)).strftime('%Y-%m-%d'),
                date.strftime('%Y-%m-%d')
            )
        if filtered.size().getInfo() > 0:
            return filtered.sort("CLOUDY_PIXEL_PERCENTAGE").first()
        return None

    def create_quadrant_roi(self, lat: float, lon: float, scale: int, max_pixels: int = 100) -> List[ee.Geometry.Rectangle]:
        """
        Cria 4 ROIs correspondentes aos 4 quadrantes em torno das coordenadas dadas.
        
        - lat, lon: centro da área total
        - scale: resolução desejada em metros/pixel (ex: 10, 30, 100)
        - max_pixels: número máximo de pixels por lado para cada quadrante

        Returns:
            Lista de 4 ee.Geometry.Rectangle, um para cada quadrante
        """
        # Calcula o tamanho do lado da área segura em metros
        max_side_meters = scale * max_pixels  # Ex: 10 m * 5000 = 50 km lado máximo

        # Aproximação: 1 grau de latitude ~ 111.32 km
        degrees_per_meter = 1 / 111320.0  # Aproximadamente

        side_deg = max_side_meters * degrees_per_meter

        # Cria 4 quadrantes
        # Quadrante 1: Superior Direito (NE)
        q1 = ee.Geometry.Rectangle([lon, lat, lon + side_deg, lat + side_deg])
        
        # Quadrante 2: Superior Esquerdo (NO)
        q2 = ee.Geometry.Rectangle([lon - side_deg, lat, lon, lat + side_deg])
        
        # Quadrante 3: Inferior Esquerdo (SO)
        q3 = ee.Geometry.Rectangle([lon - side_deg, lat - side_deg, lon, lat])
        
        # Quadrante 4: Inferior Direito (SE)
        q4 = ee.Geometry.Rectangle([lon, lat - side_deg, lon + side_deg, lat])
        
        return [q1, q2, q3, q4]
        
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

    def process_quadrant_images(self, lat: float, lon: float, first_date: datetime.datetime, 
                               second_date: datetime.datetime, scale: int, max_pixels: int = 256) -> Dict:
        """
        Processa os 4 quadrantes e retorna os resultados.
        """
        quadrants = self.create_quadrant_roi(lat, lon, scale, max_pixels)
        quadrant_names = ["NE", "NO", "SO", "SE"]
        
        collection = self.get_image_collection(ee.Geometry.MultiPolygon(quadrants))

        #dates_window = second_date - first_date
        
        first_image = self.get_nearest_image(collection, first_date)
        second_image = self.get_nearest_image(collection, second_date)
        
        if not first_image:
            return {"error": f"No image found for start date: {first_date.date()}"}
        
        if not second_image:
            return {"error": f"No image found for end date: {second_date.date()}"}
            
        print("First image ID:", first_image.get("system:index").getInfo())
        print("Second image ID:", second_image.get("system:index").getInfo())
        
        results = {
            "first_date": {},
            "second_date": {}
        }
        
        # Process each quadrant for both dates
        for i, (quadrant, name) in enumerate(zip(quadrants, quadrant_names)):
            # First date
            first_url = self.get_image_url(first_image, quadrant, scale)
            first_filename = f"satellite_image_{lat}_{lon}_{first_date.strftime('%Y-%m-%d')}_{name}.zip"
            first_result = self.download_image_if_not_exists(first_url, first_filename)
            results["first_date"][name] = first_result
            
            # Second date
            second_url = self.get_image_url(second_image, quadrant, scale)
            second_filename = f"satellite_image_{lat}_{lon}_{second_date.strftime('%Y-%m-%d')}_{name}.zip"
            second_result = self.download_image_if_not_exists(second_url, second_filename)
            results["second_date"][name] = second_result
        
        return results

    def _run(self, lat: float, lon: float, first_date: str, second_date: str) -> str:
        """
        Process input data and fetch satellite images for 4 quadrants.
        """
        print("[**DEBUG**] EarthEngineImageFetcherTool _run called!", file=sys.stderr)
        logging.debug("[**DEBUG**] EarthEngineImageFetcherTool _run called!")
        
        print(f"Resultado depois do parse: {lat}, {lon}", file=sys.stderr)
        start_date = datetime.datetime.strptime(first_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(second_date, "%Y-%m-%d")

        GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if GOOGLE_APPLICATION_CREDENTIALS is None:
            return "Error: GOOGLE_APPLICATION_CREDENTIALS not found in environment variables"

        try:
            ee.Initialize(project="ee-syncearth")
            logging.debug("Autenticação GEE bem-sucedida via Conta de Serviço (env).")
        except Exception as e:
            return f"Failed to initialize Earth Engine: {str(e)}"
        
        try:
            # Define a resolução desejada
            scale = 10  # metros por pixel
            max_pixels = 256  # pixels por lado

            # Processa os 4 quadrantes
            results = self.process_quadrant_images(lat, lon, start_date, end_date, scale, max_pixels)
            
            if "error" in results:
                return results["error"]

            print("[**DEBUG**] _run INPUTS:", lat, lon, first_date, second_date, file=sys.stderr)
            logging.debug(f"[**DEBUG**] _run INPUTS: {lat}, {lon}, {first_date}, {second_date}")

            print("[**DEBUG**] _run OUTPUTS:", results, file=sys.stderr)
            logging.debug(f"[**DEBUG**] _run OUTPUTS: {results}")
            
            return json.dumps({
                "first_date_images": results["first_date"],
                "second_date_images": results["second_date"]
            })
        except Exception as e:
            return f"Error processing quadrants: {str(e)}"